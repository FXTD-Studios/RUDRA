import numpy as np
from PIL import Image, ImageFilter
S=2048  # supersampled canvas; final 1024
c=S/2
yy,xx=np.mgrid[0:S,0:S].astype(np.float32)+0.5
def aa(d, w=1.5):  # signed distance (neg inside) -> coverage
    return np.clip(0.5 - d/w, 0, 1)
def rgba(col, a): 
    out=np.zeros((S,S,4),np.float32); out[...,:3]=col; out[...,3]=a; return out
def over(dst, src):  # premult-free 'over'
    a=src[...,3:4]; dst[...,:3]=src[...,:3]*a+dst[...,:3]*(1-a); dst[...,3:4]=a+dst[...,3:4]*(1-a); return dst
def hexc(h): h=h.lstrip('#'); return np.array([int(h[i:i+2],16)/255 for i in (0,2,4)],np.float32)
def lerp(a,b,t): return a+(b-a)*t[...,None]
def blur(layer, r):
    im=Image.fromarray((np.clip(layer,0,1)*255).astype(np.uint8),'RGBA').filter(ImageFilter.GaussianBlur(r))
    return np.asarray(im).astype(np.float32)/255

def squircle_d(half, n=5):
    x=np.abs(xx-c)/half; y=np.abs(yy-c)/half
    f=(x**n+y**n)**(1/n)
    return (f-1)*half  # approx distance

def tile(base, style):
    half=S*0.80/2
    d=squircle_d(half)
    cov=aa(d,2)
    # shadow
    sh=np.zeros((S,S,4),np.float32); sd=squircle_d(half*1.0); 
    shm=aa(squircle_d(half*0.98)+0, 2)
    sh[...,3]=shm*0.55
    sh=np.roll(sh, int(S*0.022), axis=0); sh=blur(sh, S*0.02)
    over(base, sh)
    t=np.clip((yy-(c-half))/(2*half),0,1)
    if style=='graphite':
        col=lerp(hexc('#2c2c2f'),hexc('#121214'),t)
    else:  # black glass
        col=lerp(hexc('#17181c'),hexc('#060608'),t)
    layer=np.concatenate([col,cov[...,None]],-1); over(base,layer)
    # specular top
    spec=np.clip(1-(yy-(c-half))/(half*0.9),0,1)**2*0.10
    over(base, rgba(np.ones(3,np.float32), spec*cov))
    # rim hairline top-lit
    ring=aa(np.abs(d)-S*0.0012,2)*np.clip(1-(yy-(c-half))/(half*1.1),0,1)*0.45
    over(base, rgba(np.ones(3,np.float32), ring*cov))
    return base

def eclipse(base, R=S*0.215, glow=True, mono=False):
    r=np.hypot(xx-c,yy-c)
    disc=aa(r-R,2)
    left=np.clip(0.5-(xx-c)/1.5,0,1)  # x<c
    right=1-left
    # warm half: light emanating from the seam
    dx=np.abs(xx-c)/R
    t=np.clip(dx,0,1)
    warm=lerp(hexc('#fffaf0'),hexc('#ffb340'),np.clip(t*1.1,0,1)**0.8)
    warm=lerp(warm,hexc('#e8742a'),np.clip((t-0.7)/0.3,0,1)*0.6)
    if mono: warm=lerp(np.ones(3,np.float32)*1.0, np.ones(3,np.float32)*0.78, t)
    if glow:
        g=np.zeros((S,S,4),np.float32); g[...,:3]=hexc('#ffb340'); g[...,3]=disc*left*0.55
        over(base, blur(g, S*0.03))
        g2=np.zeros((S,S,4),np.float32); g2[...,:3]=hexc('#6e7bff'); 
        rimb=aa(np.abs(r-R)-S*0.004,3)*right
        g2[...,3]=rimb*0.6; over(base, blur(g2, S*0.012))
    over(base, np.concatenate([warm,(disc*left)[...,None]],-1))
    # dark half
    dark=hexc('#07080b')
    over(base, rgba(dark, disc*right))
    # cool rim on the dark half: thin, strongest at the far edge, fading to the seam
    ang=np.clip((xx-c)/R,0,1)
    rimc=lerp(hexc('#5e6cff'),hexc('#b86cff'),np.clip((yy-c)/R*0.5+0.5,0,1))
    if mono: rimc=np.ones((S,S,3),np.float32)*0.85
    rim=aa(np.abs(r-(R-S*0.004))-S*0.0035,2)*right*np.clip(ang*1.6,0,1)
    over(base, np.concatenate([rimc,rim[...,None]],-1))
    # the seam: a hairline of light, a little longer than the disc, tapered
    L=R*1.32
    w=S*0.0035
    taper=np.clip(1-np.abs(yy-c)/L,0,1)**0.7
    seam=aa(np.abs(xx-c)-w*taper,1.5)*(np.abs(yy-c)<L)
    sc=hexc('#fffdf6')
    if glow:
        sg=np.zeros((S,S,4),np.float32); sg[...,:3]=hexc('#ffd28a'); sg[...,3]=seam*0.9
        over(base, blur(sg, S*0.008))
    over(base, rgba(sc, seam*taper**0.3))
    return base

def save(base, name, size=1024):
    im=Image.fromarray((np.clip(base,0,1)*255).astype(np.uint8),'RGBA').resize((size,size),Image.LANCZOS)
    im.save(name); return im

for style in ('graphite','glass'):
    b=np.zeros((S,S,4),np.float32); tile(b,style); eclipse(b); save(b,f'icon_{style}.png')
# mark alone on transparent (for splash / web)
b=np.zeros((S,S,4),np.float32); eclipse(b,R=S*0.30); save(b,'mark.png')
# mono / small-size glyph (template)
b=np.zeros((S,S,4),np.float32); tile(b,'graphite'); eclipse(b,glow=False,mono=True); save(b,'icon_mono.png')
print('ok')
