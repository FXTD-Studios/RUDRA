You may download single files directly through your web browser on the
download.opencontent.netflix.com page, but for large files and long frame sequences,
you may wish to use command line tools such as aws cli. Instructions are posted here:
https://docs.aws.amazon.com/cli/latest/userguide/installing.html

After installation, you should be able to download the public assets. Try
running these sample commands. If the download is interrupted, you can run the
same command immediately and aws cli will resume the download where it left off.

Download a single file (0.7 kB):
Usage:   aws s3 cp --no-sign-request <s3 URI> <local destination>
Example: aws s3 cp --no-sign-request s3://download.opencontent.netflix.com/Netflix_test_conditions/ \
Chimera-Aerial_3840x2160_2997fps_10bit_420.yuv .

Sync entire directory (~100 GB):
Usage:   aws s3 sync --no-sign-request <s3 URI> <local destination>
Example: aws s3 sync --no-sign-request s3://download.opencontent.netflix.com/aom_test_materials/ .

***

Sparks
Natural Video, Cinematic, HDR
4096x2160
P3-D65
12
59.94 fps
Creative Commons, Attribution-NonCommercial-NoDerivatives 4.0
http://download.opencontent.netflix.com.s3.amazonaws.com/TechblogAssets/Sparks/sparks_license.txt

Meridian
Natural Video
Cinematic, HDR
3840x2160
P3-D65
12
59.94 fps
Creative Commons, Attribution-NonCommercial-NoDerivatives 4.0 International
http://download.opencontent.netflix.com.s3.amazonaws.com/Meridian/MERIDIAN_SHR_C_EN-XX_US-NR_51_LTRT_UHD_ML7_SL4_20160915_OV.zip

Sol Levante
Animation, Anime, HDR
3840x2160
P3-D65
12
24.00 fps
Creative Commons, Attribution-NonCommercial-NoDerivatives 4.0 International
http://download.opencontent.netflix.com.s3.amazonaws.com/SolLevante/sollevante-cclicense.txt

Cosmos Laundromat
Animation, HDR
3840x2160
P3-D65
12
24.00 fps
Creative Commons Attribution 3.0
https://gooseberry.blender.org/license/

Nocturne
Natural Video,
Cinematic, HDR
3840x2160
P3-D65
12
60.00 fps
Creative Commons, Attribution-NonCommercial-NoDerivatives 4.0 International
http://download.opencontent.netflix.com.s3.amazonaws.com/Nocturne/nocturne_license.txt
