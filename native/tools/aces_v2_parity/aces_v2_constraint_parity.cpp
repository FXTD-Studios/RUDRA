#include <cstdio>
#include <cstdint>
#include <fstream>
#include <vector>
#include "rudra/core/aces_v2.hpp"
#include "rudra/infer/aces_v2_runtime.hpp"
using namespace rudra; using namespace rudra::aces_v2;
static PlanarBuffer rd(const char* p) { std::ifstream f(p, std::ios::binary); int32_t s[3]; f.read((char*)s, 12);
  std::vector<float> v((size_t)s[0]*s[1]*s[2]); f.read((char*)v.data(), v.size()*4);
  PlanarBuffer b(s[2], s[0], s[1]); for (int c=0;c<s[2];++c) for (size_t i=0;i<b.plane_size();++i) b.plane(c)[i]=v[i*s[2]+c]; return b; }
int main(int, char** a) {
  auto src=rd(a[1]), pred=rd(a[2]), base=rd(a[3]), w=rd(a[4]); auto view=make_ocio_view(1);
  ConstraintStats cs; SmoothingStats ss;
  auto sm = smooth_clipped_hybrid(pred, src, 64.0, &ss);
  auto out=constrain_colour_v2(src, sm, base, w, **view, true, &cs);
  std::ofstream f(a[5], std::ios::binary); int32_t s[3]={out.height(),out.width(),3}; f.write((char*)s,12);
  std::vector<float> v(out.plane_size()*3); for(int c=0;c<3;++c) for(size_t i=0;i<out.plane_size();++i) v[i*3+c]=out.plane(c)[i];
  f.write((char*)v.data(), v.size()*4);
  std::printf("%zu %zu %zu %zu comps %zu\n", cs.support_pixels, cs.changed_pixels, cs.baseline_fallback_pixels, cs.remaining_display_errors, ss.components);
}
