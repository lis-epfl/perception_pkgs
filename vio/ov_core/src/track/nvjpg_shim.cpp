// ---------------------------------------------------------------------------------------
// OV_NVJPG: isolated NVJPG hardware-decode shim.  BUILT INTO ITS OWN libov_nvjpg.so.
//
// THIS TRANSLATION UNIT MUST NEVER SEE /usr/include/jpeglib.h AND MUST NEVER SEE OpenCV.
// Jetson's libnvjpeg.so is a full libjpeg FORK: it exports all 116 jpeg_* symbols, and
// NVIDIA's own jpeglib.h (include/libjpeg-8b) inserts 14 extra members into the
// jpeg_common_fields macro -- the prefix shared by jpeg_{de,}compress_struct -- so every
// field from `client_data` onwards sits at a different offset than stock libjpeg's.
// Mixing the two in one link scope corrupts cinfo in WHICHEVER direction binds wrong.
// The isolation is: this .so is dlopen'd RTLD_LOCAL|RTLD_DEEPBIND by the estimator, so
//   - libnvjpeg's jpeg_* never enter the global scope   -> cv::imdecode keeps libjpeg.so.8
//   - our NvJpegDecoder.cpp's jpeg_* bind DEEPBIND-first -> we keep libnvjpeg
//
// C ABI, POD only. Anything allocated in here is freed in here (DEEPBIND can give this
// object its own malloc resolution).
// ---------------------------------------------------------------------------------------
#include "NvJpegDecoder.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

#define OVNV_EXPORT extern "C" __attribute__((visibility("default")))
#define OVNV_ABI_COOKIE 0x4e564a01 /* 'NVJ' 01 */

namespace {
struct Handle {
  int n;
  NvJPEGDecoder **d;
};
} // namespace

OVNV_EXPORT int ovnv_abi(void) { return OVNV_ABI_COOKIE; }

OVNV_EXPORT void *ovnv_open(int ninst) {
  if (ninst < 1 || ninst > 128) return nullptr;
  Handle *h = (Handle *)std::calloc(1, sizeof(Handle));
  if (!h) return nullptr;
  h->n = ninst;
  h->d = (NvJPEGDecoder **)std::calloc((size_t)ninst, sizeof(NvJPEGDecoder *));
  if (!h->d) { std::free(h); return nullptr; }
  for (int i = 0; i < ninst; i++) {
    char nm[32];
    std::snprintf(nm, sizeof(nm), "ovjd%d", i);
    h->d[i] = NvJPEGDecoder::createJPEGDecoder(nm);
    if (!h->d[i]) {
      for (int k = 0; k < i; k++) delete h->d[k];
      std::free(h->d);
      std::free(h);
      return nullptr;
    }
  }
  return (void *)h;
}

// Thin wrapper over decodeToFd.  A NONNEGATIVE fd out of here is IMPOSSIBLE against stock
// libjpeg (stock's jpeg_decompress_struct has no `fd` member -- this TU would not compile
// against it), so fd>=0 at runtime is itself an in-binary proof the hardware path bound.
OVNV_EXPORT int ovnv_decode(void *hv, int inst, unsigned char *jpg, unsigned long n, int *out_fd,
                            unsigned *out_w, unsigned *out_h, unsigned *out_pixfmt) {
  Handle *h = (Handle *)hv;
  if (!h || inst < 0 || inst >= h->n || !jpg || !n || !out_fd) return -1;
  int fd = -1;
  uint32_t w = 0, ht = 0, pf = 0;
  int r = h->d[inst]->decodeToFd(fd, jpg, (unsigned long)n, pf, w, ht);
  if (r < 0 || fd < 0) return -1;
  *out_fd = fd;
  if (out_w) *out_w = w;
  if (out_h) *out_h = ht;
  if (out_pixfmt) *out_pixfmt = pf;
  return 0;
}

// Which library actually supplied the libjpeg C API to THIS object's call sites.
OVNV_EXPORT const char *ovnv_jpeg_provider(void) {
  static char buf[512];
  buf[0] = 0;
  // Take the address the way NvJpegDecoder.cpp's own call sites take it: through this
  // object's PLT.  &jpeg_read_header here resolves under our DEEPBIND scope.
  void *p = (void *)&jpeg_read_header;
  Dl_info di;
  if (dladdr(p, &di) && di.dli_fname) std::snprintf(buf, sizeof(buf), "%s", di.dli_fname);
  else std::snprintf(buf, sizeof(buf), "<unknown>");
  return buf;
}

OVNV_EXPORT void ovnv_close(void *hv) {
  Handle *h = (Handle *)hv;
  if (!h) return;
  for (int i = 0; i < h->n; i++) delete h->d[i];
  std::free(h->d);
  std::free(h);
}
