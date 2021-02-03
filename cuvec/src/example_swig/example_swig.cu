/**
 * Example external SWIG extension module using CuVec.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "cuvec.cuh" // CuVec
/** functions */
/// dst = src + 1
__global__ void _d_incr(float *dst, float *src, int N) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= N) return;
  dst[i] = src[i] + 1;
}
void increment_inplace_f(CuVec<float> &src, bool sync) {
  dim3 thrds((src.size() + 1023) / 1024, 1, 1);
  dim3 blcks(1024, 1, 1);
  _d_incr<<<thrds, blcks>>>(src.data(), src.data(), src.size());
  cuvec::HandleError(cudaGetLastError(), __FILE__, __LINE__);
  if (sync) cudaDeviceSynchronize();
}
CuVec<float> *increment_f(CuVec<float> &src, bool sync) {
  CuVec<float> *dst = new CuVec<float>;
  *dst = src;
  increment_inplace_f(*dst, sync);
  return dst;
}
