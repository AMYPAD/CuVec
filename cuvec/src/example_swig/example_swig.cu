/**
 * Example external SWIG extension module using CuVec.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "cuvec.cuh" // CuVec
#include <stdexcept> // std::length_error
/** functions */
/// dst = src + 1
__global__ void _d_incr(float *dst, float *src, int N) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i >= N) return;
  dst[i] = src[i] + 1;
}
CuVec<float> *increment_f(CuVec<float> &src, CuVec<float> *dst, bool timing) {
  cudaEvent_t eStart, eAlloc, eKern;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eAlloc);
  cudaEventCreate(&eKern);
  cudaEventRecord(eStart);
  if (!dst) {
    dst = new CuVec<float>;
    dst->resize(src.size());
  }
  if (src.size() != dst->size()) throw std::length_error("`output` must be same shape as `src`");
  cudaEventRecord(eAlloc);
  dim3 thrds((src.size() + 1023) / 1024, 1, 1);
  dim3 blcks(1024, 1, 1);
  _d_incr<<<thrds, blcks>>>(dst->data(), src.data(), src.size());
  cuvec::HandleError(cudaGetLastError(), __FILE__, __LINE__);
  // cudaDeviceSynchronize();
  cudaEventRecord(eKern);
  cudaEventSynchronize(eKern);
  float alloc_ms, kernel_ms;
  cudaEventElapsedTime(&alloc_ms, eStart, eAlloc);
  cudaEventElapsedTime(&kernel_ms, eAlloc, eKern);
  // fprintf(stderr, "%.3f ms, %.3f ms\n", alloc_ms, kernel_ms);
  if (timing) {
    // hack: store times in last two elements of dst
    (*dst)[src.size() - 2] = alloc_ms;
    (*dst)[src.size() - 1] = kernel_ms;
  }
  return dst;
}
