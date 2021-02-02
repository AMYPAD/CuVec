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
float increment_inplace_f(CuVec<float> &src) {
  cudaEvent_t eStart, eKern;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eKern);
  cudaEventRecord(eStart);
  dim3 thrds((src.size() + 1023) / 1024, 1, 1);
  dim3 blcks(1024, 1, 1);
  _d_incr<<<thrds, blcks>>>(src.data(), src.data(), src.size());
  // cudaDeviceSynchronize();
  cudaEventRecord(eKern);
  cudaEventSynchronize(eKern);
  float kernel_ms;
  cudaEventElapsedTime(&kernel_ms, eStart, eKern);
  // fprintf(stderr, "%.3f ms\n", kernel_ms);
  return kernel_ms;
}
CuVec<float> *increment_f(CuVec<float> &src) {
  cudaEvent_t eStart, eAlloc;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eAlloc);
  cudaEventRecord(eStart);
  CuVec<float> *dst = new CuVec<float>;
  *dst = src;
  cudaEventRecord(eAlloc);
  float alloc_ms;
  cudaEventElapsedTime(&alloc_ms, eStart, eAlloc);
  float kernel_ms = increment_inplace_f(*dst);
  // fprintf(stderr, "%.3f ms, %.3f ms\n", alloc_ms, kernel_ms);
  return dst;
}
