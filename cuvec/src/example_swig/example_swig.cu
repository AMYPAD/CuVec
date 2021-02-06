/**
 * Example external SWIG extension module using CuVec.
 *
 * Copyright (2021) Casper da Costa-Luis
 */
#include "cuvec.cuh" // SwigCuVec, SwigCuVec_new
#include <stdexcept> // std::length_error
#ifdef CUVEC_DISABLE_CUDA
#include <chrono> // std::chrono
#else
/// dst = src + 1
__global__ void _d_incr(float *dst, float *src, int X, int Y) {
  int x = threadIdx.x + blockDim.x * blockIdx.x;
  if (x >= X) return;
  int y = threadIdx.y + blockDim.y * blockIdx.y;
  if (y >= Y) return;
  dst[y * X + x] = src[y * X + x] + 1;
}
#endif // CUVEC_DISABLE_CUDA
SwigCuVec<float> *increment2d_f(SwigCuVec<float> &src, SwigCuVec<float> *output, bool timing) {
  auto &N = src.shape;
  if (N.size() != 2) throw std::length_error("`src` must be 2D");

#ifndef CUVEC_DISABLE_CUDA
  cudaEvent_t eStart, eAlloc, eKern;
  cudaEventCreate(&eStart);
  cudaEventCreate(&eAlloc);
  cudaEventCreate(&eKern);
  cudaEventRecord(eStart);
#else
  auto eStart = std::chrono::steady_clock::now();
#endif

  if (!output)
    output = SwigCuVec_new<float>(N);
  else if (N != output->shape)
    throw std::length_error("`output` must be same shape as `src`");

#ifndef CUVEC_DISABLE_CUDA
  cudaEventRecord(eAlloc);
  dim3 thrds((N[1] + 31) / 32, (N[0] + 31) / 32);
  dim3 blcks(32, 32);
  _d_incr<<<thrds, blcks>>>(output->vec.data(), src.vec.data(), N[1], N[0]);
  cuvec::HandleError(cudaGetLastError(), __FILE__, __LINE__);
  // cudaDeviceSynchronize();
  cudaEventRecord(eKern);
  cudaEventSynchronize(eKern);
  float alloc_ms, kernel_ms;
  cudaEventElapsedTime(&alloc_ms, eStart, eAlloc);
  cudaEventElapsedTime(&kernel_ms, eAlloc, eKern);
// fprintf(stderr, "%.3f ms, %.3f ms\n", alloc_ms, kernel_ms);
#else
  auto eAlloc = std::chrono::steady_clock::now();
  for (size_t i = 0; i < src.vec.size(); i++) output->vec[i] = src.vec[i] + 1;
  auto eKern = std::chrono::steady_clock::now();
  double alloc_ms = std::chrono::duration<double, std::milli>(eAlloc - eStart).count();
  double kernel_ms = std::chrono::duration<double, std::milli>(eKern - eAlloc).count();
// fprintf(stderr, "%.3lf ms, %.3lf ms\n", alloc_ms, kernel_ms);
#endif
  if (timing) {
    // hack: store times in first two elements of output
    output->vec[0] = alloc_ms;
    output->vec[1] = kernel_ms;
  }
  return output;
}
