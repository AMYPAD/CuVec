/**
 * Pure CUDA/C++11 template header. Provides:
 * - CuVec<T> // analogous to `std::vector` but using CUDA unified memory
 *
 * Helpers wrapping `CuVec<T>`. Provides:
 * - NDCuVec<T> // contains `CuVec<T> vec` and `std::vector<size_t> shape`
 */
#ifndef _CUVEC_H_
#define _CUVEC_H_

#ifndef CUVEC_DISABLE_CUDA
#include "cuda_runtime.h"
#endif
#include <cstdio>    // fprintf
#include <cstdlib>   // std::size_t, std::malloc, std::free
#include <limits>    // std::numeric_limits
#include <new>       // std::bad_alloc
#include <stdexcept> // std::length_error
#include <vector>    // std::vector

#ifndef CUVEC_DISABLE_CUDA
namespace cuvec {
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
} // namespace cuvec
#endif

template <class T> struct CuAlloc {
  typedef T value_type;

  CuAlloc() = default;
  template <class U> constexpr CuAlloc(const CuAlloc<U> &) noexcept {}

#if __cplusplus > 201703L
  [[nodiscard]]
#endif
  T *allocate(std::size_t n) {
    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();

    T *p;
#ifndef CUVEC_DISABLE_CUDA
    cuvec::HandleError(cudaMallocManaged((void **)&p, n * sizeof(T)), __FILE__, __LINE__);
#else
    p = (T *)std::malloc(n * sizeof(T));
#endif
    if (p) {
      report(p, n);
      return p;
    }

    throw std::bad_alloc();
  }

  void deallocate(T *p, std::size_t n) noexcept {
    report(p, n, false);
#ifndef CUVEC_DISABLE_CUDA
    cuvec::HandleError(cudaFree((void *)p), __FILE__, __LINE__);
#else
    std::free(p);
#endif
  }

private:
  void report(T *p, std::size_t n, bool alloc = true) const {
#ifdef CUVEC_DEBUG
    fprintf(stderr, "d> %s: %zd B at 0x%zx\n", alloc ? "Alloc" : "Free", sizeof(T) * n,
            (size_t)(void *)p);
#endif
  }
};

template <class T, class U> bool operator==(const CuAlloc<T> &, const CuAlloc<U> &) {
  return true;
}
template <class T, class U> bool operator!=(const CuAlloc<T> &, const CuAlloc<U> &) {
  return false;
}

template <class T> using CuVec = std::vector<T, CuAlloc<T>>;

/// extension helpers
#ifndef _CUVEC_HALF
#ifndef CUVEC_DISABLE_CUDA
#include "cuda_fp16.h" // __half
#define _CUVEC_HALF __half
#else // CUVEC_DISABLE_CUDA
#ifdef __fp16
#define _CUVEC_HALF __fp16
#endif // __fp16
#endif // CUVEC_DISABLE_CUDA
#endif // _CUVEC_HALF

/// external wrapper helper
template <class T> struct NDCuVec {
  CuVec<T> vec;
  std::vector<size_t> shape;
  NDCuVec() = default;
  NDCuVec(const std::vector<size_t> &shape) : shape(shape) {
    size_t size = 1;
    for (auto &i : shape) size *= i;
    vec.resize(size);
  }
  void reshape(const std::vector<size_t> &shape) {
    size_t size = 1;
    for (auto &i : shape) size *= i;
    if (size != vec.size()) throw std::length_error("reshape: size mismatch");
    this->shape = shape;
  }
  std::vector<size_t> strides() const {
    const size_t ndim = this->shape.size();
    std::vector<size_t> s(ndim);
    s[ndim - 1] = sizeof(T);
    for (int i = ndim - 2; i >= 0; i--) s[i] = this->shape[i + 1] * s[i + 1];
    return s;
  }
};

#endif // _CUVEC_H_
