/**
 * Pure CUDA/C++11 template header providing `CuVec`
 * (anaologous to `std::vector` but using CUDA unified memory).
 */
#ifndef _CUVEC_H_
#define _CUVEC_H_

#include "cuda_runtime.h"
#include <cstdio>  // fprintf
#include <cstdlib> // std::size_t
#include <limits>  // std::numeric_limits
#include <new>     // std::bad_alloc
#include <vector>  // std::vector

namespace cuvec {
static void HandleError(cudaError_t err, const char *file, int line) {
  if (err != cudaSuccess) {
    fprintf(stderr, "%s in %s at line %d\n", cudaGetErrorString(err), file, line);
    exit(EXIT_FAILURE);
  }
}
} // namespace cuvec

template <class T> struct CuAlloc {
  typedef T value_type;

  CuAlloc() = default;
  template <class U> constexpr CuAlloc(const CuAlloc<U> &) noexcept {}

#if __cplusplus > 201703L
  [[nodiscard]]
#endif
      T *
      allocate(std::size_t n) {
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();

        T *p;
        // p = (T *)malloc(n * sizeof(T));
        cuvec::HandleError(cudaMallocManaged((void **)&p, n * sizeof(T)), __FILE__, __LINE__);
        if (p) {
          report(p, n);
          return p;
        }

        throw std::bad_alloc();
      }

  void deallocate(T *p, std::size_t n) noexcept {
    report(p, n, false);
    cuvec::HandleError(cudaFree((void *)p), __FILE__, __LINE__); // free(p);
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

#endif // _CUVEC_H_
