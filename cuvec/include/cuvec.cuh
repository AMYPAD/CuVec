/**
 * Pure CUDA/C++11 template header. Provides:
 * - CuVec<T> // analogous to `std::vector` but using CUDA unified memory
 *
 * SWIG helpers wrapping `CuVec<T>`. Provides:
 * - SwigCuVec<T> // contains `CuVec<T> vec` and `std::vector<size_t> shape`
 * - SwigCuVec<T> *SwigCuVec_new(std::vector<size_t> shape)
 * - void SwigCuVec_del(SwigCuVec<T> *self)
 * - T *SwigCuVec_data(SwigCuVec<T> *self)
 * - size_t SwigCuVec_address(SwigCuVec<T> *self)
 * - std::vector<size_t> SwigCuVec_shape(SwigCuVec<T> *self)
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
  T
      *
      allocate(std::size_t n) {
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

/// pybind11 helpers
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
  ~NDCuVec() {
    vec.clear();
    vec.shrink_to_fit();
    shape.clear();
    shape.shrink_to_fit();
  }
};

/// SWIG helpers
template <class T> using SwigCuVec = NDCuVec<T>;
template <class T> SwigCuVec<T> *SwigCuVec_new(std::vector<size_t> shape) {
  SwigCuVec<T> *self = new SwigCuVec<T>(shape);
  return self;
}
template <class T> void SwigCuVec_del(SwigCuVec<T> *self) {
  self->~NDCuVec();
  delete self;
}
template <class T> T *SwigCuVec_data(SwigCuVec<T> *self) { return self->vec.data(); }
template <class T> size_t SwigCuVec_address(SwigCuVec<T> *self) {
  return (size_t)SwigCuVec_data(self);
}
template <class T> std::vector<size_t> SwigCuVec_shape(SwigCuVec<T> *self) { return self->shape; }
template <class T> void SwigCuVec_reshape(SwigCuVec<T> *self, const std::vector<size_t> &shape) {
  self->reshape(shape);
}

#endif // _CUVEC_H_
