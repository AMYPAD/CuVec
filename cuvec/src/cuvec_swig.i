%module cuvec_swig

%include "exception.i"
%exception {
  try {
    $action
  } catch (const std::exception &e) {
    SWIG_exception(SWIG_RuntimeError, e.what());
  }
}

%include "cuvec.i" // NDCuVec<T>

%{
template <class T> NDCuVec<T> *NDCuVec_new(std::vector<size_t> shape) {
  NDCuVec<T> *self = new NDCuVec<T>(shape);
  return self;
}
template <class T> void NDCuVec_del(NDCuVec<T> *self) {
  delete self;
}
template <class T> size_t NDCuVec_address(NDCuVec<T> *self) {
  return (size_t)self->vec.data();
}
template <class T> std::vector<size_t> NDCuVec_shape(NDCuVec<T> *self) { return self->shape; }
template <class T> void NDCuVec_reshape(NDCuVec<T> *self, const std::vector<size_t> &shape) {
  self->reshape(shape);
}
%}
template <class T> NDCuVec<T> *NDCuVec_new(std::vector<size_t> shape);
template <class T> void NDCuVec_del(NDCuVec<T> *self);
template <class T> size_t NDCuVec_address(NDCuVec<T> *self);
template <class T> std::vector<size_t> NDCuVec_shape(NDCuVec<T> *self);
template <class T> void NDCuVec_reshape(NDCuVec<T> *self, const std::vector<size_t> &shape);

%template(NDCuVec_Shape) std::vector<size_t>;
%define MKCUVEC(T, typechar)
%template(NDCuVec_ ## typechar) NDCuVec<T>;
%template(NDCuVec_ ## typechar ## _new) NDCuVec_new<T>;
%template(NDCuVec_ ## typechar ## _del) NDCuVec_del<T>;
%template(NDCuVec_ ## typechar ## _address) NDCuVec_address<T>;
%template(NDCuVec_ ## typechar ## _shape) NDCuVec_shape<T>;
%template(NDCuVec_ ## typechar ## _reshape) NDCuVec_reshape<T>;
%enddef
MKCUVEC(signed char, b)
MKCUVEC(unsigned char, B)
MKCUVEC(char, c)
MKCUVEC(short, h)
MKCUVEC(unsigned short, H)
MKCUVEC(int, i)
MKCUVEC(unsigned int, I)
MKCUVEC(long long, q)
MKCUVEC(unsigned long long, Q)
#ifdef _CUVEC_HALF
MKCUVEC(_CUVEC_HALF, e)
#endif
MKCUVEC(float, f)
MKCUVEC(double, d)

%{
static const char __author__[] = "Casper da Costa-Luis (https://github.com/casperdcl)";
static const char __date__[] = "2021-2024";
static const char __version__[] = "4.0.0";
%}
static const char __author__[];
static const char __date__[];
static const char __version__[];
