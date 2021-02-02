%include "std_vector.i"

%{
#include "cuvec.cuh"    // CuAlloc
#include "cuda_fp16.h"  // __half

template<class _Tp> size_t data(std::vector<_Tp, CuAlloc<_Tp>> &vec){
  return (size_t) vec.data();
};
%}

template<class _Tp> size_t data(std::vector<_Tp, CuAlloc<_Tp>> &vec);

%template(Vector_b) std::vector<signed char, CuAlloc<signed char>>;
%template(Vector_b_data) data<signed char>;

%template(Vector_B) std::vector<unsigned char, CuAlloc<unsigned char>>;
%template(Vector_B_data) data<unsigned char>;

%template(Vector_c) std::vector<char, CuAlloc<char>>;
%template(Vector_c_data) data<char>;

%template(Vector_h) std::vector<short, CuAlloc<short>>;
%template(Vector_h_data) data<short>;

%template(Vector_H) std::vector<unsigned short, CuAlloc<unsigned short>>;
%template(Vector_H_data) data<unsigned short>;

%template(Vector_i) std::vector<int, CuAlloc<int>>;
%template(Vector_i_data) data<int>;

%template(Vector_I) std::vector<unsigned int, CuAlloc<unsigned int>>;
%template(Vector_I_data) data<unsigned int>;

%template(Vector_q) std::vector<long long, CuAlloc<long long>>;
%template(Vector_q_data) data<long long>;

%template(Vector_Q) std::vector<unsigned long long, CuAlloc<unsigned long long>>;
%template(Vector_Q_data) data<unsigned long long>;

%template(Vector_e) std::vector<__half, CuAlloc<__half>>;
%template(Vector_e_data) data<__half>;

%template(Vector_f) std::vector<float, CuAlloc<float>>;
%template(Vector_f_data) data<float>;

%template(Vector_d) std::vector<double, CuAlloc<double>>;
%template(Vector_d_data) data<double>;
