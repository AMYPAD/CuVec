#ifndef _CU_HELPERS_
#define _CU_HELPERS_

void HandleError(cudaError_t err, const char *file, int line);
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))

bool PyHandleError(cudaError_t err, const char *file, int line);
#define HANDLE_PyErr(err) (PyHandleError(err, __FILE__, __LINE__))

#endif // _CU_HELPERS_
