
void HandleError(cudaError_t err, const char *file, int line);
#define HANDLE_ERROR(err) (HandleError(err, __FILE__, __LINE__))
