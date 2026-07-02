#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#include <string>
#include <unordered_map>

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// split tensor buffer that splits matrices by rows across multiple devices
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_split_buffer_type(int main_device, const float * tensor_split);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

// ============================================================================
// CUDA Tensor Allocation Tracking (for tensor access counting)
// ============================================================================
// These functions track CUDA buffer allocations to enable tensor access
// counting for dynamically created intermediate CUDA tensors.
//
// ggml_cuda_register_tensor: Registers a CUDA buffer allocation with its
//    associated GGUF tensor name (extracted from CUDA naming convention).
// ggml_cuda_clear_tensor_map: Clears all registered tensor mappings.
// ggml_cuda_tensor_access: Records a tensor access event for access counting.
// ggml_cuda_resolve_tensor_id: Resolves a CUDA tensor name to a GGUF tensor ID.
// ============================================================================

GGML_BACKEND_API void ggml_cuda_register_tensor(void * data_addr, const char * tensor_name);
GGML_BACKEND_API void ggml_cuda_clear_tensor_map(void);
GGML_BACKEND_API void ggml_cuda_tensor_access(const ggml_tensor * tensor, int access_count);
GGML_BACKEND_API int32_t ggml_cuda_resolve_tensor_id(const char * cuda_tensor_name,
const std::unordered_map<std::string, int32_t> * model_tensor_id_map);

// Wrapper function for tensor data access with counting hook
void * ggml_cuda_access_tensor(void * data, const ggml_tensor * tensor, int inc);

// Wrapper macro for tensor data access with counting hook
// Usage in CUDA kernels: const float * src_d = (const float *)GGML_CUDA_NAME_TENSOR(src->data, src);
#ifndef GGML_CUDA_NAME_TENSOR
#define GGML_CUDA_NAME_TENSOR(data, tensor) ggml_cuda_access_tensor((data), (tensor), 1)
#endif

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
