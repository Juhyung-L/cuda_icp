#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "cublas_v2.h"
#include "cuda_runtime.h"
#include "cusolverDn.h"

#ifndef EXIT_WAIVED
#define EXIT_WAIVED 2
#endif

#define cudaCheck(error)                                                                              \
  if (error != cudaSuccess)                                                                           \
  {                                                                                                   \
    fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(error));       \
    exit(EXIT_FAILURE);                                                                               \
  }                                                                                                   \

#define cublasCheck(error)                                                                            \
  if (error != CUBLAS_STATUS_SUCCESS)                                                                 \
  {                                                                                                   \
    fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, cublasErrorGetString(error));     \
    exit(EXIT_FAILURE);                                                                               \
  }                                                                                                   \
  
#define cusolverCheck(error)                                                                          \
  if (error != CUSOLVER_STATUS_SUCCESS)                                                               \
  {                                                                                                   \
    fprintf(stderr, "CUDA error at %s:%d %s\n", __FILE__, __LINE__, cusolverErrorGetString(error));   \
    exit(EXIT_FAILURE);                                                                               \
  }                                                                                                   \

// cuBLAS API errors
static const char *cublasErrorGetString(cublasStatus_t error) {
  switch (error) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";
    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";
    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";
    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";
    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";
    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return "CUBLAS_STATUS_NOT_SUPPORTED";
    case CUBLAS_STATUS_LICENSE_ERROR:
      return "CUBLAS_STATUS_LICENSE_ERROR";
  }

  return "<unknown>";
}

// cuSOLVER API errors
static const char *cusolverErrorGetString(cusolverStatus_t error) {
  switch (error) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      return "CUSOLVER_STATUS_NOT_INITIALIZED";
    case CUSOLVER_STATUS_ALLOC_FAILED:
      return "CUSOLVER_STATUS_ALLOC_FAILED";
    case CUSOLVER_STATUS_INVALID_VALUE:
      return "CUSOLVER_STATUS_INVALID_VALUE";
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      return "CUSOLVER_STATUS_ARCH_MISMATCH";
    case CUSOLVER_STATUS_MAPPING_ERROR:
      return "CUSOLVER_STATUS_MAPPING_ERROR";
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      return "CUSOLVER_STATUS_EXECUTION_FAILED";
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      return "CUSOLVER_STATUS_INTERNAL_ERROR";
    case CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED";
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      return "CUSOLVER_STATUS_NOT_SUPPORTED ";
    case CUSOLVER_STATUS_ZERO_PIVOT:
      return "CUSOLVER_STATUS_ZERO_PIVOT";
    case CUSOLVER_STATUS_INVALID_LICENSE:
      return "CUSOLVER_STATUS_INVALID_LICENSE";
  }

  return "<unknown>";
}