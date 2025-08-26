#pragma once
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "tensorflow/c/c_api.h"
#include "tensorflow/c/eager/c_api.h"

#ifdef __cplusplus
extern "C"
{
#endif

  // Allocates and copies `len` bytes from `data` into a new TF_Tensor scalar.
  TF_Tensor *TF4SWIFT_NewTensorScalar(TF_DataType dtype, const void *data, size_t len);

#ifdef __cplusplus
}
#endif