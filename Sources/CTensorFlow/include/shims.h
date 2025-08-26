#ifndef CTENSORFLOW_SHIMS_H
#define CTENSORFLOW_SHIMS_H

#include <stddef.h>
#include <string.h>

#ifdef __has_include
#if __has_include(<tensorflow/c/tf_status.h>)
#include <tensorflow/c/tf_status.h> // some builds put TF_Code here
#endif
#endif
#include <tensorflow/c/c_api.h>
#include <tensorflow/c/eager/c_api.h>
#include <tensorflow/c/tf_tensor.h> // TF_Tensor*

// Scalar tensor helper
static inline TF_Tensor *TF4SWIFT_NewTensorScalar(TF_DataType dtype,
                                                  const void *data,
                                                  size_t len)
{
  TF_Tensor *t = TF_AllocateTensor(dtype, /*dims*/ NULL, /*num_dims*/ 0, len);
  if (!t)
    return NULL;
  if (data && len)
  {
    memcpy(TF_TensorData(t), data, len);
  }
  return t;
}

// Always-available way to get the 'OK' status code.
static inline TF_Code TF4SWIFT_OK_CODE(void)
{
#ifdef TF_OK
  return TF_OK;
#else
  return (TF_Code)0; // TF_OK is 0 in TF_Code
#endif
}

// Create a TF_STRING scalar tensor from raw UTF-8 bytes.
TF_Tensor *TF4SWIFT_NewTensorStringScalar(const char *bytes, size_t len);

// Create a 1-D TF_STRING tensor from N strings (UTF-8, not NUL-terminated).
// `strings` points to N pointers; `lens` holds N lengths.
TF_Tensor *TF4SWIFT_NewTensorStringVector(const char *const *strings,
                                          const size_t *lens,
                                          int32_t count);

#endif // CTENSORFLOW_SHIMS_H