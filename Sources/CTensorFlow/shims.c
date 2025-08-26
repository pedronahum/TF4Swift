#include "shims.h"
#include <stdlib.h>
#include <string.h>
#include <tensorflow/c/tf_tstring.h> // TF_TString_*
#include <tensorflow/c/tf_status.h>  // TF_Status (not strictly needed here)

// Deallocator that destroys each TF_TString element before freeing the buffer.
static void TF4SWIFT_Dealloc_TStringArray(void *data, size_t len, void *arg)
{
  (void)arg;
  size_t n = len / sizeof(TF_TString);
  TF_TString *arr = (TF_TString *)data;
  for (size_t i = 0; i < n; ++i)
  {
    TF_TString_Dealloc(&arr[i]);
  }
  free(data);
}

TF_Tensor *TF4SWIFT_NewTensorStringScalar(const char *bytes, size_t len)
{
  // One TF_TString object in the data buffer.
  TF_TString *buf = (TF_TString *)malloc(sizeof(TF_TString));
  if (!buf)
    return NULL;
  TF_TString_Init(buf);
  TF_TString_Copy(buf, bytes, len);

  // 0-D (scalar) tensor; custom deallocator will call TF_TString_Dealloc then free.
  TF_Tensor *t = TF_NewTensor(TF_STRING, /*dims*/ NULL, /*num_dims*/ 0,
                              buf, sizeof(TF_TString),
                              TF4SWIFT_Dealloc_TStringArray, NULL);
  if (!t)
  {
    TF_TString_Dealloc(buf);
    free(buf);
  }
  return t;
}

TF_Tensor *TF4SWIFT_NewTensorStringVector(const char *const *strings,
                                          const size_t *lens,
                                          int32_t count)
{
  if (count < 0)
    return NULL;
  int64_t dims[1];
  dims[0] = (int64_t)count;

  size_t n = (size_t)count;
  TF_TString *buf = (TF_TString *)malloc(n * sizeof(TF_TString));
  if (!buf)
    return NULL;

  for (size_t i = 0; i < n; ++i)
  {
    TF_TString_Init(&buf[i]);
    TF_TString_Copy(&buf[i], strings[i], lens[i]);
  }

  TF_Tensor *t = TF_NewTensor(TF_STRING, dims, /*num_dims*/ 1,
                              buf, n * sizeof(TF_TString),
                              TF4SWIFT_Dealloc_TStringArray, NULL);
  if (!t)
  {
    // Roll back if TF_NewTensor fails (rare).
    for (size_t i = 0; i < n; ++i)
      TF_TString_Dealloc(&buf[i]);
    free(buf);
  }
  return t;
}
