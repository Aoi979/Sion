#pragma once
#define FETCH_FLOAT4(pointer) (reinterpret_cast<float4 *>(&(pointer))[0])
#define FETCH_CONST_FLOAT4(pointer)                                            \
  (reinterpret_cast<const float4 *>(&(pointer))[0])

