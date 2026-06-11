#ifndef SF37_OPS_H
#define SF37_OPS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float sf37_sigmoid(float x);
float sf37_silu(float x);
void sf37_rms_norm_weight1(float *out, const float *x, const float *weight,
                           uint64_t n, float eps);
void sf37_swiglu(float *out, const float *gate, const float *up,
                 uint64_t n, float clamp);
void sf37_topk_desc(const float *score, int n, int k, int *idx);

void sf37_q8_0_quantize_activation(const float *x, int8_t *xq, float *scale,
                                   uint64_t n);
float sf37_q8_0_row_dot_prequant(const uint8_t *row, const int8_t *xq,
                                  const float *xscale, uint64_t in_dim);
void sf37_q8_0_matvec(float *out, const uint8_t *weights,
                      uint64_t in_dim, uint64_t out_dim, const float *x);
void sf37_q8_0_matvec_pair(float *out0, float *out1,
                           const uint8_t *w0, const uint8_t *w1,
                           uint64_t in_dim, uint64_t out_dim,
                           const float *x);

void sf37_bf16_matvec(float *out, const uint16_t *weights,
                      uint64_t in_dim, uint64_t out_dim, const float *x);
void sf37_q3_asym_matvec(float *out, const uint8_t *weights,
                         uint64_t in_dim, uint64_t out_dim, const float *x);
void sf37_q2_asym_matvec(float *out, const uint8_t *weights,
                         uint64_t in_dim, uint64_t out_dim, const float *x);

#ifdef __cplusplus
}
#endif

#endif
