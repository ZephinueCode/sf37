#ifndef SF37_QUANT_H
#define SF37_QUANT_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define SF37_QK_K 256
#define SF37_Q8_BLOCK_SIZE 34
#define SF37_Q3_BLOCK_SIZE 110
#define SF37_Q2_BLOCK_SIZE 84

float sf37_f16_to_f32(uint16_t h);
float sf37_bf16_to_f32(uint16_t h);

void sf37_q8_0_block_decode(const uint8_t *block, float *dst32);
float sf37_q8_0_block_dot(const uint8_t *block, const float *x32);
void sf37_q3_asym_block_decode(const uint8_t *block, float *dst256);
void sf37_q2_asym_block_decode(const uint8_t *block, float *dst256);
float sf37_q3_asym_block_dot(const uint8_t *block, const float *x256);
float sf37_q2_asym_block_dot(const uint8_t *block, const float *x256);

#ifdef __cplusplus
}
#endif

#endif
