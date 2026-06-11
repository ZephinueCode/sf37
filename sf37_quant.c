#include "sf37_quant.h"

#include <string.h>

float sf37_f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1fu;
    uint32_t mant = h & 0x03ffu;
    uint32_t out;
    if (exp == 0) {
        if (mant == 0) {
            out = sign;
        } else {
            exp = 1;
            while ((mant & 0x0400u) == 0) {
                mant <<= 1;
                exp--;
            }
            mant &= 0x03ffu;
            out = sign | ((exp + 112u) << 23) | (mant << 13);
        }
    } else if (exp == 31) {
        out = sign | 0x7f800000u | (mant << 13);
    } else {
        out = sign | ((exp + 112u) << 23) | (mant << 13);
    }
    float f;
    memcpy(&f, &out, sizeof(f));
    return f;
}

float sf37_bf16_to_f32(uint16_t h) {
    uint32_t bits = (uint32_t)h << 16;
    float f;
    memcpy(&f, &bits, sizeof(f));
    return f;
}

static uint16_t read_u16le(const uint8_t *p) {
    uint16_t v;
    memcpy(&v, p, sizeof(v));
    return v;
}

static uint32_t q3_get(const uint8_t *p, int i) {
    int bit = i * 3;
    int byte = bit >> 3;
    int shift = bit & 7;
    uint32_t v = (uint32_t)(p[byte] >> shift);
    if (shift > 5) v |= (uint32_t)p[byte + 1] << (8 - shift);
    return v & 7u;
}

void sf37_q8_0_block_decode(const uint8_t *block, float *dst32) {
    const float scale = sf37_f16_to_f32(read_u16le(block));
    const int8_t *qs = (const int8_t *)(block + 2);
    for (int i = 0; i < 32; i++) dst32[i] = (float)qs[i] * scale;
}

float sf37_q8_0_block_dot(const uint8_t *block, const float *x32) {
    const float scale = sf37_f16_to_f32(read_u16le(block));
    const int8_t *qs = (const int8_t *)(block + 2);
    float sum = 0.0f;
    for (int i = 0; i < 32; i++) sum += (float)qs[i] * scale * x32[i];
    return sum;
}

void sf37_q3_asym_block_decode(const uint8_t *block, float *dst256) {
    enum { scale_off = 0, zp_off = 8, qs_off = 12, group = 64, payload = 24 };
    for (int g = 0; g < 4; g++) {
        const float scale = sf37_f16_to_f32(read_u16le(block + scale_off + g * 2));
        const int zp = block[zp_off + g] & 7;
        const uint8_t *qs = block + qs_off + g * payload;
        float *dst = dst256 + g * group;
        for (int i = 0; i < group; i++) {
            dst[i] = ((float)q3_get(qs, i) - (float)zp) * scale;
        }
    }
}

void sf37_q2_asym_block_decode(const uint8_t *block, float *dst256) {
    enum { scale_off = 0, zp_off = 8, qs_off = 12, group = 64, payload = 16 };
    for (int g = 0; g < 4; g++) {
        const float scale = sf37_f16_to_f32(read_u16le(block + scale_off + g * 2));
        const int zp = block[zp_off + g] & 3;
        const uint8_t *qs = block + qs_off + g * payload;
        float *dst = dst256 + g * group;
        for (int i = 0; i < group; i++) {
            uint32_t q = (qs[i >> 2] >> ((i & 3) * 2)) & 3u;
            dst[i] = ((float)q - (float)zp) * scale;
        }
    }
}

float sf37_q3_asym_block_dot(const uint8_t *block, const float *x256) {
    enum { scale_off = 0, zp_off = 8, qs_off = 12, group = 64, payload = 24 };
    float sum = 0.0f;
    for (int g = 0; g < 4; g++) {
        const float scale = sf37_f16_to_f32(read_u16le(block + scale_off + g * 2));
        const int zp = block[zp_off + g] & 7;
        const uint8_t *qs = block + qs_off + g * payload;
        const float *x = x256 + g * group;
        for (int i = 0; i < group; i++) {
            sum += ((float)q3_get(qs, i) - (float)zp) * scale * x[i];
        }
    }
    return sum;
}

float sf37_q2_asym_block_dot(const uint8_t *block, const float *x256) {
    enum { scale_off = 0, zp_off = 8, qs_off = 12, group = 64, payload = 16 };
    float sum = 0.0f;
    for (int g = 0; g < 4; g++) {
        const float scale = sf37_f16_to_f32(read_u16le(block + scale_off + g * 2));
        const int zp = block[zp_off + g] & 3;
        const uint8_t *qs = block + qs_off + g * payload;
        const float *x = x256 + g * group;
        for (int i = 0; i < group; i++) {
            uint32_t q = (qs[i >> 2] >> ((i & 3) * 2)) & 3u;
            sum += ((float)q - (float)zp) * scale * x[i];
        }
    }
    return sum;
}
