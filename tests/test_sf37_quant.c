#include "../sf37_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint16_t f32_to_f16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    uint32_t sign = (x >> 16) & 0x8000u;
    int exp = (int)((x >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = x & 0x7fffffu;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant |= 0x800000u;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t hmant = mant >> shift;
        if ((mant >> (shift - 1)) & 1u) hmant++;
        return (uint16_t)(sign | hmant);
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    uint32_t h = sign | ((uint32_t)exp << 10) | (mant >> 13);
    if (mant & 0x00001000u) h++;
    return (uint16_t)h;
}

static void write_u16le(uint8_t *p, uint16_t v) {
    memcpy(p, &v, sizeof(v));
}

static void put_q3(uint8_t *p, int i, uint32_t q) {
    int bit = i * 3;
    int byte = bit >> 3;
    int shift = bit & 7;
    p[byte] |= (uint8_t)((q << shift) & 0xffu);
    if (shift > 5) p[byte + 1] |= (uint8_t)(q >> (8 - shift));
}

static void make_q3_block(uint8_t *block, const uint8_t q[256],
                          const float scale[4], const uint8_t zp[4]) {
    memset(block, 0, SF37_Q3_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        write_u16le(block + g * 2, f32_to_f16(scale[g]));
        block[8 + g] = zp[g];
        uint8_t *qs = block + 12 + g * 24;
        for (int i = 0; i < 64; i++) put_q3(qs, i, q[g * 64 + i]);
    }
}

static void make_q2_block(uint8_t *block, const uint8_t q[256],
                          const float scale[4], const uint8_t zp[4]) {
    memset(block, 0, SF37_Q2_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        write_u16le(block + g * 2, f32_to_f16(scale[g]));
        block[8 + g] = zp[g];
        uint8_t *qs = block + 12 + g * 16;
        for (int i = 0; i < 64; i++) {
            qs[i >> 2] |= (uint8_t)((q[g * 64 + i] & 3u) << ((i & 3) * 2));
        }
    }
}

static void make_q8_block(uint8_t *block, const int8_t q[32], float scale) {
    memset(block, 0, SF37_Q8_BLOCK_SIZE);
    write_u16le(block, f32_to_f16(scale));
    memcpy(block + 2, q, 32);
}

static void assert_close(float a, float b, float eps, const char *what) {
    if (fabsf(a - b) > eps) {
        fprintf(stderr, "%s mismatch: got %.9g expected %.9g\n", what, a, b);
        exit(1);
    }
}

static void test_q3(void) {
    uint8_t q[256];
    float x[256];
    float dec[256];
    float ref[256];
    float scale[4] = {0.125f, 0.25f, 0.5f, 1.0f};
    uint8_t zp[4] = {0, 3, 5, 7};
    for (int i = 0; i < 256; i++) {
        q[i] = (uint8_t)((i * 5 + 3) & 7);
        x[i] = (float)((i % 13) - 6) * 0.03125f;
        ref[i] = ((float)q[i] - (float)zp[i / 64]) * scale[i / 64];
    }
    uint8_t block[SF37_Q3_BLOCK_SIZE];
    make_q3_block(block, q, scale, zp);
    sf37_q3_asym_block_decode(block, dec);
    float dot = sf37_q3_asym_block_dot(block, x);
    float ref_dot = 0.0f;
    for (int i = 0; i < 256; i++) {
        assert_close(dec[i], ref[i], 0.0001f, "q3 decode");
        ref_dot += ref[i] * x[i];
    }
    assert_close(dot, ref_dot, 0.0005f, "q3 dot");
}

static void test_q8(void) {
    int8_t q[32];
    float x[32];
    float dec[32];
    float ref[32];
    const float scale = 0.25f;
    for (int i = 0; i < 32; i++) {
        q[i] = (int8_t)(i - 16);
        x[i] = (float)((i % 7) - 3) * 0.125f;
        ref[i] = (float)q[i] * scale;
    }
    uint8_t block[SF37_Q8_BLOCK_SIZE];
    make_q8_block(block, q, scale);
    sf37_q8_0_block_decode(block, dec);
    float dot = sf37_q8_0_block_dot(block, x);
    float ref_dot = 0.0f;
    for (int i = 0; i < 32; i++) {
        assert_close(dec[i], ref[i], 0.0001f, "q8 decode");
        ref_dot += ref[i] * x[i];
    }
    assert_close(dot, ref_dot, 0.0001f, "q8 dot");
}

static void test_q2(void) {
    uint8_t q[256];
    float x[256];
    float dec[256];
    float ref[256];
    float scale[4] = {0.125f, 0.25f, 0.5f, 1.0f};
    uint8_t zp[4] = {0, 1, 2, 3};
    for (int i = 0; i < 256; i++) {
        q[i] = (uint8_t)((i * 3 + 1) & 3);
        x[i] = (float)((i % 17) - 8) * 0.015625f;
        ref[i] = ((float)q[i] - (float)zp[i / 64]) * scale[i / 64];
    }
    uint8_t block[SF37_Q2_BLOCK_SIZE];
    make_q2_block(block, q, scale, zp);
    sf37_q2_asym_block_decode(block, dec);
    float dot = sf37_q2_asym_block_dot(block, x);
    float ref_dot = 0.0f;
    for (int i = 0; i < 256; i++) {
        assert_close(dec[i], ref[i], 0.0001f, "q2 decode");
        ref_dot += ref[i] * x[i];
    }
    assert_close(dot, ref_dot, 0.0005f, "q2 dot");
}

int main(void) {
    test_q8();
    test_q3();
    test_q2();
    printf("sf37 quant tests passed\n");
    return 0;
}
