#include "../sf37_ops.h"
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

static uint16_t f32_to_bf16(float f) {
    uint32_t x;
    memcpy(&x, &f, sizeof(x));
    return (uint16_t)(x >> 16);
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

static void make_q8_row(uint8_t *row, const int8_t *q, const float *scale,
                        uint64_t in_dim) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    memset(row, 0, (size_t)blocks * SF37_Q8_BLOCK_SIZE);
    for (uint64_t b = 0; b < blocks; b++) {
        write_u16le(row + b * SF37_Q8_BLOCK_SIZE, f32_to_f16(scale[b]));
        memcpy(row + b * SF37_Q8_BLOCK_SIZE + 2, q + b * 32u, 32);
    }
}

static void make_q3_row(uint8_t *row, const uint8_t *q, const float scale[4],
                        const uint8_t zp[4]) {
    memset(row, 0, SF37_Q3_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        write_u16le(row + g * 2, f32_to_f16(scale[g]));
        row[8 + g] = zp[g];
        uint8_t *qs = row + 12 + g * 24;
        for (int i = 0; i < 64; i++) put_q3(qs, i, q[g * 64 + i]);
    }
}

static void make_q2_row(uint8_t *row, const uint8_t *q, const float scale[4],
                        const uint8_t zp[4]) {
    memset(row, 0, SF37_Q2_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        write_u16le(row + g * 2, f32_to_f16(scale[g]));
        row[8 + g] = zp[g];
        uint8_t *qs = row + 12 + g * 16;
        for (int i = 0; i < 64; i++) {
            qs[i >> 2] |= (uint8_t)((q[g * 64 + i] & 3u) << ((i & 3) * 2));
        }
    }
}

static void assert_close(float a, float b, float eps, const char *what) {
    if (fabsf(a - b) > eps) {
        fprintf(stderr, "%s mismatch: got %.9g expected %.9g\n", what, a, b);
        exit(1);
    }
}

static void test_norm_swiglu_topk(void) {
    float x[4] = {1.0f, -2.0f, 3.0f, -4.0f};
    float w[4] = {0.0f, 0.5f, -0.25f, 1.0f};
    float out[4];
    sf37_rms_norm_weight1(out, x, w, 4, 1e-5f);
    double ss = 0.0;
    for (int i = 0; i < 4; i++) ss += (double)x[i] * x[i];
    float s = 1.0f / sqrtf((float)(ss / 4.0) + 1e-5f);
    for (int i = 0; i < 4; i++) assert_close(out[i], x[i] * s * (w[i] + 1.0f), 1e-6f, "rms");

    float gate[3] = {-2.0f, 0.0f, 10.0f};
    float up[3] = {1.0f, -2.0f, 8.0f};
    sf37_swiglu(out, gate, up, 3, 6.0f);
    assert_close(out[0], sf37_silu(-2.0f) * 1.0f, 1e-6f, "swiglu0");
    assert_close(out[1], 0.0f, 1e-6f, "swiglu1");
    assert_close(out[2], sf37_silu(6.0f) * 6.0f, 1e-5f, "swiglu2");

    float score[6] = {0.1f, 9.0f, 2.0f, 9.0f, -1.0f, 4.0f};
    int idx[3];
    sf37_topk_desc(score, 6, 3, idx);
    if (idx[0] != 1 || idx[1] != 3 || idx[2] != 5) {
        fprintf(stderr, "topk mismatch: %d %d %d\n", idx[0], idx[1], idx[2]);
        exit(1);
    }
}

static void test_q8_matvec(void) {
    enum { in_dim = 64, out_dim = 3 };
    float x[in_dim];
    int8_t q[out_dim][in_dim];
    float scale[out_dim][2] = {
        {0.25f, -0.125f},
        {-0.5f, 0.0625f},
        {0.03125f, 0.125f},
    };
    uint8_t weights[out_dim * 2 * SF37_Q8_BLOCK_SIZE];
    for (int i = 0; i < in_dim; i++) x[i] = (float)((i % 11) - 5) * 0.1f;
    for (int r = 0; r < out_dim; r++) {
        for (int i = 0; i < in_dim; i++) q[r][i] = (int8_t)((r + 1) * ((i % 9) - 4));
        make_q8_row(weights + r * 2 * SF37_Q8_BLOCK_SIZE, q[r], scale[r], in_dim);
    }
    float out[out_dim];
    sf37_q8_0_matvec(out, weights, in_dim, out_dim, x);

    int8_t xq[in_dim];
    float xs[2];
    sf37_q8_0_quantize_activation(x, xq, xs, in_dim);
    for (int r = 0; r < out_dim; r++) {
        float ref = 0.0f;
        for (int b = 0; b < 2; b++) {
            int isum = 0;
            for (int i = 0; i < 32; i++) isum += (int)q[r][b * 32 + i] * (int)xq[b * 32 + i];
            ref += (float)isum * sf37_f16_to_f32(f32_to_f16(scale[r][b])) * xs[b];
        }
        assert_close(out[r], ref, 0.0005f, "q8 matvec");
    }
}

static void test_bf16_matvec(void) {
    enum { in_dim = 4, out_dim = 2 };
    float x[in_dim] = {1.0f, -2.0f, 0.5f, 3.0f};
    float wf[out_dim][in_dim] = {{0.5f, -1.0f, 2.0f, 0.25f}, {-0.5f, 0.25f, 1.5f, -2.0f}};
    uint16_t w[out_dim][in_dim];
    for (int r = 0; r < out_dim; r++) {
        for (int i = 0; i < in_dim; i++) w[r][i] = f32_to_bf16(wf[r][i]);
    }
    float out[out_dim];
    sf37_bf16_matvec(out, &w[0][0], in_dim, out_dim, x);
    for (int r = 0; r < out_dim; r++) {
        float ref = 0.0f;
        for (int i = 0; i < in_dim; i++) ref += sf37_bf16_to_f32(w[r][i]) * x[i];
        assert_close(out[r], ref, 0.0001f, "bf16 matvec");
    }
}

static void test_q3_q2_matvec(void) {
    float x[SF37_QK_K];
    uint8_t q3[SF37_QK_K];
    uint8_t q2[SF37_QK_K];
    float scale[4] = {0.125f, 0.25f, 0.5f, 1.0f};
    uint8_t zp3[4] = {0, 3, 5, 7};
    uint8_t zp2[4] = {0, 1, 2, 3};
    for (int i = 0; i < SF37_QK_K; i++) {
        x[i] = (float)((i % 19) - 9) * 0.03125f;
        q3[i] = (uint8_t)((i * 5 + 1) & 7);
        q2[i] = (uint8_t)((i * 3 + 2) & 3);
    }
    uint8_t w3[SF37_Q3_BLOCK_SIZE];
    uint8_t w2[SF37_Q2_BLOCK_SIZE];
    make_q3_row(w3, q3, scale, zp3);
    make_q2_row(w2, q2, scale, zp2);
    float out3, out2;
    sf37_q3_asym_matvec(&out3, w3, SF37_QK_K, 1, x);
    sf37_q2_asym_matvec(&out2, w2, SF37_QK_K, 1, x);
    assert_close(out3, sf37_q3_asym_block_dot(w3, x), 0.0001f, "q3 matvec");
    assert_close(out2, sf37_q2_asym_block_dot(w2, x), 0.0001f, "q2 matvec");
}

int main(void) {
    test_norm_swiglu_topk();
    test_q8_matvec();
    test_bf16_matvec();
    test_q3_q2_matvec();
    printf("sf37 ops tests passed\n");
    return 0;
}
