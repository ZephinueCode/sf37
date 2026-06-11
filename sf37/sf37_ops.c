#include "sf37_ops.h"

#include "sf37_quant.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

float sf37_sigmoid(float x) {
    if (x >= 0.0f) {
        const float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    const float z = expf(x);
    return z / (1.0f + z);
}

float sf37_silu(float x) {
    return x * sf37_sigmoid(x);
}

void sf37_rms_norm_weight1(float *out, const float *x, const float *weight,
                           uint64_t n, float eps) {
    double ss = 0.0;
    for (uint64_t i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    const float scale = 1.0f / sqrtf((float)(ss / (double)n) + eps);
    for (uint64_t i = 0; i < n; i++) out[i] = x[i] * scale * (weight[i] + 1.0f);
}

void sf37_swiglu(float *out, const float *gate, const float *up,
                 uint64_t n, float clamp) {
    for (uint64_t i = 0; i < n; i++) {
        float g = gate[i];
        float u = up[i];
        if (clamp > 1.0e-6f) {
            if (g > clamp) g = clamp;
            if (u > clamp) u = clamp;
            if (u < -clamp) u = -clamp;
        }
        out[i] = sf37_silu(g) * u;
    }
}

void sf37_topk_desc(const float *score, int n, int k, int *idx) {
    for (int i = 0; i < k; i++) idx[i] = -1;
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < k; j++) {
            if (idx[j] < 0 || score[i] > score[idx[j]]) {
                for (int m = k - 1; m > j; m--) idx[m] = idx[m - 1];
                idx[j] = i;
                break;
            }
        }
    }
}

void sf37_q8_0_quantize_activation(const float *x, int8_t *xq, float *scale,
                                   uint64_t n) {
    const uint64_t blocks = (n + 31u) / 32u;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint64_t off = b * 32u;
        uint64_t len = n - off;
        if (len > 32u) len = 32u;
        float max = 0.0f;
        float amax = 0.0f;
        for (uint64_t i = 0; i < len; i++) {
            const float ax = fabsf(x[off + i]);
            if (ax > amax) {
                amax = ax;
                max = x[off + i];
            }
        }
        if (amax == 0.0f) {
            scale[b] = 0.0f;
            memset(xq + off, 0, 32);
            continue;
        }
        (void)max;
        const float d = amax / 127.0f;
        const float iscale = 1.0f / d;
        scale[b] = d;
        for (uint64_t i = 0; i < len; i++) {
            int v = (int)lrintf(iscale * x[off + i]);
            if (v > 127) v = 127;
            if (v < -128) v = -128;
            xq[off + i] = (int8_t)v;
        }
        for (uint64_t i = len; i < 32u; i++) xq[off + i] = 0;
    }
}

float sf37_q8_0_row_dot_prequant(const uint8_t *row, const int8_t *xq,
                                  const float *xscale, uint64_t in_dim) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    float acc = 0.0f;
    for (uint64_t b = 0; b < blocks; b++) {
        const uint8_t *block = row + b * SF37_Q8_BLOCK_SIZE;
        const float wscale = sf37_f16_to_f32((uint16_t)block[0] | ((uint16_t)block[1] << 8));
        const int8_t *wq = (const int8_t *)(block + 2);
        const int8_t *xqb = xq + b * 32u;
        int isum = 0;
        uint64_t len = in_dim - b * 32u;
        if (len > 32u) len = 32u;
        for (uint64_t i = 0; i < len; i++) isum += (int)wq[i] * (int)xqb[i];
        acc += (float)isum * wscale * xscale[b];
    }
    return acc;
}

void sf37_q8_0_matvec(float *out, const uint8_t *weights,
                      uint64_t in_dim, uint64_t out_dim, const float *x) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * SF37_Q8_BLOCK_SIZE;
    int8_t *xq = malloc((size_t)blocks * 32u);
    float *xscale = malloc((size_t)blocks * sizeof(xscale[0]));
    if (!xq || !xscale) abort();
    sf37_q8_0_quantize_activation(x, xq, xscale, in_dim);
    for (uint64_t r = 0; r < out_dim; r++) {
        out[r] = sf37_q8_0_row_dot_prequant(weights + r * row_bytes, xq, xscale, in_dim);
    }
    free(xscale);
    free(xq);
}

void sf37_q8_0_matvec_pair(float *out0, float *out1,
                           const uint8_t *w0, const uint8_t *w1,
                           uint64_t in_dim, uint64_t out_dim,
                           const float *x) {
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t row_bytes = blocks * SF37_Q8_BLOCK_SIZE;
    int8_t *xq = malloc((size_t)blocks * 32u);
    float *xscale = malloc((size_t)blocks * sizeof(xscale[0]));
    if (!xq || !xscale) abort();
    sf37_q8_0_quantize_activation(x, xq, xscale, in_dim);
    for (uint64_t r = 0; r < out_dim; r++) {
        out0[r] = sf37_q8_0_row_dot_prequant(w0 + r * row_bytes, xq, xscale, in_dim);
        out1[r] = sf37_q8_0_row_dot_prequant(w1 + r * row_bytes, xq, xscale, in_dim);
    }
    free(xscale);
    free(xq);
}

void sf37_bf16_matvec(float *out, const uint16_t *weights,
                      uint64_t in_dim, uint64_t out_dim, const float *x) {
    for (uint64_t r = 0; r < out_dim; r++) {
        double acc = 0.0;
        const uint16_t *row = weights + r * in_dim;
        for (uint64_t i = 0; i < in_dim; i++) acc += (double)sf37_bf16_to_f32(row[i]) * x[i];
        out[r] = (float)acc;
    }
}

void sf37_q3_asym_matvec(float *out, const uint8_t *weights,
                         uint64_t in_dim, uint64_t out_dim, const float *x) {
    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q3_BLOCK_SIZE;
    for (uint64_t r = 0; r < out_dim; r++) {
        const uint8_t *row = weights + r * row_bytes;
        float acc = 0.0f;
        for (uint64_t b = 0; b < blocks; b++) {
            acc += sf37_q3_asym_block_dot(row + b * SF37_Q3_BLOCK_SIZE, x + b * SF37_QK_K);
        }
        out[r] = acc;
    }
}

void sf37_q2_asym_matvec(float *out, const uint8_t *weights,
                         uint64_t in_dim, uint64_t out_dim, const float *x) {
    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q2_BLOCK_SIZE;
    for (uint64_t r = 0; r < out_dim; r++) {
        const uint8_t *row = weights + r * row_bytes;
        float acc = 0.0f;
        for (uint64_t b = 0; b < blocks; b++) {
            acc += sf37_q2_asym_block_dot(row + b * SF37_Q2_BLOCK_SIZE, x + b * SF37_QK_K);
        }
        out[r] = acc;
    }
}
