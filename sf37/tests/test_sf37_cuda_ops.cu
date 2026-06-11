#include "sf37_cuda.h"
#include "sf37_ops.h"
#include "sf37_quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void assert_close(float got, float want, float tol, const char *what) {
    float diff = fabsf(got - want);
    if (diff > tol) {
        fprintf(stderr, "%s mismatch: got=%g want=%g diff=%g tol=%g\n",
                what, got, want, diff, tol);
        exit(1);
    }
}

static void assert_close_rel(float got, float want,
                             float abs_tol, float rel_tol,
                             const char *what) {
    float diff = fabsf(got - want);
    float scale = fmaxf(1.0f, fabsf(want));
    if (diff > abs_tol && diff > rel_tol * scale) {
        fprintf(stderr, "%s mismatch: got=%g want=%g diff=%g abs_tol=%g rel_tol=%g\n",
                what, got, want, diff, abs_tol, rel_tol);
        exit(1);
    }
}

static uint16_t f32_to_f16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { f };
    uint32_t sign = (v.u >> 16) & 0x8000u;
    int exp = (int)((v.u >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = v.u & 0x7fffffu;
    if (exp <= 0) return (uint16_t)sign;
    if (exp >= 31) return (uint16_t)(sign | 0x7c00u);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
}

static uint16_t f32_to_bf16(float f) {
    union {
        float f;
        uint32_t u;
    } v = { f };
    return (uint16_t)(v.u >> 16);
}

static sf37_cuda_tensor *tensor_from_host(const void *data, uint64_t bytes) {
    sf37_cuda_tensor *t = sf37_cuda_tensor_alloc(bytes);
    if (!t || !sf37_cuda_tensor_write(t, 0, data, bytes)) {
        fprintf(stderr, "failed to create CUDA tensor\n");
        exit(1);
    }
    return t;
}

static void make_q8_row(uint8_t *row, const int8_t *q, const float *scale, int n) {
    int blocks = (n + 31) / 32;
    for (int b = 0; b < blocks; b++) {
        uint8_t *blk = row + b * SF37_Q8_BLOCK_SIZE;
        uint16_t h = f32_to_f16(scale[b]);
        blk[0] = (uint8_t)(h & 0xffu);
        blk[1] = (uint8_t)(h >> 8);
        for (int i = 0; i < 32; i++) {
            int j = b * 32 + i;
            blk[2 + i] = (uint8_t)(j < n ? q[j] : 0);
        }
    }
}

static void q3_set(uint8_t *payload, int idx, uint8_t q) {
    int bit = idx * 3;
    for (int k = 0; k < 3; k++) {
        if ((q >> k) & 1u) payload[(bit + k) >> 3] |= (uint8_t)(1u << ((bit + k) & 7));
    }
}

static void q2_set(uint8_t *payload, int idx, uint8_t q) {
    payload[idx >> 2] |= (uint8_t)((q & 3u) << ((idx & 3) * 2));
}

static void make_q3_row(uint8_t *row, int r) {
    memset(row, 0, SF37_Q3_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        float scale = 0.015625f * (float)(g + 1 + r);
        uint16_t h = f32_to_f16(scale);
        row[g * 2] = (uint8_t)(h & 0xffu);
        row[g * 2 + 1] = (uint8_t)(h >> 8);
        row[8 + g] = (uint8_t)(2 + ((r + g) & 3));
        uint8_t *payload = row + 12 + g * 24;
        for (int i = 0; i < 64; i++) q3_set(payload, i, (uint8_t)((i + r + g) & 7));
    }
}

static void make_q2_row(uint8_t *row, int r) {
    memset(row, 0, SF37_Q2_BLOCK_SIZE);
    for (int g = 0; g < 4; g++) {
        float scale = 0.03125f * (float)(g + 1 + r);
        uint16_t h = f32_to_f16(scale);
        row[g * 2] = (uint8_t)(h & 0xffu);
        row[g * 2 + 1] = (uint8_t)(h >> 8);
        row[8 + g] = (uint8_t)(1 + ((r + g) & 1));
        uint8_t *payload = row + 12 + g * 16;
        for (int i = 0; i < 64; i++) q2_set(payload, i, (uint8_t)((i + r + g) & 3));
    }
}

static void test_rms_swiglu(void) {
    enum { n = 64 };
    float x[n], w[n], ref[n], got[n], gate[n], up[n], mid_ref[n], mid_got[n];
    for (int i = 0; i < n; i++) {
        x[i] = (float)((i % 13) - 6) * 0.25f;
        w[i] = (float)((i % 7) - 3) * 0.05f;
        gate[i] = (float)((i % 17) - 8) * 0.2f;
        up[i] = (float)((i % 11) - 5) * 0.3f;
    }
    sf37_rms_norm_weight1(ref, x, w, n, 1e-5f);
    sf37_swiglu(mid_ref, gate, up, n, 3.0f);

    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dw = tensor_from_host(w, sizeof(w));
    sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
    sf37_cuda_tensor *dg = tensor_from_host(gate, sizeof(gate));
    sf37_cuda_tensor *du = tensor_from_host(up, sizeof(up));
    sf37_cuda_tensor *dm = sf37_cuda_tensor_alloc(sizeof(mid_got));
    if (!sf37_cuda_rms_norm_weight1_f32(dout, dx, dw, n, 1e-5f) ||
        !sf37_cuda_swiglu_f32(dm, dg, du, n, 3.0f)) {
        fprintf(stderr, "CUDA rms/swiglu launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    sf37_cuda_tensor_read(dm, 0, mid_got, sizeof(mid_got));
    for (int i = 0; i < n; i++) {
        assert_close(got[i], ref[i], 2e-5f, "cuda rms");
        assert_close(mid_got[i], mid_ref[i], 2e-5f, "cuda swiglu");
    }
    sf37_cuda_tensor_free(dm);
    sf37_cuda_tensor_free(du);
    sf37_cuda_tensor_free(dg);
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    enum { n_tok = 3 };
    float xb[n_tok][n], bref[n_tok][n], bgot[n_tok][n], w_bf[n];
    uint16_t wb[n];
    for (int i = 0; i < n; i++) {
        wb[i] = f32_to_bf16(w[i]);
        w_bf[i] = sf37_bf16_to_f32(wb[i]);
    }
    for (int t = 0; t < n_tok; t++) {
        for (int i = 0; i < n; i++) xb[t][i] = x[i] + (float)(t - 1) * 0.03125f;
        sf37_rms_norm_weight1(bref[t], xb[t], w_bf, n, 1e-5f);
    }
    static uint16_t mapped_wb[n];
    memcpy(mapped_wb, wb, sizeof(mapped_wb));
    if (!sf37_cuda_set_model_map_range(mapped_wb, sizeof(mapped_wb), 0,
                                       sizeof(mapped_wb), sizeof(mapped_wb))) {
        fprintf(stderr, "CUDA set model map for batch rms failed\n");
        exit(1);
    }
    dx = tensor_from_host(xb, sizeof(xb));
    dout = sf37_cuda_tensor_alloc(sizeof(bgot));
    if (!sf37_cuda_rms_norm_weight1_bf16_batch_mapped(dout, dx, mapped_wb, sizeof(mapped_wb), 0,
                                                       n, n_tok, 1e-5f)) {
        fprintf(stderr, "CUDA batch rms mapped launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, bgot, sizeof(bgot));
    for (int t = 0; t < n_tok; t++) {
        for (int i = 0; i < n; i++) assert_close(bgot[t][i], bref[t][i], 2e-5f, "cuda batch rms");
    }
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dx);
}

static void cpu_rope_split_half(float *x, int n_head, int head_dim,
                                int rotary_dim, double theta, int pos) {
    int half = rotary_dim / 2;
    for (int h = 0; h < n_head; h++) {
        float *head = x + h * head_dim;
        for (int i = 0; i < half; i++) {
            double inv = 1.0 / pow(theta, (double)(i * 2) / (double)rotary_dim);
            float c = cosf((float)((double)pos * inv));
            float s = sinf((float)((double)pos * inv));
            float a = head[i];
            float b = head[i + half];
            head[i] = a * c - b * s;
            head[i + half] = b * c + a * s;
        }
    }
}

static float sigmoid_ref(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

static void cpu_attention_decode(float *out, const float *q,
                                 const float *k_cache, const float *v_cache,
                                 const float *head_gate, int n_cache,
                                 int cache_cap, int q_heads, int kv_heads,
                                 int head_dim, int sliding, int window) {
    int repeat = q_heads / kv_heads;
    int kv_row = kv_heads * head_dim;
    int start = 0;
    if (sliding && n_cache > window) start = n_cache - window;
    for (int h = 0; h < q_heads; h++) {
        int kvh = h / repeat;
        const float *qh = q + h * head_dim;
        float max_score = -INFINITY;
        for (int r = start; r < n_cache; r++) {
            int row = r % cache_cap;
            const float *kh = k_cache + row * kv_row + kvh * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) score += qh[d] * kh[d];
            score *= 1.0f / sqrtf((float)head_dim);
            if (score > max_score) max_score = score;
        }
        float denom = 0.0f;
        float *tmp = (float *)calloc((size_t)head_dim, sizeof(float));
        if (!tmp) {
            fprintf(stderr, "cpu attention tmp allocation failed\n");
            exit(1);
        }
        for (int r = start; r < n_cache; r++) {
            int row = r % cache_cap;
            const float *kh = k_cache + row * kv_row + kvh * head_dim;
            const float *vh = v_cache + row * kv_row + kvh * head_dim;
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) score += qh[d] * kh[d];
            score *= 1.0f / sqrtf((float)head_dim);
            float w = expf(score - max_score);
            denom += w;
            for (int d = 0; d < head_dim; d++) tmp[d] += w * vh[d];
        }
        float g = sigmoid_ref(head_gate[h]);
        for (int d = 0; d < head_dim; d++) out[h * head_dim + d] = tmp[d] / denom * g;
        free(tmp);
    }
}

static void test_head_rope_attention(void) {
    enum { n_head = 3, head_dim = 8, rotary_dim = 4 };
    float x[n_head * head_dim], ref[n_head * head_dim], got[n_head * head_dim];
    float wf[head_dim];
    uint16_t wb[head_dim];
    for (int i = 0; i < n_head * head_dim; i++) x[i] = (float)((i % 17) - 8) * 0.125f;
    for (int i = 0; i < head_dim; i++) {
        wf[i] = (float)((i % 5) - 2) * 0.04f;
        wb[i] = f32_to_bf16(wf[i]);
        wf[i] = sf37_bf16_to_f32(wb[i]);
    }
    for (int h = 0; h < n_head; h++) {
        sf37_rms_norm_weight1(ref + h * head_dim, x + h * head_dim, wf, head_dim, 1e-5f);
    }

    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dw = tensor_from_host(wb, sizeof(wb));
    if (!sf37_cuda_head_rms_norm_weight1_bf16(dx, dw, n_head, head_dim, 1e-5f)) {
        fprintf(stderr, "CUDA head norm launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
    for (int i = 0; i < n_head * head_dim; i++) assert_close(got[i], ref[i], 2e-5f, "cuda head norm");

    cpu_rope_split_half(ref, n_head, head_dim, rotary_dim, 10000.0, 3);
    if (!sf37_cuda_rope_split_half(dx, n_head, head_dim, rotary_dim, 10000.0, 0, 3)) {
        fprintf(stderr, "CUDA rope launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
    for (int i = 0; i < n_head * head_dim; i++) assert_close(got[i], ref[i], 2e-5f, "cuda rope");
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    enum { q_heads = 4, kv_heads = 2, n_cache = 3, cache_cap = 4 };
    float q[q_heads * head_dim], k[cache_cap * kv_heads * head_dim];
    float v[cache_cap * kv_heads * head_dim], gate[q_heads];
    float aref[q_heads * head_dim], agot[q_heads * head_dim];
    for (int i = 0; i < q_heads * head_dim; i++) q[i] = (float)((i % 11) - 5) * 0.08f;
    for (int i = 0; i < cache_cap * kv_heads * head_dim; i++) {
        k[i] = (float)((i % 13) - 6) * 0.05f;
        v[i] = (float)((i % 7) - 3) * 0.09f;
    }
    for (int i = 0; i < q_heads; i++) gate[i] = (float)(i - 2) * 0.4f;
    cpu_attention_decode(aref, q, k, v, gate, n_cache, cache_cap, q_heads, kv_heads, head_dim, 0, 0);
    sf37_cuda_tensor *dq = tensor_from_host(q, sizeof(q));
    sf37_cuda_tensor *dk = tensor_from_host(k, sizeof(k));
    sf37_cuda_tensor *dv = tensor_from_host(v, sizeof(v));
    sf37_cuda_tensor *dg = tensor_from_host(gate, sizeof(gate));
    sf37_cuda_tensor *doa = sf37_cuda_tensor_alloc(sizeof(agot));
    if (!sf37_cuda_attention_decode_heads(doa, dq, dk, dv, dg, n_cache, cache_cap,
                                           q_heads, kv_heads, head_dim, 0, 0)) {
        fprintf(stderr, "CUDA attention decode launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(doa, 0, agot, sizeof(agot));
    for (int i = 0; i < q_heads * head_dim; i++) assert_close(agot[i], aref[i], 2e-5f, "cuda attention");
    sf37_cuda_tensor_free(doa);
    sf37_cuda_tensor_free(dg);
    sf37_cuda_tensor_free(dv);
    sf37_cuda_tensor_free(dk);
    sf37_cuda_tensor_free(dq);

    enum { qh128 = 8, kvh128 = 2, hd128 = 128, cap128 = 520 };
    float q128[qh128 * hd128], k128[cap128 * kvh128 * hd128];
    float v128[cap128 * kvh128 * hd128], gate128[qh128];
    float ref128[qh128 * hd128], got128[qh128 * hd128];
    for (int i = 0; i < qh128 * hd128; i++) q128[i] = (float)((i % 23) - 11) * 0.013f;
    for (int i = 0; i < cap128 * kvh128 * hd128; i++) {
        k128[i] = (float)((i % 29) - 14) * 0.009f;
        v128[i] = (float)((i % 31) - 15) * 0.011f;
    }
    for (int i = 0; i < qh128; i++) gate128[i] = (float)(i - 4) * 0.17f;
    dq = tensor_from_host(q128, sizeof(q128));
    dk = tensor_from_host(k128, sizeof(k128));
    dv = tensor_from_host(v128, sizeof(v128));
    dg = tensor_from_host(gate128, sizeof(gate128));
    doa = sf37_cuda_tensor_alloc(sizeof(got128));
    const int cache_cases[] = {1, 2, 17, 513};
    for (int ci = 0; ci < 4; ci++) {
        int nc = cache_cases[ci];
        int sliding = nc == 513;
        cpu_attention_decode(ref128, q128, k128, v128, gate128, nc, cap128,
                             qh128, kvh128, hd128, sliding, 512);
        if (!sf37_cuda_attention_decode_heads(doa, dq, dk, dv, dg, nc, cap128,
                                               qh128, kvh128, hd128, sliding, 512)) {
            fprintf(stderr, "CUDA attention128 decode launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(doa, 0, got128, sizeof(got128));
        for (int i = 0; i < qh128 * hd128; i++) assert_close(got128[i], ref128[i], 5e-5f, "cuda attention128");
    }
    setenv("SF37_CUDA_NO_WARP_ATTENTION", "1", 1);
    cpu_attention_decode(ref128, q128, k128, v128, gate128, 17, cap128,
                         qh128, kvh128, hd128, 0, 512);
    if (!sf37_cuda_attention_decode_heads(doa, dq, dk, dv, dg, 17, cap128,
                                           qh128, kvh128, hd128, 0, 512)) {
        fprintf(stderr, "CUDA attention128 fallback launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(doa, 0, got128, sizeof(got128));
    for (int i = 0; i < qh128 * hd128; i++) assert_close(got128[i], ref128[i], 5e-5f, "cuda attention128 fallback");
    unsetenv("SF37_CUDA_NO_WARP_ATTENTION");
    sf37_cuda_tensor_free(doa);
    sf37_cuda_tensor_free(dg);
    sf37_cuda_tensor_free(dv);
    sf37_cuda_tensor_free(dk);
    sf37_cuda_tensor_free(dq);
}

static void test_q8_bf16(void) {
    enum { in_dim = 70, out_dim = 7, q8_blocks = (in_dim + 31) / 32 };
    float x[in_dim], ref[out_dim], got[out_dim];
    int8_t q[out_dim][in_dim];
    float scale[out_dim][q8_blocks];
    uint8_t weights[out_dim * q8_blocks * SF37_Q8_BLOCK_SIZE];
    for (int i = 0; i < in_dim; i++) x[i] = (float)((i % 19) - 9) * 0.07f;
    for (int r = 0; r < out_dim; r++) {
        for (int b = 0; b < q8_blocks; b++) scale[r][b] = 0.015625f * (float)(r + b + 1);
        for (int i = 0; i < in_dim; i++) q[r][i] = (int8_t)((r + 1) * ((i % 9) - 4));
        make_q8_row(weights + r * q8_blocks * SF37_Q8_BLOCK_SIZE, q[r], scale[r], in_dim);
    }
    sf37_q8_0_matvec(ref, weights, in_dim, out_dim, x);

    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dw = tensor_from_host(weights, sizeof(weights));
    sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
    if (!sf37_cuda_matvec_q8_0(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q8 matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 1e-4f, "cuda q8");
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    float ref1[out_dim], got1[out_dim];
    sf37_q8_0_matvec(ref, weights, in_dim, out_dim, x);
    sf37_q8_0_matvec(ref1, weights, in_dim, out_dim, x);
    dx = tensor_from_host(x, sizeof(x));
    dw = tensor_from_host(weights, sizeof(weights));
    sf37_cuda_tensor *dout1 = sf37_cuda_tensor_alloc(sizeof(got1));
    dout = sf37_cuda_tensor_alloc(sizeof(got));
    if (!sf37_cuda_matvec_q8_0_pair(dout, dout1, dw, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q8 pair matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    sf37_cuda_tensor_read(dout1, 0, got1, sizeof(got1));
    for (int r = 0; r < out_dim; r++) {
        assert_close(got[r], ref[r], 1e-4f, "cuda q8 pair0");
        assert_close(got1[r], ref1[r], 1e-4f, "cuda q8 pair1");
    }
    setenv("SF37_CUDA_NO_Q8_PAIR_FUSED", "1", 1);
    if (!sf37_cuda_matvec_q8_0_pair(dout, dout1, dw, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q8 pair fallback matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    sf37_cuda_tensor_read(dout1, 0, got1, sizeof(got1));
    for (int r = 0; r < out_dim; r++) {
        assert_close(got[r], ref[r], 1e-4f, "cuda q8 pair fallback0");
        assert_close(got1[r], ref1[r], 1e-4f, "cuda q8 pair fallback1");
    }
    unsetenv("SF37_CUDA_NO_Q8_PAIR_FUSED");
    sf37_cuda_tensor_free(dout1);
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    enum { n_tok = 3 };
    float xb[n_tok][in_dim], bref[n_tok][out_dim], bgot[n_tok][out_dim];
    for (int t = 0; t < n_tok; t++) {
        for (int i = 0; i < in_dim; i++) {
            xb[t][i] = (float)(((t + 3) * (i % 23) - 17) % 31) * 0.03125f;
        }
        sf37_q8_0_matvec(bref[t], weights, in_dim, out_dim, xb[t]);
    }
    static uint8_t mapped_weights[out_dim * q8_blocks * SF37_Q8_BLOCK_SIZE];
    memcpy(mapped_weights, weights, sizeof(mapped_weights));
    if (!sf37_cuda_set_model_map_range(mapped_weights, sizeof(mapped_weights), 0,
                                       sizeof(mapped_weights), sizeof(mapped_weights))) {
        fprintf(stderr, "CUDA set model map for q8 batch failed\n");
        exit(1);
    }
    dx = tensor_from_host(xb, sizeof(xb));
    dout = sf37_cuda_tensor_alloc(sizeof(bgot));
    setenv("SF37_CUDA_NO_Q8_F16_CACHE", "1", 1);
    setenv("SF37_CUDA_NO_Q8_F32_CACHE", "1", 1);
    if (!sf37_cuda_matmul_q8_0_mapped(dout, mapped_weights, sizeof(mapped_weights), 0,
                                      in_dim, out_dim, dx, n_tok)) {
        fprintf(stderr, "CUDA q8 mapped batch fallback matmul launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, bgot, sizeof(bgot));
    for (int t = 0; t < n_tok; t++) {
        for (int r = 0; r < out_dim; r++) assert_close(bgot[t][r], bref[t][r], 1e-4f, "cuda q8 batch fallback");
    }
    unsetenv("SF37_CUDA_NO_Q8_F16_CACHE");
    unsetenv("SF37_CUDA_NO_Q8_F32_CACHE");
    setenv("SF37_CUDA_Q8_F16_CACHE_MB", "64", 1);
    if (!sf37_cuda_cache_q8_f16_range(mapped_weights, sizeof(mapped_weights), 0,
                                      sizeof(mapped_weights),
                                      in_dim, out_dim, "q8_0_test")) {
        fprintf(stderr, "CUDA q8 f16 cache preload failed\n");
        exit(1);
    }
    if (!sf37_cuda_matmul_q8_0_mapped(dout, mapped_weights, sizeof(mapped_weights), 0,
                                      in_dim, out_dim, dx, n_tok)) {
        fprintf(stderr, "CUDA q8 mapped batch f16 matmul launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, bgot, sizeof(bgot));
    for (int t = 0; t < n_tok; t++) {
        for (int r = 0; r < out_dim; r++) assert_close_rel(bgot[t][r], bref[t][r], 1e-2f, 2e-2f, "cuda q8 batch f16");
    }
    unsetenv("SF37_CUDA_Q8_F16_CACHE_MB");
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dx);

    uint16_t bw[out_dim][in_dim];
    for (int r = 0; r < out_dim; r++) {
        for (int i = 0; i < in_dim; i++) {
            float v = (float)((r * 3 + i) % 23 - 11) * 0.0625f;
            bw[r][i] = f32_to_bf16(v);
        }
    }
    sf37_bf16_matvec(ref, &bw[0][0], in_dim, out_dim, x);
    dx = tensor_from_host(x, sizeof(x));
    dw = tensor_from_host(bw, sizeof(bw));
    dout = sf37_cuda_tensor_alloc(sizeof(got));
    if (!sf37_cuda_matvec_bf16(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA bf16 matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 1e-4f, "cuda bf16");

    int32_t top = -1;
    if (!sf37_cuda_argmax_f32(&top, dout, out_dim, -1)) {
        fprintf(stderr, "CUDA f32 argmax launch failed\n");
        exit(1);
    }
    int ref_top = 0;
    for (int r = 1; r < out_dim; r++) {
        if (ref[r] > ref[ref_top]) ref_top = r;
    }
    if (top != ref_top) {
        fprintf(stderr, "CUDA f32 argmax mismatch: got=%d want=%d\n", top, ref_top);
        exit(1);
    }
    if (!sf37_cuda_argmax_f32(&top, dout, out_dim, ref_top)) {
        fprintf(stderr, "CUDA f32 argmax exclude launch failed\n");
        exit(1);
    }
    int ref_excl = ref_top == 0 ? 1 : 0;
    for (int r = 0; r < out_dim; r++) {
        if (r != ref_top && ref[r] > ref[ref_excl]) ref_excl = r;
    }
    if (top != ref_excl) {
        fprintf(stderr, "CUDA f32 argmax exclude mismatch: got=%d want=%d\n", top, ref_excl);
        exit(1);
    }

    static uint16_t mapped_bf16[out_dim][in_dim];
    memcpy(mapped_bf16, bw, sizeof(mapped_bf16));
    if (!sf37_cuda_set_model_map_range(mapped_bf16, sizeof(mapped_bf16), 0,
                                       sizeof(mapped_bf16), sizeof(mapped_bf16))) {
        fprintf(stderr, "CUDA set model map for bf16 argmax failed\n");
        exit(1);
    }
    if (!sf37_cuda_matvec_bf16_argmax_mapped(&top, mapped_bf16, sizeof(mapped_bf16), 0,
                                             in_dim, out_dim, dx, -1)) {
        fprintf(stderr, "CUDA bf16 fused argmax launch failed\n");
        exit(1);
    }
    if (top != ref_top) {
        fprintf(stderr, "CUDA bf16 fused argmax mismatch: got=%d want=%d\n", top, ref_top);
        exit(1);
    }
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    float fw[out_dim][in_dim];
    for (int r = 0; r < out_dim; r++) {
        double acc = 0.0;
        for (int i = 0; i < in_dim; i++) {
            fw[r][i] = (float)((r * 5 + i) % 31 - 15) * 0.03125f;
            acc += (double)fw[r][i] * x[i];
        }
        ref[r] = (float)acc;
    }
    dx = tensor_from_host(x, sizeof(x));
    dw = tensor_from_host(fw, sizeof(fw));
    dout = sf37_cuda_tensor_alloc(sizeof(got));
    if (!sf37_cuda_matvec_f32(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA f32 matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 1e-4f, "cuda f32");
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);
}

static void test_q3_q2(void) {
    enum { in_dim = 256, out_dim = 5 };
    float x[in_dim], ref[out_dim], got[out_dim];
    uint8_t q3[out_dim * SF37_Q3_BLOCK_SIZE];
    uint8_t q2[out_dim * SF37_Q2_BLOCK_SIZE];
    for (int i = 0; i < in_dim; i++) x[i] = (float)((i % 29) - 14) * 0.02f;
    for (int r = 0; r < out_dim; r++) {
        make_q3_row(q3 + r * SF37_Q3_BLOCK_SIZE, r);
        make_q2_row(q2 + r * SF37_Q2_BLOCK_SIZE, r);
    }

    sf37_q3_asym_matvec(ref, q3, in_dim, out_dim, x);
    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dw = tensor_from_host(q3, sizeof(q3));
    sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
    setenv("SF37_CUDA_NO_QLOW_Q8K", "1", 1);
    if (!sf37_cuda_matvec_q3_asym(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q3 matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 1e-4f, "cuda q3");
    unsetenv("SF37_CUDA_NO_QLOW_Q8K");
    setenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS", "0", 1);
    if (!sf37_cuda_matvec_q3_asym(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q3 q8k matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 4e-3f, "cuda q3 q8k");
    unsetenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS");
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);

    sf37_q2_asym_matvec(ref, q2, in_dim, out_dim, x);
    dx = tensor_from_host(x, sizeof(x));
    dw = tensor_from_host(q2, sizeof(q2));
    dout = sf37_cuda_tensor_alloc(sizeof(got));
    setenv("SF37_CUDA_NO_QLOW_Q8K", "1", 1);
    if (!sf37_cuda_matvec_q2_asym(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q2 matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 1e-4f, "cuda q2");
    unsetenv("SF37_CUDA_NO_QLOW_Q8K");
    setenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS", "0", 1);
    if (!sf37_cuda_matvec_q2_asym(dout, dw, in_dim, out_dim, dx)) {
        fprintf(stderr, "CUDA q2 q8k matvec launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 4e-3f, "cuda q2 q8k");
    unsetenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS");
    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dw);
    sf37_cuda_tensor_free(dx);
}

static void test_router_warp_top8(void) {
    enum { n_expert = 288, topk = 8 };
    float logits[n_expert], bias[n_expert], prob[n_expert], got_prob[n_expert];
    float score[n_expert], ref_w[topk], got_w[topk];
    int ref_sel[topk];
    int32_t got_sel[topk];
    for (int e = 0; e < n_expert; e++) {
        logits[e] = -4.0f + (float)((e * 17) % 53) * 0.013f;
        bias[e] = (float)((e * 11) % 19 - 9) * 0.002f;
    }
    const int forced[topk] = {40, 41, 7, 280, 13, 120, 121, 200};
    for (int i = 0; i < topk; i++) {
        logits[forced[i]] = 4.0f - (float)(i / 2) * 0.25f;
        bias[forced[i]] = (i < 2) ? 0.75f : 0.5f - (float)i * 0.01f;
    }
    logits[40] = logits[41];
    bias[40] = bias[41];

    for (int e = 0; e < n_expert; e++) {
        prob[e] = sigmoid_ref(logits[e]);
        score[e] = prob[e] + bias[e];
    }
    sf37_topk_desc(score, n_expert, topk, ref_sel);
    float sum = 0.0f;
    for (int i = 0; i < topk; i++) {
        ref_w[i] = prob[ref_sel[i]];
        sum += ref_w[i];
    }
    for (int i = 0; i < topk; i++) ref_w[i] = ref_w[i] / sum * 3.0f;

    sf37_cuda_tensor *d_logits = tensor_from_host(logits, sizeof(logits));
    sf37_cuda_tensor *d_bias = tensor_from_host(bias, sizeof(bias));
    sf37_cuda_tensor *d_prob = sf37_cuda_tensor_alloc(sizeof(got_prob));
    sf37_cuda_tensor *d_sel = sf37_cuda_tensor_alloc(sizeof(got_sel));
    sf37_cuda_tensor *d_w = sf37_cuda_tensor_alloc(sizeof(got_w));
    if (!sf37_cuda_router_select(d_sel, d_w, d_prob, d_logits, d_bias,
                                 n_expert, topk, 3.0f)) {
        fprintf(stderr, "CUDA router warp top8 launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(d_sel, 0, got_sel, sizeof(got_sel));
    sf37_cuda_tensor_read(d_w, 0, got_w, sizeof(got_w));
    sf37_cuda_tensor_read(d_prob, 0, got_prob, sizeof(got_prob));
    for (int i = 0; i < topk; i++) {
        if (got_sel[i] != ref_sel[i]) {
            fprintf(stderr, "CUDA router warp selected mismatch at %d: got=%d want=%d\n",
                    i, got_sel[i], ref_sel[i]);
            exit(1);
        }
        assert_close(got_w[i], ref_w[i], 1e-6f, "cuda router warp weight");
    }
    for (int e = 0; e < n_expert; e++) assert_close(got_prob[e], prob[e], 1e-6f, "cuda router warp prob");

    setenv("SF37_CUDA_NO_WARP_ROUTER_SELECT", "1", 1);
    if (!sf37_cuda_router_select(d_sel, d_w, d_prob, d_logits, d_bias,
                                 n_expert, topk, 3.0f)) {
        fprintf(stderr, "CUDA router scalar fallback launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(d_sel, 0, got_sel, sizeof(got_sel));
    sf37_cuda_tensor_read(d_w, 0, got_w, sizeof(got_w));
    for (int i = 0; i < topk; i++) {
        if (got_sel[i] != ref_sel[i]) {
            fprintf(stderr, "CUDA router fallback selected mismatch at %d: got=%d want=%d\n",
                    i, got_sel[i], ref_sel[i]);
            exit(1);
        }
        assert_close(got_w[i], ref_w[i], 1e-6f, "cuda router fallback weight");
    }
    unsetenv("SF37_CUDA_NO_WARP_ROUTER_SELECT");

    sf37_cuda_tensor_free(d_w);
    sf37_cuda_tensor_free(d_sel);
    sf37_cuda_tensor_free(d_prob);
    sf37_cuda_tensor_free(d_bias);
    sf37_cuda_tensor_free(d_logits);
}

static void test_router_moe(void) {
    enum { n_expert = 4, topk = 2, in_dim = 256, mid_dim = 256, out_dim = 6 };
    float x[in_dim], logits[n_expert], bias[n_expert], prob[n_expert], score[n_expert];
    float ref[out_dim], got[out_dim], direct_got[out_dim], router_w[topk], got_w[topk];
    int ref_sel[topk];
    int32_t got_sel[topk];
    for (int i = 0; i < in_dim; i++) x[i] = (float)((i % 37) - 18) * 0.015f;
    for (int e = 0; e < n_expert; e++) {
        logits[e] = (float)(e - 1) * 0.375f;
        bias[e] = (float)((e * 7) % 5 - 2) * 0.025f;
        prob[e] = sf37_sigmoid(logits[e]);
        score[e] = prob[e] + bias[e];
    }
    sf37_topk_desc(score, n_expert, topk, ref_sel);
    float sum = 0.0f;
    for (int i = 0; i < topk; i++) {
        router_w[i] = prob[ref_sel[i]];
        sum += router_w[i];
    }
    for (int i = 0; i < topk; i++) router_w[i] = router_w[i] / sum * 3.0f;

    sf37_cuda_tensor *d_logits = tensor_from_host(logits, sizeof(logits));
    sf37_cuda_tensor *d_bias = tensor_from_host(bias, sizeof(bias));
    sf37_cuda_tensor *d_prob = sf37_cuda_tensor_alloc(sizeof(prob));
    sf37_cuda_tensor *d_sel = sf37_cuda_tensor_alloc(sizeof(got_sel));
    sf37_cuda_tensor *d_w = sf37_cuda_tensor_alloc(sizeof(got_w));
    if (!sf37_cuda_router_select(d_sel, d_w, d_prob, d_logits, d_bias,
                                 n_expert, topk, 3.0f)) {
        fprintf(stderr, "CUDA router_select launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(d_sel, 0, got_sel, sizeof(got_sel));
    sf37_cuda_tensor_read(d_w, 0, got_w, sizeof(got_w));
    for (int i = 0; i < topk; i++) {
        if (got_sel[i] != ref_sel[i]) {
            fprintf(stderr, "CUDA router selected mismatch: got=%d want=%d\n",
                    got_sel[i], ref_sel[i]);
            exit(1);
        }
        assert_close(got_w[i], router_w[i], 1e-6f, "cuda router weight");
    }

    const size_t q3_bytes = (size_t)n_expert * mid_dim * SF37_Q3_BLOCK_SIZE;
    const size_t q2_bytes = (size_t)n_expert * out_dim * SF37_Q2_BLOCK_SIZE;
    uint8_t *q3_gate = (uint8_t *)calloc(1, q3_bytes);
    uint8_t *q3_up = (uint8_t *)calloc(1, q3_bytes);
    uint8_t *q2_down = (uint8_t *)calloc(1, q2_bytes);
    float *gate = (float *)malloc((size_t)mid_dim * sizeof(float));
    float *up = (float *)malloc((size_t)mid_dim * sizeof(float));
    float *mid = (float *)malloc((size_t)mid_dim * sizeof(float));
    float *down = (float *)malloc((size_t)out_dim * sizeof(float));
    if (!q3_gate || !q3_up || !q2_down || !gate || !up || !mid || !down) {
        fprintf(stderr, "router_moe test allocation failed\n");
        exit(1);
    }
    for (int e = 0; e < n_expert; e++) {
        for (int r = 0; r < mid_dim; r++) {
            make_q3_row(q3_gate + ((size_t)e * mid_dim + r) * SF37_Q3_BLOCK_SIZE,
                        e * 13 + r);
            make_q3_row(q3_up + ((size_t)e * mid_dim + r) * SF37_Q3_BLOCK_SIZE,
                        e * 17 + r + 3);
        }
        for (int r = 0; r < out_dim; r++) {
            make_q2_row(q2_down + ((size_t)e * out_dim + r) * SF37_Q2_BLOCK_SIZE,
                        e * 19 + r);
        }
    }

    memset(ref, 0, sizeof(ref));
    for (int slot = 0; slot < topk; slot++) {
        const int e = ref_sel[slot];
        sf37_q3_asym_matvec(gate, q3_gate + (size_t)e * mid_dim * SF37_Q3_BLOCK_SIZE,
                            in_dim, mid_dim, x);
        sf37_q3_asym_matvec(up, q3_up + (size_t)e * mid_dim * SF37_Q3_BLOCK_SIZE,
                            in_dim, mid_dim, x);
        sf37_swiglu(mid, gate, up, mid_dim, 0.0f);
        for (int i = 0; i < mid_dim; i++) mid[i] *= router_w[slot];
        sf37_q2_asym_matvec(down, q2_down + (size_t)e * out_dim * SF37_Q2_BLOCK_SIZE,
                            mid_dim, out_dim, mid);
        for (int r = 0; r < out_dim; r++) ref[r] += down[r];
    }

    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dwg = tensor_from_host(q3_gate, q3_bytes);
    sf37_cuda_tensor *dwu = tensor_from_host(q3_up, q3_bytes);
    sf37_cuda_tensor *dwd = tensor_from_host(q2_down, q2_bytes);
    sf37_cuda_tensor *dg = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    sf37_cuda_tensor *du = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    sf37_cuda_tensor *dm = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
    sf37_cuda_tensor *dd = sf37_cuda_tensor_alloc((uint64_t)topk * out_dim * sizeof(float));
    sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
    setenv("SF37_CUDA_NO_QLOW_Q8K", "1", 1);
    if (!sf37_cuda_routed_moe_one(dout, dg, du, dm, dd,
                                  dwg, dwu, dwd,
                                  d_sel, d_w,
                                  n_expert, topk,
                                  in_dim, mid_dim, out_dim,
                                  0.0f, dx)) {
        fprintf(stderr, "CUDA routed_moe_one launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close(got[r], ref[r], 5e-3f, "cuda routed moe");
    memcpy(direct_got, got, sizeof(direct_got));
    unsetenv("SF37_CUDA_NO_QLOW_Q8K");
    setenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS", "0", 1);
    if (!sf37_cuda_routed_moe_one(dout, dg, du, dm, dd,
                                  dwg, dwu, dwd,
                                  d_sel, d_w,
                                  n_expert, topk,
                                  in_dim, mid_dim, out_dim,
                                  0.0f, dx)) {
        fprintf(stderr, "CUDA routed_moe_one q8k launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) assert_close_rel(got[r], ref[r], 5e-2f, 1e-1f, "cuda routed moe q8k");
    unsetenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS");

    setenv("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM", "1", 1);
    setenv("SF37_CUDA_NO_QLOW_Q8K", "1", 1);
    if (!sf37_cuda_routed_moe_one(dout, dg, du, dm, dd,
                                  dwg, dwu, dwd,
                                  d_sel, d_w,
                                  n_expert, topk,
                                  in_dim, mid_dim, out_dim,
                                  0.0f, dx)) {
        fprintf(stderr, "CUDA routed_moe_one fallback launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
    for (int r = 0; r < out_dim; r++) {
        assert_close(got[r], ref[r], 5e-3f, "cuda routed moe fallback");
        assert_close(got[r], direct_got[r], 1e-5f, "cuda routed moe direct-vs-fallback");
    }
    unsetenv("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM");
    unsetenv("SF37_CUDA_NO_QLOW_Q8K");

    sf37_cuda_tensor_free(dout);
    sf37_cuda_tensor_free(dd);
    sf37_cuda_tensor_free(dm);
    sf37_cuda_tensor_free(du);
    sf37_cuda_tensor_free(dg);
    sf37_cuda_tensor_free(dwd);
    sf37_cuda_tensor_free(dwu);
    sf37_cuda_tensor_free(dwg);
    sf37_cuda_tensor_free(dx);
    sf37_cuda_tensor_free(d_w);
    sf37_cuda_tensor_free(d_sel);
    sf37_cuda_tensor_free(d_prob);
    sf37_cuda_tensor_free(d_bias);
    sf37_cuda_tensor_free(d_logits);
    free(down);
    free(mid);
    free(up);
    free(gate);
    free(q2_down);
    free(q3_up);
    free(q3_gate);
}

static void cpu_attention_prefill_ref(float *out, const float *q,
                                      const float *k_cache, const float *v_cache,
                                      const float *head_gate, int pos0,
                                      int n_tok, int cache_cap, int q_heads,
                                      int kv_heads, int head_dim, int sliding,
                                      int window) {
    int repeat = q_heads / kv_heads;
    int kv_row = kv_heads * head_dim;
    int q_dim = q_heads * head_dim;
    for (int t = 0; t < n_tok; t++) {
        int pos = pos0 + t;
        int have = pos + 1;
        int start = have > cache_cap ? have - cache_cap : 0;
        if (sliding && have > window && have - window > start) start = have - window;
        for (int h = 0; h < q_heads; h++) {
            int kvh = h / repeat;
            const float *qh = q + t * q_dim + h * head_dim;
            float max_score = -INFINITY;
            for (int r = start; r <= pos; r++) {
                int row = r % cache_cap;
                const float *kh = k_cache + row * kv_row + kvh * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) score += qh[d] * kh[d];
                score *= 1.0f / sqrtf((float)head_dim);
                if (score > max_score) max_score = score;
            }
            float denom = 0.0f;
            float tmp[128];
            memset(tmp, 0, sizeof(tmp));
            for (int r = start; r <= pos; r++) {
                int row = r % cache_cap;
                const float *kh = k_cache + row * kv_row + kvh * head_dim;
                const float *vh = v_cache + row * kv_row + kvh * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) score += qh[d] * kh[d];
                score *= 1.0f / sqrtf((float)head_dim);
                float w = expf(score - max_score);
                denom += w;
                for (int d = 0; d < head_dim; d++) tmp[d] += w * vh[d];
            }
            float g = sigmoid_ref(head_gate[t * q_heads + h]);
            for (int d = 0; d < head_dim; d++) {
                out[t * q_dim + h * head_dim + d] = tmp[d] / denom * g;
            }
        }
    }
}

static void cpu_router_sf37_top8_batch(const float *logits, const float *bias,
                                       int n_tok, int32_t *sel, float *w,
                                       float *prob) {
    enum { n_expert = 288, topk = 8 };
    for (int t = 0; t < n_tok; t++) {
        float best_score[topk];
        float best_prob[topk];
        int32_t best_idx[topk];
        for (int k = 0; k < topk; k++) {
            best_score[k] = -INFINITY;
            best_prob[k] = 0.0f;
            best_idx[k] = -1;
        }
        for (int e = 0; e < n_expert; e++) {
            float p = sigmoid_ref(logits[t * n_expert + e]);
            if (prob) prob[t * n_expert + e] = p;
            float s = p + bias[e];
            for (int k = 0; k < topk; k++) {
                if (best_idx[k] < 0 || s > best_score[k] ||
                    (s == best_score[k] && e < best_idx[k])) {
                    for (int j = topk - 1; j > k; j--) {
                        best_score[j] = best_score[j - 1];
                        best_prob[j] = best_prob[j - 1];
                        best_idx[j] = best_idx[j - 1];
                    }
                    best_score[k] = s;
                    best_prob[k] = p;
                    best_idx[k] = e;
                    break;
                }
            }
        }
        float sum = 0.0f;
        for (int k = 0; k < topk; k++) sum += best_prob[k];
        if (sum < 1.0e-20f) sum = 1.0e-20f;
        for (int k = 0; k < topk; k++) {
            sel[t * topk + k] = best_idx[k];
            w[t * topk + k] = best_prob[k] / sum * 3.0f;
        }
    }
}

static void test_stage6_batch_helpers(void) {
    {
        enum { vocab = 5, dim = 6, n_tok = 4 };
        static uint16_t emb[vocab][dim];
        int32_t tokens[n_tok] = {3, 0, 4, 1};
        float ref[n_tok][dim], got[n_tok][dim];
        for (int t = 0; t < vocab; t++) {
            for (int d = 0; d < dim; d++) {
                emb[t][d] = f32_to_bf16((float)(t * 17 + d - 13) * 0.03125f);
            }
        }
        for (int t = 0; t < n_tok; t++) {
            for (int d = 0; d < dim; d++) ref[t][d] = sf37_bf16_to_f32(emb[tokens[t]][d]);
        }
        if (!sf37_cuda_set_model_map_range(emb, sizeof(emb), 0, sizeof(emb), sizeof(emb))) {
            fprintf(stderr, "CUDA set model map for embed batch failed\n");
            exit(1);
        }
        sf37_cuda_tensor *dtok = tensor_from_host(tokens, sizeof(tokens));
        sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
        if (!sf37_cuda_embed_tokens_bf16_mapped(dout, emb, sizeof(emb), 0,
                                                dim, vocab, dtok, n_tok)) {
            fprintf(stderr, "CUDA embed batch launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
        for (int i = 0; i < n_tok * dim; i++) assert_close(((float *)got)[i], ((float *)ref)[i], 0.0f, "cuda embed batch");
        sf37_cuda_tensor_free(dout);
        sf37_cuda_tensor_free(dtok);
    }

    {
        enum { in_dim = 6, out_dim = 5, n_tok = 3 };
        float x[n_tok][in_dim], bf_ref[n_tok][out_dim], bf_got[n_tok][out_dim];
        static uint16_t bw[out_dim][in_dim];
        static float fw[out_dim][in_dim];
        float f_ref[n_tok][out_dim], f_got[n_tok][out_dim];
        for (int t = 0; t < n_tok; t++) {
            for (int i = 0; i < in_dim; i++) x[t][i] = (float)((t * 7 + i) % 17 - 8) * 0.0625f;
        }
        for (int r = 0; r < out_dim; r++) {
            for (int i = 0; i < in_dim; i++) {
                float v = (float)((r * 5 + i) % 19 - 9) * 0.03125f;
                bw[r][i] = f32_to_bf16(v);
                fw[r][i] = v * 0.5f;
            }
        }
        for (int t = 0; t < n_tok; t++) {
            sf37_bf16_matvec(bf_ref[t], &bw[0][0], in_dim, out_dim, x[t]);
            for (int r = 0; r < out_dim; r++) {
                float acc = 0.0f;
                for (int i = 0; i < in_dim; i++) acc += fw[r][i] * x[t][i];
                f_ref[t][r] = acc;
            }
        }
        sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
        sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(bf_got));
        if (!sf37_cuda_set_model_map_range(bw, sizeof(bw), 0, sizeof(bw), sizeof(bw)) ||
            !sf37_cuda_matmul_bf16_mapped(dout, bw, sizeof(bw), 0,
                                          in_dim, out_dim, dx, n_tok)) {
            fprintf(stderr, "CUDA bf16 batch matmul launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, bf_got, sizeof(bf_got));
        for (int i = 0; i < n_tok * out_dim; i++) assert_close(((float *)bf_got)[i], ((float *)bf_ref)[i], 1e-4f, "cuda bf16 batch matmul");
        sf37_cuda_tensor_free(dout);
        dout = sf37_cuda_tensor_alloc(sizeof(f_got));
        if (!sf37_cuda_set_model_map_range(fw, sizeof(fw), 0, sizeof(fw), sizeof(fw)) ||
            !sf37_cuda_matmul_f32_mapped(dout, fw, sizeof(fw), 0,
                                         in_dim, out_dim, dx, n_tok)) {
            fprintf(stderr, "CUDA f32 batch matmul launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, f_got, sizeof(f_got));
        for (int i = 0; i < n_tok * out_dim; i++) assert_close(((float *)f_got)[i], ((float *)f_ref)[i], 1e-5f, "cuda f32 batch matmul");
        sf37_cuda_tensor_free(dout);
        sf37_cuda_tensor_free(dx);
    }

    {
        enum { in_dim = 33, out_dim = 5, blocks = (in_dim + 31) / 32, n_tok = 3 };
        float x[n_tok][in_dim], ref0[n_tok][out_dim], ref1[n_tok][out_dim];
        float got0[n_tok][out_dim], got1[n_tok][out_dim];
        static uint8_t weights[2][out_dim * blocks * SF37_Q8_BLOCK_SIZE];
        int8_t q[out_dim][in_dim];
        float scale[out_dim][blocks];
        for (int t = 0; t < n_tok; t++) {
            for (int i = 0; i < in_dim; i++) x[t][i] = (float)((t * 11 + i) % 23 - 11) * 0.04f;
        }
        for (int r = 0; r < out_dim; r++) {
            for (int b = 0; b < blocks; b++) scale[r][b] = 0.02f * (float)(r + b + 1);
            for (int i = 0; i < in_dim; i++) q[r][i] = (int8_t)((r + i) % 9 - 4);
            make_q8_row(weights[0] + r * blocks * SF37_Q8_BLOCK_SIZE, q[r], scale[r], in_dim);
            for (int i = 0; i < in_dim; i++) q[r][i] = (int8_t)((r * 3 + i) % 11 - 5);
            make_q8_row(weights[1] + r * blocks * SF37_Q8_BLOCK_SIZE, q[r], scale[r], in_dim);
        }
        for (int t = 0; t < n_tok; t++) {
            sf37_q8_0_matvec(ref0[t], weights[0], in_dim, out_dim, x[t]);
            sf37_q8_0_matvec(ref1[t], weights[1], in_dim, out_dim, x[t]);
        }
        sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
        sf37_cuda_tensor *d0 = sf37_cuda_tensor_alloc(sizeof(got0));
        sf37_cuda_tensor *d1 = sf37_cuda_tensor_alloc(sizeof(got1));
        setenv("SF37_CUDA_NO_Q8_F16_CACHE", "1", 1);
        setenv("SF37_CUDA_NO_Q8_F32_CACHE", "1", 1);
        if (!sf37_cuda_set_model_map_range(weights, sizeof(weights), 0, sizeof(weights), sizeof(weights)) ||
            !sf37_cuda_matmul_q8_0_pair_mapped(d0, d1, weights, sizeof(weights),
                                               0, sizeof(weights[0]),
                                               in_dim, out_dim, dx, n_tok)) {
            fprintf(stderr, "CUDA q8 pair batch mapped launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(d0, 0, got0, sizeof(got0));
        sf37_cuda_tensor_read(d1, 0, got1, sizeof(got1));
        for (int i = 0; i < n_tok * out_dim; i++) {
            assert_close(((float *)got0)[i], ((float *)ref0)[i], 1e-4f, "cuda q8 pair batch0");
            assert_close(((float *)got1)[i], ((float *)ref1)[i], 1e-4f, "cuda q8 pair batch1");
        }
        unsetenv("SF37_CUDA_NO_Q8_F16_CACHE");
        unsetenv("SF37_CUDA_NO_Q8_F32_CACHE");
        sf37_cuda_tensor_free(d1);
        sf37_cuda_tensor_free(d0);
        sf37_cuda_tensor_free(dx);
    }

    {
        enum { n_tok = 3, n_head = 2, head_dim = 8, rotary_dim = 4 };
        float x[n_tok][n_head * head_dim], ref[n_tok][n_head * head_dim];
        float got[n_tok][n_head * head_dim], wf[head_dim];
        static uint16_t wb[head_dim];
        for (int i = 0; i < n_tok * n_head * head_dim; i++) ((float *)x)[i] = (float)((i % 29) - 14) * 0.03125f;
        for (int i = 0; i < head_dim; i++) {
            wb[i] = f32_to_bf16((float)((i % 7) - 3) * 0.05f);
            wf[i] = sf37_bf16_to_f32(wb[i]);
        }
        memcpy(ref, x, sizeof(ref));
        for (int t = 0; t < n_tok; t++) {
            for (int h = 0; h < n_head; h++) {
                sf37_rms_norm_weight1(ref[t] + h * head_dim,
                                      ref[t] + h * head_dim,
                                      wf, head_dim, 1e-5f);
            }
            cpu_rope_split_half(ref[t], n_head, head_dim, rotary_dim, 10000.0, 5 + t);
        }
        sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
        if (!sf37_cuda_set_model_map_range(wb, sizeof(wb), 0, sizeof(wb), sizeof(wb)) ||
            !sf37_cuda_head_rms_norm_weight1_bf16_batch_mapped(dx, wb, sizeof(wb), 0,
                                                               n_tok, n_head, head_dim, 1e-5f) ||
            !sf37_cuda_rope_split_half_batch(dx, n_tok, n_head, head_dim,
                                             rotary_dim, 10000.0, 0, 5)) {
            fprintf(stderr, "CUDA head norm/rope batch launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
        for (int i = 0; i < n_tok * n_head * head_dim; i++) assert_close(((float *)got)[i], ((float *)ref)[i], 3e-5f, "cuda head rope batch");
        sf37_cuda_tensor_free(dx);
    }

    {
        enum { n_tok = 3, pos0 = 2, q_heads = 8, kv_heads = 2, head_dim = 128, cache_cap = 8 };
        enum { q_dim = q_heads * head_dim, kv_dim = kv_heads * head_dim };
        float q[n_tok][q_dim], k[n_tok][kv_dim], v[n_tok][kv_dim], gate[n_tok][q_heads];
        float k_cache[cache_cap][kv_dim], v_cache[cache_cap][kv_dim];
        float ref[n_tok][q_dim], got[n_tok][q_dim];
        for (int i = 0; i < n_tok * q_dim; i++) ((float *)q)[i] = (float)((i % 37) - 18) * 0.006f;
        for (int i = 0; i < n_tok * kv_dim; i++) {
            ((float *)k)[i] = (float)((i % 31) - 15) * 0.005f;
            ((float *)v)[i] = (float)((i % 29) - 14) * 0.007f;
        }
        for (int i = 0; i < n_tok * q_heads; i++) ((float *)gate)[i] = (float)((i % 11) - 5) * 0.09f;
        for (int i = 0; i < cache_cap * kv_dim; i++) {
            ((float *)k_cache)[i] = (float)((i % 23) - 11) * 0.004f;
            ((float *)v_cache)[i] = (float)((i % 19) - 9) * 0.008f;
        }
        for (int t = 0; t < n_tok; t++) {
            memcpy(k_cache[(pos0 + t) % cache_cap], k[t], sizeof(k[t]));
            memcpy(v_cache[(pos0 + t) % cache_cap], v[t], sizeof(v[t]));
        }
        cpu_attention_prefill_ref(&ref[0][0], &q[0][0], &k_cache[0][0], &v_cache[0][0],
                                  &gate[0][0], pos0, n_tok, cache_cap,
                                  q_heads, kv_heads, head_dim, 0, 512);
        sf37_cuda_tensor *dq = tensor_from_host(q, sizeof(q));
        sf37_cuda_tensor *dk = tensor_from_host(k, sizeof(k));
        sf37_cuda_tensor *dv = tensor_from_host(v, sizeof(v));
        sf37_cuda_tensor *dg = tensor_from_host(gate, sizeof(gate));
        sf37_cuda_tensor *dkc = tensor_from_host(k_cache, sizeof(k_cache));
        sf37_cuda_tensor *dvc = tensor_from_host(v_cache, sizeof(v_cache));
        sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
        if (!sf37_cuda_store_kv_cache_batch(dkc, dvc, dk, dv, pos0, n_tok, cache_cap, kv_dim) ||
            !sf37_cuda_attention_prefill_heads(dout, dq, dkc, dvc, dg, pos0, n_tok,
                                               cache_cap, q_heads, kv_heads, head_dim, 0, 512)) {
            fprintf(stderr, "CUDA attention prefill batch launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
        for (int i = 0; i < n_tok * q_dim; i++) assert_close(((float *)got)[i], ((float *)ref)[i], 6e-5f, "cuda attention prefill batch");
        sf37_cuda_tensor_free(dout);
        sf37_cuda_tensor_free(dvc);
        sf37_cuda_tensor_free(dkc);
        sf37_cuda_tensor_free(dg);
        sf37_cuda_tensor_free(dv);
        sf37_cuda_tensor_free(dk);
        sf37_cuda_tensor_free(dq);
    }

    {
        enum { n_tok = 3, n_expert = 288, topk = 8 };
        float logits[n_tok][n_expert], prob[n_tok][n_expert], got_prob[n_tok][n_expert];
        static float bias[n_expert];
        float ref_w[n_tok][topk], got_w[n_tok][topk];
        int32_t ref_sel[n_tok][topk], got_sel[n_tok][topk];
        for (int e = 0; e < n_expert; e++) bias[e] = (float)((e * 13) % 23 - 11) * 0.003f;
        for (int t = 0; t < n_tok; t++) {
            for (int e = 0; e < n_expert; e++) logits[t][e] = (float)(((t + 1) * e * 7) % 61 - 30) * 0.021f;
            logits[t][40] = 5.0f;
            logits[t][41] = 5.0f;
            bias[40] = 0.5f;
            bias[41] = 0.5f;
        }
        cpu_router_sf37_top8_batch(&logits[0][0], bias, n_tok, &ref_sel[0][0], &ref_w[0][0], &prob[0][0]);
        if (!sf37_cuda_set_model_map_range(bias, sizeof(bias), 0, sizeof(bias), sizeof(bias))) {
            fprintf(stderr, "CUDA set model map for router batch failed\n");
            exit(1);
        }
        sf37_cuda_tensor *dl = tensor_from_host(logits, sizeof(logits));
        sf37_cuda_tensor *dp = sf37_cuda_tensor_alloc(sizeof(got_prob));
        sf37_cuda_tensor *ds = sf37_cuda_tensor_alloc(sizeof(got_sel));
        sf37_cuda_tensor *dw = sf37_cuda_tensor_alloc(sizeof(got_w));
        if (!sf37_cuda_router_select_batch_mapped(ds, dw, dp, dl, bias, sizeof(bias), 0,
                                                  n_expert, topk, 3.0f, n_tok)) {
            fprintf(stderr, "CUDA router batch launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(ds, 0, got_sel, sizeof(got_sel));
        sf37_cuda_tensor_read(dw, 0, got_w, sizeof(got_w));
        sf37_cuda_tensor_read(dp, 0, got_prob, sizeof(got_prob));
        for (int i = 0; i < n_tok * topk; i++) {
            if (((int32_t *)got_sel)[i] != ((int32_t *)ref_sel)[i]) {
                fprintf(stderr, "CUDA router batch selected mismatch at %d: got=%d want=%d\n",
                        i, ((int32_t *)got_sel)[i], ((int32_t *)ref_sel)[i]);
                exit(1);
            }
            assert_close(((float *)got_w)[i], ((float *)ref_w)[i], 1e-6f, "cuda router batch weight");
        }
        for (int i = 0; i < n_tok * n_expert; i++) assert_close(((float *)got_prob)[i], ((float *)prob)[i], 1e-6f, "cuda router batch prob");
        setenv("SF37_CUDA_NO_BATCH_ROUTER", "1", 1);
        if (!sf37_cuda_router_select_batch_mapped(ds, dw, dp, dl, bias, sizeof(bias), 0,
                                                  n_expert, topk, 3.0f, n_tok)) {
            fprintf(stderr, "CUDA router batch fallback launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(ds, 0, got_sel, sizeof(got_sel));
        for (int i = 0; i < n_tok * topk; i++) {
            if (((int32_t *)got_sel)[i] != ((int32_t *)ref_sel)[i]) {
                fprintf(stderr, "CUDA router batch fallback selected mismatch at %d\n", i);
                exit(1);
            }
        }
        unsetenv("SF37_CUDA_NO_BATCH_ROUTER");
        sf37_cuda_tensor_free(dw);
        sf37_cuda_tensor_free(ds);
        sf37_cuda_tensor_free(dp);
        sf37_cuda_tensor_free(dl);
    }

    {
        enum {
            n_tok = 3,
            n_expert = 4,
            topk = 2,
            in_dim = 256,
            mid_dim = 256,
            out_dim = 6,
            q3_bytes = n_expert * mid_dim * SF37_Q3_BLOCK_SIZE,
            q2_bytes = n_expert * out_dim * SF37_Q2_BLOCK_SIZE,
            model_bytes = q3_bytes * 2 + q2_bytes
        };
        static uint8_t moe_model[model_bytes];
        uint8_t *q3_gate = moe_model;
        uint8_t *q3_up = moe_model + q3_bytes;
        uint8_t *q2_down = moe_model + q3_bytes * 2;
        float x[n_tok][in_dim];
        int32_t selected[n_tok][topk] = {
            {3, 1},
            {2, -1},
            {1, 3},
        };
        float router_w[n_tok][topk] = {
            {1.1f, 0.7f},
            {0.9f, 1.3f},
            {1.4f, 0.6f},
        };
        float ref[n_tok][out_dim], got[n_tok][out_dim], got_sorted[n_tok][out_dim];
        float got_fallback[n_tok][out_dim], got_scratch[n_tok][out_dim];
        float *gate = (float *)malloc((size_t)mid_dim * sizeof(float));
        float *up = (float *)malloc((size_t)mid_dim * sizeof(float));
        float *mid = (float *)malloc((size_t)mid_dim * sizeof(float));
        float *down = (float *)malloc((size_t)out_dim * sizeof(float));
        if (!gate || !up || !mid || !down) {
            fprintf(stderr, "batch routed_moe test allocation failed\n");
            exit(1);
        }
        for (int t = 0; t < n_tok; t++) {
            for (int i = 0; i < in_dim; i++) {
                x[t][i] = (float)(((t + 2) * (i % 41) - 33) % 47) * 0.011f;
            }
        }
        for (int e = 0; e < n_expert; e++) {
            for (int r = 0; r < mid_dim; r++) {
                make_q3_row(q3_gate + ((size_t)e * mid_dim + r) * SF37_Q3_BLOCK_SIZE,
                            e * 23 + r + 1);
                make_q3_row(q3_up + ((size_t)e * mid_dim + r) * SF37_Q3_BLOCK_SIZE,
                            e * 29 + r + 5);
            }
            for (int r = 0; r < out_dim; r++) {
                make_q2_row(q2_down + ((size_t)e * out_dim + r) * SF37_Q2_BLOCK_SIZE,
                            e * 31 + r + 7);
            }
        }
        memset(ref, 0, sizeof(ref));
        for (int t = 0; t < n_tok; t++) {
            for (int slot = 0; slot < topk; slot++) {
                int e = selected[t][slot];
                if (e < 0 || e >= n_expert) e = 0;
                sf37_q3_asym_matvec(gate, q3_gate + (size_t)e * mid_dim * SF37_Q3_BLOCK_SIZE,
                                    in_dim, mid_dim, x[t]);
                sf37_q3_asym_matvec(up, q3_up + (size_t)e * mid_dim * SF37_Q3_BLOCK_SIZE,
                                    in_dim, mid_dim, x[t]);
                sf37_swiglu(mid, gate, up, mid_dim, 0.0f);
                for (int i = 0; i < mid_dim; i++) mid[i] *= router_w[t][slot];
                sf37_q2_asym_matvec(down, q2_down + (size_t)e * out_dim * SF37_Q2_BLOCK_SIZE,
                                    mid_dim, out_dim, mid);
                for (int r = 0; r < out_dim; r++) ref[t][r] += down[r];
            }
        }

        if (!sf37_cuda_set_model_map_range(moe_model, sizeof(moe_model), 0,
                                           sizeof(moe_model), sizeof(moe_model))) {
            fprintf(stderr, "CUDA set model map for batch routed_moe failed\n");
            exit(1);
        }
        sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
        sf37_cuda_tensor *ds = tensor_from_host(selected, sizeof(selected));
        sf37_cuda_tensor *dw = tensor_from_host(router_w, sizeof(router_w));
        sf37_cuda_tensor *dg = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
        sf37_cuda_tensor *du = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
        sf37_cuda_tensor *dm = sf37_cuda_tensor_alloc((uint64_t)topk * mid_dim * sizeof(float));
        sf37_cuda_tensor *dd = sf37_cuda_tensor_alloc((uint64_t)topk * out_dim * sizeof(float));
        sf37_cuda_tensor *dout = sf37_cuda_tensor_alloc(sizeof(got));
        setenv("SF37_CUDA_QLOW_Q8K", "1", 1);
        setenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS", "0", 1);
        unsetenv("SF37_CUDA_NO_MOE_SORTED_PAIRS");
        unsetenv("SF37_CUDA_MOE_NO_EXPERT_TILES");
        unsetenv("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM");
        if (!sf37_cuda_routed_moe_batch_mapped(dout, dg, du, dm, dd,
                                               moe_model, sizeof(moe_model),
                                               0, q3_bytes, q3_bytes * 2,
                                               ds, dw, n_expert, topk,
                                               in_dim, mid_dim, out_dim,
                                               0.0f, dx, n_tok)) {
            fprintf(stderr, "CUDA batch routed_moe expert-tile launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got, sizeof(got));
        for (int i = 0; i < n_tok * out_dim; i++) {
            assert_close_rel(((float *)got)[i], ((float *)ref)[i], 7e-2f, 1.5e-1f, "cuda batch routed moe q8k tile");
        }

        setenv("SF37_CUDA_MOE_NO_EXPERT_TILES", "1", 1);
        if (!sf37_cuda_routed_moe_batch_mapped(dout, dg, du, dm, dd,
                                               moe_model, sizeof(moe_model),
                                               0, q3_bytes, q3_bytes * 2,
                                               ds, dw, n_expert, topk,
                                               in_dim, mid_dim, out_dim,
                                               0.0f, dx, n_tok)) {
            fprintf(stderr, "CUDA batch routed_moe sorted launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got_sorted, sizeof(got_sorted));
        for (int i = 0; i < n_tok * out_dim; i++) {
            assert_close_rel(((float *)got_sorted)[i], ((float *)got)[i], 2e-3f, 2e-3f, "cuda batch routed moe tile-vs-sorted");
        }

        setenv("SF37_CUDA_NO_MOE_SORTED_PAIRS", "1", 1);
        unsetenv("SF37_CUDA_MOE_NO_EXPERT_TILES");
        if (!sf37_cuda_routed_moe_batch_mapped(dout, dg, du, dm, dd,
                                               moe_model, sizeof(moe_model),
                                               0, q3_bytes, q3_bytes * 2,
                                               ds, dw, n_expert, topk,
                                               in_dim, mid_dim, out_dim,
                                               0.0f, dx, n_tok)) {
            fprintf(stderr, "CUDA batch routed_moe fallback launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got_fallback, sizeof(got_fallback));
        for (int i = 0; i < n_tok * out_dim; i++) {
            assert_close_rel(((float *)got_fallback)[i], ((float *)got)[i], 2e-3f, 2e-3f, "cuda batch routed moe sorted-vs-fallback");
        }

        unsetenv("SF37_CUDA_NO_MOE_SORTED_PAIRS");
        setenv("SF37_CUDA_MOE_NO_EXPERT_TILES", "1", 1);
        setenv("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM", "1", 1);
        if (!sf37_cuda_routed_moe_batch_mapped(dout, dg, du, dm, dd,
                                               moe_model, sizeof(moe_model),
                                               0, q3_bytes, q3_bytes * 2,
                                               ds, dw, n_expert, topk,
                                               in_dim, mid_dim, out_dim,
                                               0.0f, dx, n_tok)) {
            fprintf(stderr, "CUDA batch routed_moe scratch-sum launch failed\n");
            exit(1);
        }
        sf37_cuda_tensor_read(dout, 0, got_scratch, sizeof(got_scratch));
        for (int i = 0; i < n_tok * out_dim; i++) {
            assert_close_rel(((float *)got_scratch)[i], ((float *)got)[i], 2e-3f, 2e-3f, "cuda batch routed moe scratch-vs-direct");
        }
        unsetenv("SF37_CUDA_QLOW_Q8K");
        unsetenv("SF37_CUDA_QLOW_Q8K_MIN_ROWS");
        unsetenv("SF37_CUDA_MOE_NO_EXPERT_TILES");
        unsetenv("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM");

        sf37_cuda_tensor_free(dout);
        sf37_cuda_tensor_free(dd);
        sf37_cuda_tensor_free(dm);
        sf37_cuda_tensor_free(du);
        sf37_cuda_tensor_free(dg);
        sf37_cuda_tensor_free(dw);
        sf37_cuda_tensor_free(ds);
        sf37_cuda_tensor_free(dx);
        free(down);
        free(mid);
        free(up);
        free(gate);
    }
}

static void cpu_layer_norm(float *out, const float *x,
                           const float *w, const float *b,
                           int rows, int n, float eps) {
    for (int r = 0; r < rows; r++) {
        const float *xr = x + (size_t)r * n;
        float *orow = out + (size_t)r * n;
        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += xr[i];
        mean /= (float)n;
        float var = 0.0f;
        for (int i = 0; i < n; i++) {
            float d = xr[i] - mean;
            var += d * d;
        }
        float scale = 1.0f / sqrtf(var / (float)n + eps);
        for (int i = 0; i < n; i++) orow[i] = (xr[i] - mean) * scale * w[i] + b[i];
    }
}

static float cpu_quick_gelu(float x) {
    return x / (1.0f + expf(-1.702f * x));
}

static void cpu_conv2d_nchw_to_nlc(float *out, const float *x,
                                   const float *w, const float *bias,
                                   int n_img, int in_c, int in_h, int in_w,
                                   int out_c, int kernel, int stride, int pad,
                                   int input_nlc) {
    int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    int out_w = (in_w + 2 * pad - kernel) / stride + 1;
    for (int n = 0; n < n_img; n++) {
        for (int oy = 0; oy < out_h; oy++) {
            for (int ox = 0; ox < out_w; ox++) {
                for (int oc = 0; oc < out_c; oc++) {
                    float acc = bias ? bias[oc] : 0.0f;
                    for (int ic = 0; ic < in_c; ic++) {
                        for (int ky = 0; ky < kernel; ky++) {
                            int iy = oy * stride + ky - pad;
                            if (iy < 0 || iy >= in_h) continue;
                            for (int kx = 0; kx < kernel; kx++) {
                                int ix = ox * stride + kx - pad;
                                if (ix < 0 || ix >= in_w) continue;
                                float xv;
                                if (input_nlc) {
                                    xv = x[((size_t)n * in_h * in_w + (size_t)iy * in_w + ix) * in_c + ic];
                                } else {
                                    xv = x[((size_t)n * in_c + ic) * in_h * in_w + (size_t)iy * in_w + ix];
                                }
                                float wv = w[((size_t)oc * in_c + ic) * kernel * kernel + (size_t)ky * kernel + kx];
                                acc += xv * wv;
                            }
                        }
                    }
                    out[((size_t)n * out_h * out_w + (size_t)oy * out_w + ox) * out_c + oc] = acc;
                }
            }
        }
    }
}

static float cpu_rope2d_one(float self, float mate, int d, int head_dim,
                            int row, int col, double theta) {
    int half = head_dim / 2;
    int axis_d = d < half ? d : d - half;
    int pair = axis_d / 2;
    float coord = d < half ? (float)col : (float)row;
    double inv = 1.0 / pow(theta, (double)(pair * 2) / (double)half);
    float f = (float)((double)coord * inv);
    float c = cosf(f);
    float s = sinf(f);
    return (d & 1) ? self * c + mate * s : self * c - mate * s;
}

static void cpu_vision_qkv_split_rope(float *q, float *k, float *v,
                                      const float *qkv, const float *bias,
                                      int n_tok, int grid_w,
                                      int n_heads, int head_dim,
                                      double theta) {
    int width = n_heads * head_dim;
    for (int t = 0; t < n_tok; t++) {
        int row = t / grid_w;
        int col = t - row * grid_w;
        for (int h = 0; h < n_heads; h++) {
            for (int d = 0; d < head_dim; d++) {
                int dg = h * head_dim + d;
                int dm = h * head_dim + (d ^ 1);
                float qs = qkv[(size_t)t * width * 3 + dg] + bias[dg];
                float qm = qkv[(size_t)t * width * 3 + dm] + bias[dm];
                float ks = qkv[(size_t)t * width * 3 + width + dg] + bias[width + dg];
                float km = qkv[(size_t)t * width * 3 + width + dm] + bias[width + dm];
                q[(size_t)t * width + dg] = cpu_rope2d_one(qs, qm, d, head_dim, row, col, theta);
                k[(size_t)t * width + dg] = cpu_rope2d_one(ks, km, d, head_dim, row, col, theta);
                v[(size_t)t * width + dg] = qkv[(size_t)t * width * 3 + width * 2 + dg] + bias[width * 2 + dg];
            }
        }
    }
}

static void cpu_vision_attention(float *out, const float *q, const float *k, const float *v,
                                 int n_tok, int n_heads, int head_dim) {
    int width = n_heads * head_dim;
    float *tmp = (float *)calloc((size_t)head_dim, sizeof(float));
    if (!tmp) {
        fprintf(stderr, "cpu vision attention allocation failed\n");
        exit(1);
    }
    for (int t = 0; t < n_tok; t++) {
        for (int h = 0; h < n_heads; h++) {
            const float *qh = q + (size_t)t * width + h * head_dim;
            float max_score = -INFINITY;
            for (int s = 0; s < n_tok; s++) {
                const float *kh = k + (size_t)s * width + h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                dot /= sqrtf((float)head_dim);
                if (dot > max_score) max_score = dot;
            }
            memset(tmp, 0, (size_t)head_dim * sizeof(float));
            float denom = 0.0f;
            for (int s = 0; s < n_tok; s++) {
                const float *kh = k + (size_t)s * width + h * head_dim;
                const float *vh = v + (size_t)s * width + h * head_dim;
                float dot = 0.0f;
                for (int d = 0; d < head_dim; d++) dot += qh[d] * kh[d];
                float ww = expf(dot / sqrtf((float)head_dim) - max_score);
                denom += ww;
                for (int d = 0; d < head_dim; d++) tmp[d] += ww * vh[d];
            }
            for (int d = 0; d < head_dim; d++) out[(size_t)t * width + h * head_dim + d] = tmp[d] / denom;
        }
    }
    free(tmp);
}

static void test_vision_cuda_primitives(void) {
    enum { rows = 3, n = 8 };
    float x[rows * n], wf[n], bf[n], ref[rows * n], got[rows * n];
    uint16_t wb[n], bb[n];
    for (int i = 0; i < rows * n; i++) x[i] = (float)((i % 11) - 5) * 0.17f;
    for (int i = 0; i < n; i++) {
        wf[i] = 0.7f + (float)i * 0.03f;
        bf[i] = (float)(i - 3) * 0.02f;
        wb[i] = f32_to_bf16(wf[i]);
        bb[i] = f32_to_bf16(bf[i]);
        wf[i] = sf37_bf16_to_f32(wb[i]);
        bf[i] = sf37_bf16_to_f32(bb[i]);
    }
    struct {
        uint16_t w[n];
        uint16_t b[n];
    } ln_model;
    memcpy(ln_model.w, wb, sizeof(wb));
    memcpy(ln_model.b, bb, sizeof(bb));
    cpu_layer_norm(ref, x, wf, bf, rows, n, 1e-5f);
    if (!sf37_cuda_set_model_map_range(&ln_model, sizeof(ln_model), 0, sizeof(ln_model), sizeof(ln_model))) {
        fprintf(stderr, "CUDA set model map for vision LN failed\n");
        exit(1);
    }
    sf37_cuda_tensor *dx = tensor_from_host(x, sizeof(x));
    sf37_cuda_tensor *dy = sf37_cuda_tensor_alloc(sizeof(got));
    if (!sf37_cuda_layer_norm_bf16_mapped(dy, dx, &ln_model, sizeof(ln_model),
                                          0, sizeof(ln_model.w), n, rows, 1e-5f)) {
        fprintf(stderr, "CUDA vision layernorm launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dy, 0, got, sizeof(got));
    for (int i = 0; i < rows * n; i++) assert_close(got[i], ref[i], 3e-5f, "cuda vision layernorm");

    memcpy(ref, x, sizeof(x));
    for (int r = 0; r < rows; r++) {
        for (int i = 0; i < n; i++) ref[r * n + i] += bf[i];
    }
    if (!sf37_cuda_tensor_write(dx, 0, x, sizeof(x)) ||
        !sf37_cuda_add_bias_bf16_mapped(dx, &ln_model, sizeof(ln_model),
                                        sizeof(ln_model.w), n, rows)) {
        fprintf(stderr, "CUDA vision bias launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
    for (int i = 0; i < rows * n; i++) assert_close(got[i], ref[i], 2e-5f, "cuda vision bias");

    memcpy(ref, x, sizeof(x));
    for (int i = 0; i < rows * n; i++) ref[i] += x[i] * wf[i % n];
    if (!sf37_cuda_tensor_write(dx, 0, x, sizeof(x)) ||
        !sf37_cuda_add_scaled_bf16_mapped(dx, dx, &ln_model, sizeof(ln_model),
                                          0, n, rows)) {
        fprintf(stderr, "CUDA vision add scaled launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
    for (int i = 0; i < rows * n; i++) assert_close(got[i], ref[i], 2e-5f, "cuda vision add scaled");

    for (int i = 0; i < rows * n; i++) ref[i] = cpu_quick_gelu(x[i]);
    if (!sf37_cuda_tensor_write(dx, 0, x, sizeof(x)) ||
        !sf37_cuda_quick_gelu_f32(dx, rows * n)) {
        fprintf(stderr, "CUDA vision quick_gelu launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, got, sizeof(got));
    for (int i = 0; i < rows * n; i++) assert_close(got[i], ref[i], 2e-5f, "cuda vision quick_gelu");
    sf37_cuda_tensor_free(dy);
    sf37_cuda_tensor_free(dx);

    enum { img = 1, in_c = 2, in_h = 4, in_w = 5, out_c = 3, kernel = 3, stride = 2, pad = 1 };
    enum { out_h = (in_h + 2 * pad - kernel) / stride + 1, out_w = (in_w + 2 * pad - kernel) / stride + 1 };
    float conv_x[img * in_c * in_h * in_w], conv_w[out_c * in_c * kernel * kernel], conv_b[out_c];
    float conv_ref[img * out_h * out_w * out_c], conv_got[img * out_h * out_w * out_c];
    struct {
        uint16_t w[out_c * in_c * kernel * kernel];
        uint16_t b[out_c];
    } conv_model;
    for (int i = 0; i < img * in_c * in_h * in_w; i++) conv_x[i] = (float)((i % 17) - 8) * 0.05f;
    for (int i = 0; i < out_c * in_c * kernel * kernel; i++) {
        conv_w[i] = (float)((i % 13) - 6) * 0.03125f;
        conv_model.w[i] = f32_to_bf16(conv_w[i]);
        conv_w[i] = sf37_bf16_to_f32(conv_model.w[i]);
    }
    for (int i = 0; i < out_c; i++) {
        conv_b[i] = (float)(i - 1) * 0.07f;
        conv_model.b[i] = f32_to_bf16(conv_b[i]);
        conv_b[i] = sf37_bf16_to_f32(conv_model.b[i]);
    }
    cpu_conv2d_nchw_to_nlc(conv_ref, conv_x, conv_w, conv_b,
                           img, in_c, in_h, in_w, out_c, kernel, stride, pad, 0);
    if (!sf37_cuda_set_model_map_range(&conv_model, sizeof(conv_model), 0, sizeof(conv_model), sizeof(conv_model))) {
        fprintf(stderr, "CUDA set model map for vision conv failed\n");
        exit(1);
    }
    dx = tensor_from_host(conv_x, sizeof(conv_x));
    dy = sf37_cuda_tensor_alloc(sizeof(conv_got));
    if (!sf37_cuda_vision_conv2d_bf16_mapped(dy, dx, &conv_model, sizeof(conv_model),
                                             0, sizeof(conv_model.w),
                                             img, in_c, in_h, in_w, out_c,
                                             kernel, stride, pad, 0)) {
        fprintf(stderr, "CUDA vision conv launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dy, 0, conv_got, sizeof(conv_got));
    for (int i = 0; i < img * out_h * out_w * out_c; i++) {
        assert_close(conv_got[i], conv_ref[i], 3e-5f, "cuda vision conv");
    }
    sf37_cuda_tensor_free(dy);
    sf37_cuda_tensor_free(dx);

    enum { pos_rows = 4, pos_width = 5 };
    struct {
        uint8_t pos[pos_rows][SF37_Q8_BLOCK_SIZE];
    } pos_model;
    float hidden[pos_rows * pos_width], pos_ref[pos_rows * pos_width], pos_got[pos_rows * pos_width];
    memset(hidden, 0, sizeof(hidden));
    for (int r = 0; r < pos_rows; r++) {
        int8_t q[32] = {0};
        float scale = 0.125f;
        for (int d = 0; d < pos_width; d++) {
            q[d] = (int8_t)(r * 3 + d - 5);
            pos_ref[r * pos_width + d] = scale * (float)q[d];
        }
        make_q8_row(pos_model.pos[r], q, &scale, pos_width);
    }
    if (!sf37_cuda_set_model_map_range(&pos_model, sizeof(pos_model), 0, sizeof(pos_model), sizeof(pos_model))) {
        fprintf(stderr, "CUDA set model map for vision pos failed\n");
        exit(1);
    }
    dx = tensor_from_host(hidden, sizeof(hidden));
    if (!sf37_cuda_vision_add_pos_q8_0_mapped(dx, &pos_model, sizeof(pos_model),
                                              0, pos_width, 2, 2, 2)) {
        fprintf(stderr, "CUDA vision pos launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dx, 0, pos_got, sizeof(pos_got));
    for (int i = 0; i < pos_rows * pos_width; i++) assert_close(pos_got[i], pos_ref[i], 2e-5f, "cuda vision pos");
    sf37_cuda_tensor_free(dx);

    enum { vt = 4, vh = 2, vhd = 4, vw = vh * vhd };
    float qkv[vt * vw * 3], bias[vw * 3], q_ref[vt * vw], k_ref[vt * vw], v_ref[vt * vw];
    float q_got[vt * vw], k_got[vt * vw], v_got[vt * vw], att_ref[vt * vw], att_got[vt * vw];
    struct {
        uint16_t bias[vw * 3];
    } qkv_model;
    for (int i = 0; i < vt * vw * 3; i++) qkv[i] = (float)((i % 19) - 9) * 0.04f;
    for (int i = 0; i < vw * 3; i++) {
        bias[i] = (float)((i % 7) - 3) * 0.015f;
        qkv_model.bias[i] = f32_to_bf16(bias[i]);
        bias[i] = sf37_bf16_to_f32(qkv_model.bias[i]);
    }
    cpu_vision_qkv_split_rope(q_ref, k_ref, v_ref, qkv, bias, vt, 2, vh, vhd, 10000.0);
    cpu_vision_attention(att_ref, q_ref, k_ref, v_ref, vt, vh, vhd);
    if (!sf37_cuda_set_model_map_range(&qkv_model, sizeof(qkv_model), 0, sizeof(qkv_model), sizeof(qkv_model))) {
        fprintf(stderr, "CUDA set model map for vision qkv failed\n");
        exit(1);
    }
    sf37_cuda_tensor *dqkv = tensor_from_host(qkv, sizeof(qkv));
    sf37_cuda_tensor *dq = sf37_cuda_tensor_alloc(sizeof(q_got));
    sf37_cuda_tensor *dk = sf37_cuda_tensor_alloc(sizeof(k_got));
    sf37_cuda_tensor *dv = sf37_cuda_tensor_alloc(sizeof(v_got));
    sf37_cuda_tensor *datt = sf37_cuda_tensor_alloc(sizeof(att_got));
    if (!sf37_cuda_vision_qkv_split_rope_bf16_mapped(dq, dk, dv, dqkv,
                                                     &qkv_model, sizeof(qkv_model),
                                                     0, vt, 2, 2, vh, vhd, 2, 10000.0) ||
        !sf37_cuda_vision_attention(datt, dq, dk, dv, vt, vh, vhd)) {
        fprintf(stderr, "CUDA vision qkv/attention launch failed\n");
        exit(1);
    }
    sf37_cuda_tensor_read(dq, 0, q_got, sizeof(q_got));
    sf37_cuda_tensor_read(dk, 0, k_got, sizeof(k_got));
    sf37_cuda_tensor_read(dv, 0, v_got, sizeof(v_got));
    sf37_cuda_tensor_read(datt, 0, att_got, sizeof(att_got));
    for (int i = 0; i < vt * vw; i++) {
        assert_close(q_got[i], q_ref[i], 3e-5f, "cuda vision q rope");
        assert_close(k_got[i], k_ref[i], 3e-5f, "cuda vision k rope");
        assert_close(v_got[i], v_ref[i], 3e-5f, "cuda vision v split");
        assert_close(att_got[i], att_ref[i], 4e-5f, "cuda vision attention");
    }
    sf37_cuda_tensor_free(datt);
    sf37_cuda_tensor_free(dv);
    sf37_cuda_tensor_free(dk);
    sf37_cuda_tensor_free(dq);
    sf37_cuda_tensor_free(dqkv);
}

int main(void) {
    int devices = sf37_cuda_device_count();
    if (devices <= 0) {
        fprintf(stderr, "sf37 CUDA tests skipped: no CUDA device\n");
        return 0;
    }
    if (!sf37_cuda_init()) return 1;
    test_rms_swiglu();
    test_head_rope_attention();
    test_q8_bf16();
    test_q3_q2();
    test_router_warp_top8();
    test_router_moe();
    test_stage6_batch_helpers();
    test_vision_cuda_primitives();
    if (!sf37_cuda_synchronize()) return 1;
    sf37_cuda_cleanup();
    printf("sf37 CUDA op tests passed\n");
    return 0;
}
