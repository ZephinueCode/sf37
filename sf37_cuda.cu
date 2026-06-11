#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "sf37_cuda.h"
#include "sf37_quant.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#include <unordered_map>
#include <vector>

#define SF37_CUDA_MAX_STAT_LAYERS 64
#define SF37_CUDA_OUTPUT_LAYER 45

struct sf37_cuda_tensor {
    void *ptr;
    uint64_t bytes;
};

static void *g_tmp;
static uint64_t g_tmp_bytes;
static cublasHandle_t g_cublas;
static int g_cublas_ready;

static const void *g_model_host_base;
static const char *g_model_device_base;
static uint64_t g_model_registered_size;
static int g_model_registered;
static int g_model_device_owned;
static int g_model_range_mapping_supported = 1;
static int g_model_hmm_direct;
static int g_model_fd = -1;
static const void *g_model_fd_host_base;
static int g_model_direct_fd = -1;
static uint64_t g_model_direct_align = 1;
static uint64_t g_model_file_size;
static int g_model_cache_full;
static int g_model_preload_only;
static int g_model_mapping_failure_notice_printed;
static cudaStream_t g_model_prefetch_stream;
static cudaStream_t g_model_upload_stream;

struct cuda_model_range {
    const void *host_base;
    uint64_t offset;
    uint64_t bytes;
    char *device_ptr;
    void *registered_base;
    char *registered_device_base;
    uint64_t registered_bytes;
    int host_registered;
    int arena_allocated;
};

struct cuda_model_arena {
    char *device_ptr;
    uint64_t bytes;
    uint64_t used;
};

struct cuda_q8_f16_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    __half *device_ptr;
};

struct cuda_q8_f32_range {
    const void *host_base;
    uint64_t offset;
    uint64_t weight_bytes;
    uint64_t in_dim;
    uint64_t out_dim;
    float *device_ptr;
};

static std::vector<cuda_model_range> g_model_ranges;
static std::vector<cuda_model_arena> g_model_arenas;
static std::unordered_map<uint64_t, size_t> g_model_range_by_offset;
static std::vector<cuda_q8_f16_range> g_q8_f16_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f16_by_offset;
static std::vector<cuda_q8_f32_range> g_q8_f32_ranges;
static std::unordered_map<uint64_t, size_t> g_q8_f32_by_offset;
static uint64_t g_q8_f16_bytes;
static uint64_t g_q8_f32_bytes;
static int g_q8_f16_disabled_after_oom;
static int g_q8_f16_budget_notice_printed;
static uint64_t g_model_range_bytes;
static uint64_t g_model_load_progress_next;
static double g_model_load_progress_last;
static int g_model_load_progress_started;
static int g_model_load_progress_tty;
static void *g_model_stage_raw[4];
static void *g_model_stage[4];
static cudaEvent_t g_model_stage_event[4];
static uint64_t g_model_stage_bytes;
static uint64_t g_model_cache_hits;
static uint64_t g_model_cache_misses;
static uint64_t g_model_cache_load_bytes;
static uint64_t g_model_cache_evict_count;
static uint64_t g_model_cache_evict_bytes;
static int g_active_layer = -1;

struct cuda_layer_residency_stat {
    uint64_t loaded_bytes;
    uint64_t reload_count;
    uint64_t evict_count;
    uint64_t q8_expanded_hits;
    uint64_t q8_expanded_misses;
    double moe_load_sec;
};

static cuda_layer_residency_stat g_layer_stats[SF37_CUDA_MAX_STAT_LAYERS];
static std::unordered_map<uint64_t, uint64_t> g_model_load_count_by_offset;

static int cuda_ok(cudaError_t err, const char *what) {
    if (err == cudaSuccess) return 1;
    fprintf(stderr, "sf37: CUDA %s failed: %s\n",
            what ? what : "operation", cudaGetErrorString(err));
    (void)cudaGetLastError();
    return 0;
}

static int cublas_ok(cublasStatus_t st, const char *what) {
    if (st == CUBLAS_STATUS_SUCCESS) return 1;
    fprintf(stderr, "sf37: cuBLAS %s failed: status %d\n",
            what ? what : "operation", (int)st);
    return 0;
}

static const char *cuda_env_value(const char *sf37_name, const char *ds4_name) {
    const char *v = sf37_name ? getenv(sf37_name) : NULL;
    if (v && v[0]) return v;
    v = ds4_name ? getenv(ds4_name) : NULL;
    if (v && v[0]) return v;
    return NULL;
}

static int cuda_env_present(const char *sf37_name, const char *ds4_name) {
    return (sf37_name && getenv(sf37_name) != NULL) ||
           (ds4_name && getenv(ds4_name) != NULL);
}

static int cuda_q8_use_dp4a(void) {
    return !cuda_env_present("SF37_CUDA_NO_DP4A", "DS4_CUDA_NO_DP4A");
}

static int cuda_qlow_q8k_enabled(void) {
    return cuda_q8_use_dp4a() &&
           !cuda_env_present("SF37_CUDA_NO_QLOW_Q8K",
                             "DS4_CUDA_NO_QLOW_Q8K");
}

static uint64_t cuda_qlow_q8k_min_rows(void) {
    uint64_t v = 512;
    const char *env = cuda_env_value("SF37_CUDA_QLOW_Q8K_MIN_ROWS",
                                     "DS4_CUDA_QLOW_Q8K_MIN_ROWS");
    if (env) {
        char *end = NULL;
        unsigned long long parsed = strtoull(env, &end, 10);
        if (end != env) v = (uint64_t)parsed;
    }
    return v;
}

static int cuda_qlow_q8k_use_for(uint64_t out_dim) {
    return cuda_qlow_q8k_enabled() && out_dim >= cuda_qlow_q8k_min_rows();
}

static int cuda_bf16_cublas_enabled(void) {
    return !cuda_env_present("SF37_CUDA_NO_BF16_CUBLAS",
                             "DS4_CUDA_NO_BF16_CUBLAS");
}

static uint64_t cuda_bf16_cublas_min_out_dim(void) {
    uint64_t v = 1024;
    const char *env = cuda_env_value("SF37_CUDA_BF16_CUBLAS_MIN_OUT",
                                     "DS4_CUDA_BF16_CUBLAS_MIN_OUT");
    if (env) {
        char *end = NULL;
        unsigned long long parsed = strtoull(env, &end, 10);
        if (end != env) v = (uint64_t)parsed;
    }
    return v;
}

static int cuda_moe_profile_enabled(void) {
    return cuda_env_present("SF37_CUDA_MOE_PROFILE", "DS4_CUDA_MOE_PROFILE");
}

static int cuda_moe_write_gate_up(void) {
    return cuda_env_present("SF37_CUDA_MOE_WRITE_GATE_UP", "DS4_CUDA_MOE_WRITE_GATE_UP");
}

static int cuda_moe_direct_down_sum_enabled(void) {
    return !cuda_env_present("SF37_CUDA_NO_DIRECT_MOE_DOWN_SUM",
                             "DS4_CUDA_MOE_NO_DIRECT_DOWN_SUM6");
}

static int cuda_moe_sorted_pairs_enabled(void) {
    return !cuda_env_present("SF37_CUDA_NO_MOE_SORTED_PAIRS",
                             "DS4_CUDA_MOE_NO_SORTED_PAIRS");
}

static int cuda_moe_expert_tiles_enabled(void) {
    return !cuda_env_present("SF37_CUDA_MOE_NO_EXPERT_TILES",
                             "DS4_CUDA_MOE_NO_EXPERT_TILES");
}

static int cuda_moe_selected_expert_cache_enabled(void) {
    return !cuda_env_present("SF37_CUDA_NO_MOE_SELECTED_CACHE",
                             "DS4_CUDA_NO_MOE_SELECTED_CACHE");
}

static int cuda_label_is_moe(const char *what) {
    return what && (strstr(what, "moe") || strstr(what, "routed"));
}

static void cuda_residency_note_load(uint64_t offset,
                                     uint64_t bytes,
                                     const char *what,
                                     double sec) {
    if (g_active_layer < 0 || g_active_layer >= SF37_CUDA_MAX_STAT_LAYERS) return;
    cuda_layer_residency_stat &s = g_layer_stats[g_active_layer];
    s.loaded_bytes += bytes;
    uint64_t &cnt = g_model_load_count_by_offset[offset];
    if (cnt > 0) s.reload_count++;
    cnt++;
    if (cuda_label_is_moe(what)) s.moe_load_sec += sec;
}

static void cuda_residency_note_q8_expanded(int hit) {
    if (g_active_layer < 0 || g_active_layer >= SF37_CUDA_MAX_STAT_LAYERS) return;
    cuda_layer_residency_stat &s = g_layer_stats[g_active_layer];
    if (hit) s.q8_expanded_hits++;
    else s.q8_expanded_misses++;
}

static void cuda_residency_print_layer_stats(void) {
    for (uint32_t i = 0; i < SF37_CUDA_MAX_STAT_LAYERS; i++) {
        const cuda_layer_residency_stat &s = g_layer_stats[i];
        if (s.loaded_bytes == 0 &&
            s.reload_count == 0 &&
            s.evict_count == 0 &&
            s.q8_expanded_hits == 0 &&
            s.q8_expanded_misses == 0 &&
            s.moe_load_sec == 0.0) {
            continue;
        }
        if (i == SF37_CUDA_OUTPUT_LAYER) {
            fprintf(stderr,
                    "sf37: CUDA layer residency output loaded=%.2f MiB reloads=%llu "
                    "evicts=%llu q8_expanded_hits=%llu q8_expanded_misses=%llu "
                    "moe_load=%.3f ms\n",
                    (double)s.loaded_bytes / 1048576.0,
                    (unsigned long long)s.reload_count,
                    (unsigned long long)s.evict_count,
                    (unsigned long long)s.q8_expanded_hits,
                    (unsigned long long)s.q8_expanded_misses,
                    s.moe_load_sec * 1000.0);
        } else {
            fprintf(stderr,
                    "sf37: CUDA layer residency L%02u loaded=%.2f MiB reloads=%llu "
                    "evicts=%llu q8_expanded_hits=%llu q8_expanded_misses=%llu "
                    "moe_load=%.3f ms\n",
                    i,
                    (double)s.loaded_bytes / 1048576.0,
                    (unsigned long long)s.reload_count,
                    (unsigned long long)s.evict_count,
                    (unsigned long long)s.q8_expanded_hits,
                    (unsigned long long)s.q8_expanded_misses,
                    s.moe_load_sec * 1000.0);
        }
    }
}

static int cuda_moe_profile_start(cudaEvent_t ev[5]) {
    if (!cuda_moe_profile_enabled()) return 0;
    for (uint32_t i = 0; i < 5u; i++) {
        if (cudaEventCreate(&ev[i]) != cudaSuccess) {
            for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(ev[j]);
            memset(ev, 0, 5u * sizeof(ev[0]));
            (void)cudaGetLastError();
            return 0;
        }
    }
    (void)cudaEventRecord(ev[0], 0);
    return 1;
}

static void cuda_moe_profile_abort(cudaEvent_t ev[5]) {
    if (!ev[0]) return;
    for (uint32_t i = 0; i < 5u; i++) {
        if (ev[i]) (void)cudaEventDestroy(ev[i]);
        ev[i] = NULL;
    }
}

static void cuda_moe_profile_finish(cudaEvent_t ev[5],
                                    int q8k_path,
                                    uint32_t topk,
                                    uint32_t in_dim,
                                    uint32_t expert_mid_dim,
                                    uint32_t out_dim) {
    if (!ev[0]) return;
    if (cudaEventSynchronize(ev[4]) == cudaSuccess) {
        float ms_input = 0.0f;
        float ms_gate = 0.0f;
        float ms_midq = 0.0f;
        float ms_downsum = 0.0f;
        float ms_total = 0.0f;
        (void)cudaEventElapsedTime(&ms_input, ev[0], ev[1]);
        (void)cudaEventElapsedTime(&ms_gate, ev[1], ev[2]);
        if (q8k_path) {
            (void)cudaEventElapsedTime(&ms_midq, ev[2], ev[3]);
            (void)cudaEventElapsedTime(&ms_downsum, ev[3], ev[4]);
        } else {
            (void)cudaEventElapsedTime(&ms_downsum, ev[2], ev[4]);
        }
        (void)cudaEventElapsedTime(&ms_total, ev[0], ev[4]);
        fprintf(stderr,
                "sf37: CUDA MoE profile topk=%u in=%u mid=%u out=%u input_q8k=%.3f gateup=%.3f mid_q8k=%.3f downsum=%.3f total=%.3f ms\n",
                topk, in_dim, expert_mid_dim, out_dim,
                ms_input, ms_gate, ms_midq, ms_downsum, ms_total);
    }
    for (uint32_t i = 0; i < 5u; i++) (void)cudaEventDestroy(ev[i]);
}

static int cuda_moe_batch_profile_start(cudaEvent_t ev[6]) {
    if (!cuda_moe_profile_enabled()) return 0;
    for (uint32_t i = 0; i < 6u; i++) {
        if (cudaEventCreate(&ev[i]) != cudaSuccess) {
            for (uint32_t j = 0; j < i; j++) (void)cudaEventDestroy(ev[j]);
            memset(ev, 0, 6u * sizeof(ev[0]));
            (void)cudaGetLastError();
            return 0;
        }
    }
    (void)cudaEventRecord(ev[0], 0);
    return 1;
}

static void cuda_moe_batch_profile_abort(cudaEvent_t ev[6]) {
    if (!ev[0]) return;
    for (uint32_t i = 0; i < 6u; i++) {
        if (ev[i]) (void)cudaEventDestroy(ev[i]);
        ev[i] = NULL;
    }
}

static void cuda_moe_batch_profile_finish(cudaEvent_t ev[6],
                                          uint32_t n_tok,
                                          uint32_t topk,
                                          uint32_t in_dim,
                                          uint32_t expert_mid_dim,
                                          uint32_t out_dim,
                                          int expert_tiles,
                                          int direct_down) {
    if (!ev[0]) return;
    if (cudaEventSynchronize(ev[5]) == cudaSuccess) {
        float ms_input = 0.0f;
        float ms_sort = 0.0f;
        float ms_gate = 0.0f;
        float ms_midq = 0.0f;
        float ms_downsum = 0.0f;
        float ms_total = 0.0f;
        (void)cudaEventElapsedTime(&ms_input, ev[0], ev[1]);
        (void)cudaEventElapsedTime(&ms_sort, ev[1], ev[2]);
        (void)cudaEventElapsedTime(&ms_gate, ev[2], ev[3]);
        (void)cudaEventElapsedTime(&ms_midq, ev[3], ev[4]);
        (void)cudaEventElapsedTime(&ms_downsum, ev[4], ev[5]);
        (void)cudaEventElapsedTime(&ms_total, ev[0], ev[5]);
        fprintf(stderr,
                "sf37: CUDA MoE batch profile tokens=%u pairs=%u in=%u mid=%u out=%u mode=%s down=%s input_q8k=%.3f sort=%.3f gateup=%.3f mid_q8k=%.3f downsum=%.3f total=%.3f ms\n",
                n_tok, n_tok * topk, in_dim, expert_mid_dim, out_dim,
                expert_tiles ? "expert_tile" : "sorted",
                direct_down ? "atomic_direct" : "scratch_sum",
                ms_input, ms_sort, ms_gate, ms_midq, ms_downsum, ms_total);
    }
    for (uint32_t i = 0; i < 6u; i++) (void)cudaEventDestroy(ev[i]);
}

static uint64_t cuda_align_u64(uint64_t v, uint64_t a) {
    return (v + a - 1u) & ~(a - 1u);
}

static double cuda_wall_sec(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
    }
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

static void cuda_model_load_progress_reset(void) {
    g_model_load_progress_next = 0;
    g_model_load_progress_last = 0.0;
    g_model_load_progress_started = 0;
    g_model_load_progress_tty = 0;
}

static void cuda_model_load_progress_note(uint64_t cached_bytes) {
    const double now = cuda_wall_sec();
    if (!g_model_load_progress_started) {
        g_model_load_progress_started = 1;
        g_model_load_progress_tty = isatty(STDERR_FILENO);
        g_model_load_progress_next = (g_model_load_progress_tty ? 2ull : 16ull) * 1073741824ull;
        g_model_load_progress_last = now;
        if (g_model_load_progress_tty) {
            fprintf(stderr, "\r\033[Ksf37: CUDA loading model tensors: 0.00 GiB");
        } else {
            fprintf(stderr, "sf37: CUDA loading model tensors\n");
        }
        fflush(stderr);
        return;
    }

    if (cached_bytes < g_model_load_progress_next &&
        now - g_model_load_progress_last < (g_model_load_progress_tty ? 2.0 : 10.0)) {
        return;
    }
    if (g_model_load_progress_tty) {
        fprintf(stderr, "\r\033[Ksf37: CUDA loading model tensors: %.2f GiB",
                (double)cached_bytes / 1073741824.0);
    } else {
        fprintf(stderr, "sf37: CUDA loading model tensors %.2f GiB cached\n",
                (double)cached_bytes / 1073741824.0);
    }
    fflush(stderr);
    g_model_load_progress_last = now;
    const uint64_t step = (g_model_load_progress_tty ? 2ull : 16ull) * 1073741824ull;
    while (g_model_load_progress_next <= cached_bytes) g_model_load_progress_next += step;
}

static uint64_t cuda_model_cache_limit_bytes(void) {
    uint64_t gb = 0;
    const char *env = cuda_env_value("SF37_CUDA_WEIGHT_CACHE_LIMIT_GB",
                                     "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB");
    if (env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env) gb = (uint64_t)v;
        return gb * 1073741824ull;
    }
    return 96ull * 1073741824ull;
}

static int cuda_model_cache_limit_explicit(void) {
    return cuda_env_value("SF37_CUDA_WEIGHT_CACHE_LIMIT_GB",
                          "DS4_CUDA_WEIGHT_CACHE_LIMIT_GB") != NULL;
}

static uint64_t cuda_model_local_model_limit_bytes(void) {
    const uint64_t default_limit = 96ull * 1073741824ull;
    if (!cuda_model_cache_limit_explicit()) return default_limit;
    const uint64_t explicit_limit = cuda_model_cache_limit_bytes();
    return explicit_limit > default_limit ? explicit_limit : default_limit;
}

static const char *cuda_model_ptr(const void *model_map, uint64_t offset) {
    if (model_map == g_model_host_base && g_model_device_base) return g_model_device_base + offset;
    return (const char *)model_map + offset;
}

static void cuda_model_range_release_all(void) {
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_registered && r.registered_base) {
            (void)cudaHostUnregister(r.registered_base);
        } else if (r.device_ptr && !r.arena_allocated) {
            (void)cudaFree(r.device_ptr);
        }
    }
    for (const cuda_model_arena &a : g_model_arenas) {
        if (a.device_ptr) (void)cudaFree(a.device_ptr);
    }
    g_model_arenas.clear();
    g_model_ranges.clear();
    g_model_range_by_offset.clear();
    g_model_range_bytes = 0;
    g_model_cache_full = 0;
    cuda_model_load_progress_reset();
}

static void cuda_q8_expanded_cache_release_all(void) {
    for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
        if (r.device_ptr) (void)cudaFree(r.device_ptr);
    }
    for (const cuda_q8_f32_range &r : g_q8_f32_ranges) {
        if (r.device_ptr) (void)cudaFree(r.device_ptr);
    }
    g_q8_f16_ranges.clear();
    g_q8_f16_by_offset.clear();
    g_q8_f32_ranges.clear();
    g_q8_f32_by_offset.clear();
    g_q8_f16_bytes = 0;
    g_q8_f32_bytes = 0;
}

static void cuda_model_cache_stats_reset(void) {
    g_model_cache_hits = 0;
    g_model_cache_misses = 0;
    g_model_cache_load_bytes = 0;
    g_model_cache_evict_count = 0;
    g_model_cache_evict_bytes = 0;
    memset(g_layer_stats, 0, sizeof(g_layer_stats));
    g_model_load_count_by_offset.clear();
    g_active_layer = -1;
}

static int cuda_model_eviction_allowed(void) {
    return !cuda_env_present("SF37_CUDA_NO_WEIGHT_EVICT", "DS4_CUDA_NO_WEIGHT_EVICT");
}

static void cuda_model_evict_cached_ranges(const char *why) {
    if (g_model_ranges.empty() && g_model_arenas.empty()) return;
    (void)cudaDeviceSynchronize();
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "sf37: CUDA evicting %.2f GiB model cache for %s "
                "(hits=%llu misses=%llu loaded=%.2f GiB evicts=%llu evicted=%.2f GiB)\n",
                (double)g_model_range_bytes / 1073741824.0,
                why ? why : "new range",
                (unsigned long long)g_model_cache_hits,
                (unsigned long long)g_model_cache_misses,
                (double)g_model_cache_load_bytes / 1073741824.0,
                (unsigned long long)g_model_cache_evict_count,
                (double)g_model_cache_evict_bytes / 1073741824.0);
    }
    g_model_cache_evict_count++;
    g_model_cache_evict_bytes += g_model_range_bytes;
    if (g_active_layer >= 0 && g_active_layer < SF37_CUDA_MAX_STAT_LAYERS) {
        g_layer_stats[g_active_layer].evict_count++;
    }
    cuda_model_range_release_all();
}

extern "C" void sf37_cuda_evict_model_cache(const char *reason) {
    cuda_model_evict_cached_ranges(reason ? reason : "explicit eviction");
}

static uint64_t cuda_round_down(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    return (v / align) * align;
}

static uint64_t cuda_round_up(uint64_t v, uint64_t align) {
    if (align <= 1) return v;
    const uint64_t rem = v % align;
    return rem == 0 ? v : v + (align - rem);
}

static void *cuda_align_ptr(void *ptr, uint64_t align) {
    if (align <= 1) return ptr;
    uintptr_t p = (uintptr_t)ptr;
    uintptr_t a = (uintptr_t)align;
    return (void *)(((p + a - 1u) / a) * a);
}

static uint64_t cuda_model_copy_chunk_bytes(void) {
    uint64_t mb = 64;
    const char *env = cuda_env_value("SF37_CUDA_MODEL_COPY_CHUNK_MB",
                                     "DS4_CUDA_MODEL_COPY_CHUNK_MB");
    if (env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 16) mb = 16;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static void cuda_model_discard_source_pages(const void *model_map, uint64_t model_size,
                                            uint64_t offset, uint64_t bytes) {
#if defined(POSIX_MADV_DONTNEED)
    if (cuda_env_present("SF37_CUDA_KEEP_MODEL_PAGES", "DS4_CUDA_KEEP_MODEL_PAGES") ||
        !model_map || bytes == 0 || offset > model_size) {
        return;
    }
    if (bytes > model_size - offset) bytes = model_size - offset;
    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t h1 = h0 + bytes;
    const uintptr_t p0 = h0 & ~(uintptr_t)(page_sz - 1u);
    const uintptr_t p1 = (h1 + page_sz - 1u) & ~(uintptr_t)(page_sz - 1u);
    if (p1 > p0) (void)posix_madvise((void *)p0, (size_t)(p1 - p0), POSIX_MADV_DONTNEED);
#else
    (void)model_map;
    (void)model_size;
    (void)offset;
    (void)bytes;
#endif
}

static void cuda_model_drop_file_pages(uint64_t offset, uint64_t bytes) {
#if defined(POSIX_FADV_DONTNEED)
    if (g_model_fd < 0 ||
        cuda_env_present("SF37_CUDA_KEEP_MODEL_PAGES", "DS4_CUDA_KEEP_MODEL_PAGES") ||
        bytes == 0) {
        return;
    }
    (void)posix_fadvise(g_model_fd, (off_t)offset, (off_t)bytes, POSIX_FADV_DONTNEED);
#else
    (void)offset;
    (void)bytes;
#endif
}

static int cuda_model_stage_pool_alloc(uint64_t bytes) {
    if (g_model_stage_bytes >= bytes) return 1;
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (!g_model_upload_stream) {
        cudaError_t err = cudaStreamCreateWithFlags(&g_model_upload_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model upload stream creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    for (size_t i = 0; i < 4; i++) {
        cudaError_t err = cudaMallocHost(&g_model_stage_raw[i], (size_t)bytes);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA pinned model staging allocation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
        g_model_stage[i] = cuda_align_ptr(g_model_stage_raw[i], g_model_direct_align);
        err = cudaEventCreateWithFlags(&g_model_stage_event[i], cudaEventDisableTiming);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model staging event creation failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    g_model_stage_bytes = bytes;
    return 1;
}

static int cuda_pread_full(int fd, void *buf, uint64_t bytes, uint64_t offset) {
    uint64_t done = 0;
    while (done < bytes) {
        const size_t n_req = (bytes - done > (uint64_t)SSIZE_MAX)
                           ? (size_t)SSIZE_MAX
                           : (size_t)(bytes - done);
        ssize_t n = pread(fd, (char *)buf + done, n_req, (off_t)(offset + done));
        if (n < 0) {
            if (errno == EINTR) continue;
            return 0;
        }
        if (n == 0) return 0;
        done += (uint64_t)n;
    }
    return 1;
}

static int cuda_model_stage_read(void *stage, uint64_t stage_bytes,
                                 uint64_t offset, uint64_t bytes,
                                 const char **payload) {
    *payload = (const char *)stage;
#if defined(__linux__) && defined(O_DIRECT)
    if (g_model_direct_fd >= 0 && g_model_direct_align > 1 && g_model_file_size != 0) {
        const uint64_t aligned_off = cuda_round_down(offset, g_model_direct_align);
        const uint64_t delta = offset - aligned_off;
        uint64_t read_size = cuda_round_up(delta + bytes, g_model_direct_align);
        if (aligned_off <= g_model_file_size &&
            read_size <= stage_bytes &&
            read_size <= g_model_file_size - aligned_off) {
            const int saved_errno = errno;
            errno = 0;
            if (cuda_pread_full(g_model_direct_fd, stage, read_size, aligned_off)) {
                *payload = (const char *)stage + delta;
                errno = saved_errno;
                return 1;
            }
            const int direct_errno = errno;
            if (direct_errno == EINVAL || direct_errno == EFAULT ||
                direct_errno == ENOTSUP || direct_errno == EOPNOTSUPP) {
                if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                                     "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "sf37: CUDA direct model read disabled: %s\n",
                            strerror(direct_errno));
                }
                (void)close(g_model_direct_fd);
                g_model_direct_fd = -1;
                g_model_direct_align = 1;
            }
            errno = direct_errno;
        }
    }
#else
    (void)stage_bytes;
#endif
    return cuda_pread_full(g_model_fd, stage, bytes, offset);
}

static uint64_t cuda_model_arena_chunk_bytes(uint64_t need) {
    uint64_t mb = 1792;
    const char *env = cuda_env_value("SF37_CUDA_WEIGHT_ARENA_CHUNK_MB",
                                     "DS4_CUDA_WEIGHT_ARENA_CHUNK_MB");
    if (env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 256) mb = 256;
    if (mb > 8192) mb = 8192;
    uint64_t bytes = mb * 1048576ull;
    if (need > bytes / 2u) {
        const uint64_t align = 64ull * 1048576ull;
        return (need + align - 1u) & ~(align - 1u);
    }
    if (bytes < need) {
        const uint64_t align = 256ull * 1048576ull;
        bytes = (need + align - 1u) & ~(align - 1u);
    }
    return bytes;
}

static char *cuda_model_arena_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_model_cache_full) {
        if (!g_model_preload_only && cuda_model_eviction_allowed() && g_model_range_bytes > 0) {
            cuda_model_evict_cached_ranges(what);
        } else {
            return NULL;
        }
    }
    const uint64_t align = 256u;
    const uint64_t aligned = (bytes + align - 1u) & ~(align - 1u);

    for (cuda_model_arena &a : g_model_arenas) {
        const uint64_t used = (a.used + align - 1u) & ~(align - 1u);
        if (used <= a.bytes && aligned <= a.bytes - used) {
            char *ptr = a.device_ptr + used;
            a.used = used + aligned;
            return ptr;
        }
    }

    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || aligned > limit - g_model_range_bytes) return NULL;

    const uint64_t chunk = cuda_model_arena_chunk_bytes(aligned);
    uint64_t actual_chunk = chunk;
    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)actual_chunk);
    if (err != cudaSuccess && actual_chunk > aligned) {
        (void)cudaGetLastError();
        actual_chunk = aligned;
        err = cudaMalloc(&dev, (size_t)actual_chunk);
    }
    if (err != cudaSuccess) {
        if (!g_model_preload_only && cuda_model_eviction_allowed() && g_model_range_bytes > 0) {
            (void)cudaGetLastError();
            cuda_model_evict_cached_ranges(what);
            actual_chunk = chunk;
            err = cudaMalloc(&dev, (size_t)actual_chunk);
            if (err != cudaSuccess && actual_chunk > aligned) {
                (void)cudaGetLastError();
                actual_chunk = aligned;
                err = cudaMalloc(&dev, (size_t)actual_chunk);
            }
            if (err == cudaSuccess) goto arena_alloc_ok;
        }
        fprintf(stderr, "sf37: CUDA model arena alloc failed for %s (%.2f MiB chunk): %s\n",
                what ? what : "weights",
                (double)actual_chunk / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        g_model_cache_full = 1;
        return NULL;
    }
arena_alloc_ok:
    g_model_arenas.push_back({(char *)dev, actual_chunk, aligned});
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        uint64_t arena_bytes = 0;
        for (const cuda_model_arena &a : g_model_arenas) arena_bytes += a.bytes;
        fprintf(stderr, "sf37: CUDA model arena allocated %.2f MiB (arenas %.2f GiB)\n",
                (double)actual_chunk / 1048576.0,
                (double)arena_bytes / 1073741824.0);
    }
    return (char *)dev;
}

static const char *cuda_model_direct_fallback_ptr(const void *model_map, uint64_t offset) {
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct ||
        cuda_env_present("SF37_CUDA_DIRECT_MODEL", "DS4_CUDA_DIRECT_MODEL")) {
        return cuda_model_ptr(model_map, offset);
    }
    return NULL;
}

static const char *cuda_model_range_register_mapped(const void *model_map,
                                                    uint64_t offset,
                                                    uint64_t bytes,
                                                    const char *what) {
    if (!g_model_range_mapping_supported || bytes == 0) return NULL;
    const double t0 = cuda_wall_sec();

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + offset);
    const uintptr_t reg_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t reg_delta = (uint64_t)(host_addr - reg_addr);
    uint64_t reg_bytes = (reg_delta + bytes + page_sz - 1u) & ~(page_sz - 1u);
    if (model_map == g_model_host_base &&
        g_model_registered_size >= 88ull * 1073741824ull &&
        g_model_registered_size <= 96ull * 1073741824ull &&
        g_model_range_bytes >= 80ull * 1073741824ull) {
        const uintptr_t model_base = (uintptr_t)model_map;
        const uintptr_t model_end = model_base + (uintptr_t)g_model_registered_size;
        if (model_end > model_base && model_end > reg_addr) {
            const uint64_t tail_bytes = (uint64_t)(model_end - reg_addr);
            reg_bytes = (tail_bytes + page_sz - 1u) & ~(page_sz - 1u);
        }
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (cuda_env_present("SF37_CUDA_HOST_REGISTER_PLAIN", "DS4_CUDA_HOST_REGISTER_PLAIN")) {
        flags = cudaHostRegisterMapped;
    }

    cudaError_t err = cudaHostRegister((void *)reg_addr, (size_t)reg_bytes, flags);
    if (err != cudaSuccess &&
        (flags & cudaHostRegisterReadOnly) != 0 &&
        (err == cudaErrorNotSupported || err == cudaErrorInvalidValue)) {
        (void)cudaGetLastError();
        err = cudaHostRegister((void *)reg_addr, (size_t)reg_bytes, cudaHostRegisterMapped);
    }
    if (err == cudaSuccess) {
        void *reg_dev = NULL;
        err = cudaHostGetDevicePointer(&reg_dev, (void *)reg_addr, 0);
        if (err == cudaSuccess && reg_dev) {
            char *dev_ptr = (char *)reg_dev + reg_delta;
            g_model_ranges.push_back({model_map, offset, bytes, dev_ptr,
                                      (void *)reg_addr, (char *)reg_dev,
                                      reg_bytes, 1, 0});
            g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
            g_model_cache_load_bytes += bytes;
            cuda_residency_note_load(offset, bytes, what, cuda_wall_sec() - t0);
            if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                                 "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "sf37: CUDA mapped %s %.2f MiB\n",
                        what ? what : "weights", (double)bytes / 1048576.0);
            }
            return dev_ptr;
        }
        fprintf(stderr, "sf37: CUDA model range map pointer failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaHostUnregister((void *)reg_addr);
        (void)cudaGetLastError();
        return NULL;
    }

    if (err == cudaErrorNotSupported || err == cudaErrorInvalidValue) {
        g_model_range_mapping_supported = 0;
    }
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "sf37: CUDA model range map skipped for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
    }
    (void)cudaGetLastError();
    return NULL;
}

static const char *cuda_model_range_populate_device_copy(const void *model_map,
                                                         uint64_t offset,
                                                         uint64_t bytes,
                                                         const char *what) {
    if (g_model_cache_full) return NULL;
    const double t0 = cuda_wall_sec();
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        g_model_cache_full = 1;
        if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                             "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "sf37: CUDA skipped device copy for %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)limit / 1073741824.0);
        }
        return NULL;
    }

    void *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model range alloc failed for %s (%.2f MiB): %s\n",
                what ? what : "weights", (double)bytes / 1048576.0,
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    const char *src = (const char *)model_map + offset;
    const uint64_t chunk = 64ull * 1024ull * 1024ull;
    for (uint64_t done = 0; done < bytes; done += chunk) {
        uint64_t n = bytes - done < chunk ? bytes - done : chunk;
        err = cudaMemcpy((char *)dev + done, src + done, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model range copy failed for %s at %.2f/%.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)done / 1048576.0,
                    (double)bytes / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return NULL;
        }
    }
    g_model_ranges.push_back({model_map, offset, bytes, (char *)dev, NULL, NULL, 0, 0, 0});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    g_model_cache_load_bytes += bytes;
    cuda_residency_note_load(offset, bytes, what, cuda_wall_sec() - t0);
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "sf37: CUDA cached %s %.2f MiB (total %.2f GiB, hits=%llu misses=%llu loaded=%.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0,
                (unsigned long long)g_model_cache_hits,
                (unsigned long long)g_model_cache_misses,
                (double)g_model_cache_load_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static const char *cuda_model_range_ptr_from_fd(const void *model_map,
                                                uint64_t offset,
                                                uint64_t bytes,
                                                const char *what) {
    if (g_model_fd < 0 || bytes == 0) return NULL;
    if (g_model_fd_host_base != NULL && model_map != g_model_fd_host_base) return NULL;
    const double t0 = cuda_wall_sec();
    const uint64_t limit = cuda_model_cache_limit_bytes();
    if (g_model_range_bytes > limit || bytes > limit - g_model_range_bytes) {
        g_model_cache_full = 1;
        if (!g_model_preload_only && cuda_model_eviction_allowed() && g_model_range_bytes > 0) {
            cuda_model_evict_cached_ranges(what);
        } else {
            if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                                 "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "sf37: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                        what ? what : "weights",
                        (double)bytes / 1048576.0,
                        (double)limit / 1073741824.0);
            }
            return cuda_model_direct_fallback_ptr(model_map, offset);
        }
    }
    if (bytes > cuda_model_cache_limit_bytes()) {
        if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                             "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
            fprintf(stderr, "sf37: CUDA direct %s %.2f MiB (cache budget %.2f GiB exhausted)\n",
                    what ? what : "weights",
                    (double)bytes / 1048576.0,
                    (double)cuda_model_cache_limit_bytes() / 1073741824.0);
        }
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    char *dev = cuda_model_arena_alloc(bytes, what);
    if (!dev) {
        if (cuda_env_present("SF37_CUDA_STRICT_WEIGHT_CACHE", "DS4_CUDA_STRICT_WEIGHT_CACHE")) return NULL;
        return cuda_model_direct_fallback_ptr(model_map, offset);
    }

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    const uint64_t stage_bytes = chunk + (g_model_direct_align > 1 ? g_model_direct_align : 1);
    if (!cuda_model_stage_pool_alloc(stage_bytes)) return NULL;

    uint64_t copied = 0;
    uint64_t chunk_idx = 0;
    while (copied < bytes) {
        cudaError_t err = cudaSuccess;
        const uint64_t n = (bytes - copied < chunk) ? (bytes - copied) : chunk;
        const uint64_t bi = chunk_idx % 4u;
        if (chunk_idx >= 4u) {
            err = cudaEventSynchronize(g_model_stage_event[bi]);
            if (err != cudaSuccess) {
                fprintf(stderr, "sf37: CUDA model staging wait failed for %s: %s\n",
                        what ? what : "weights", cudaGetErrorString(err));
                (void)cudaGetLastError();
                return NULL;
            }
        }
        const char *payload = NULL;
        if (!cuda_model_stage_read(g_model_stage[bi], g_model_stage_bytes,
                                   offset + copied, n, &payload)) {
            fprintf(stderr, "sf37: CUDA model range read failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    strerror(errno));
            return NULL;
        }
        err = cudaMemcpyAsync(dev + copied, payload, (size_t)n,
                              cudaMemcpyHostToDevice, g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model range copy failed for %s at %.2f MiB: %s\n",
                    what ? what : "weights",
                    (double)copied / 1048576.0,
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        err = cudaEventRecord(g_model_stage_event[bi], g_model_upload_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model staging record failed for %s: %s\n",
                    what ? what : "weights", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return NULL;
        }
        cuda_model_drop_file_pages(offset + copied, n);
        cuda_model_discard_source_pages(model_map, g_model_registered_size, offset + copied, n);
        copied += n;
        cuda_model_load_progress_note(g_model_range_bytes + copied);
        chunk_idx++;
    }

    cudaError_t err = cudaStreamSynchronize(g_model_upload_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model range upload sync failed for %s: %s\n",
                what ? what : "weights", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }

    g_model_ranges.push_back({model_map, offset, bytes, dev, NULL, NULL, 0, 0, 1});
    g_model_range_by_offset[offset] = g_model_ranges.size() - 1u;
    g_model_range_bytes += bytes;
    g_model_cache_load_bytes += bytes;
    cuda_model_load_progress_note(g_model_range_bytes);
    cuda_residency_note_load(offset, bytes, what, cuda_wall_sec() - t0);
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "sf37: CUDA fd-cached %s %.2f MiB (total %.2f GiB, hits=%llu misses=%llu loaded=%.2f GiB)\n",
                what ? what : "weights",
                (double)bytes / 1048576.0,
                (double)g_model_range_bytes / 1073741824.0,
                (unsigned long long)g_model_cache_hits,
                (unsigned long long)g_model_cache_misses,
                (double)g_model_cache_load_bytes / 1073741824.0);
    }
    return (const char *)dev;
}

static const char *cuda_model_range_ptr(const void *model_map,
                                        uint64_t offset,
                                        uint64_t bytes,
                                        const char *what) {
    if (bytes == 0) return cuda_model_ptr(model_map, offset);
    const uint64_t end = offset + bytes;
    if (end < offset) return NULL;

    auto exact = g_model_range_by_offset.find(offset);
    if (exact != g_model_range_by_offset.end()) {
        const cuda_model_range &r = g_model_ranges[exact->second];
        if (r.host_base == model_map && bytes <= r.bytes) {
            g_model_cache_hits++;
            return r.device_ptr;
        }
    }
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            g_model_cache_hits++;
            return r.device_ptr + (offset - r.offset);
        }
        if (r.host_base == model_map && r.host_registered &&
            r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) {
                g_model_cache_hits++;
                return r.registered_device_base + (h0 - r0);
            }
        }
    }

    g_model_cache_misses++;
    if (g_model_device_owned || g_model_registered) return cuda_model_ptr(model_map, offset);
    if (g_model_hmm_direct &&
        !cuda_env_present("SF37_CUDA_WEIGHT_CACHE", "DS4_CUDA_WEIGHT_CACHE") &&
        !cuda_env_present("SF37_CUDA_WEIGHT_PRELOAD", "DS4_CUDA_WEIGHT_PRELOAD")) {
        return cuda_model_ptr(model_map, offset);
    }
    if (cuda_env_present("SF37_CUDA_DIRECT_MODEL", "DS4_CUDA_DIRECT_MODEL")) {
        return cuda_model_ptr(model_map, offset);
    }

    if (!cuda_env_present("SF37_CUDA_NO_FD_CACHE", "DS4_CUDA_NO_FD_CACHE")) {
        const char *fd_ptr = cuda_model_range_ptr_from_fd(model_map, offset, bytes, what);
        if (fd_ptr) return fd_ptr;
        if (g_model_preload_only && g_model_cache_full) return NULL;
    }

    const char *mapped = cuda_model_range_register_mapped(model_map, offset, bytes, what);
    if (mapped) return mapped;

    return cuda_model_range_populate_device_copy(model_map, offset, bytes, what);
}

static int cuda_model_range_is_cached(const void *model_map, uint64_t offset, uint64_t bytes) {
    if (bytes == 0) return 1;
    if (g_model_device_owned || g_model_registered || g_model_hmm_direct) return 1;
    const uint64_t end = offset + bytes;
    if (end < offset) return 0;
    for (const cuda_model_range &r : g_model_ranges) {
        if (r.host_base == model_map &&
            offset >= r.offset &&
            end <= r.offset + r.bytes) {
            return 1;
        }
        if (r.host_base == model_map && r.host_registered &&
            r.registered_base && r.registered_device_base) {
            const uintptr_t h0 = (uintptr_t)((const char *)model_map + offset);
            const uintptr_t h1 = h0 + bytes;
            const uintptr_t r0 = (uintptr_t)r.registered_base;
            const uintptr_t r1 = r0 + r.registered_bytes;
            if (h1 >= h0 && h0 >= r0 && h1 <= r1) return 1;
        }
    }
    return 0;
}

static int cuda_model_prefetch_range(const void *model_map,
                                     uint64_t model_size,
                                     uint64_t map_offset,
                                     uint64_t map_size) {
    if (!model_map || map_size == 0 ||
        map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (cuda_env_present("SF37_CUDA_NO_MODEL_PREFETCH", "DS4_CUDA_NO_MODEL_PREFETCH") ||
        cuda_env_present("SF37_CUDA_COPY_MODEL", "DS4_CUDA_COPY_MODEL") ||
        cuda_env_present("SF37_CUDA_WEIGHT_CACHE", "DS4_CUDA_WEIGHT_CACHE") ||
        cuda_env_present("SF37_CUDA_WEIGHT_PRELOAD", "DS4_CUDA_WEIGHT_PRELOAD")) {
        return 0;
    }

    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    int pageable = 0;
    cudaError_t err = cudaDeviceGetAttribute(&pageable, cudaDevAttrPageableMemoryAccess, device);
    if (err != cudaSuccess || !pageable) {
        (void)cudaGetLastError();
        return 0;
    }
    cudaMemLocation loc;
    memset(&loc, 0, sizeof(loc));
    loc.type = cudaMemLocationTypeDevice;
    loc.id = device;

    const long page_sz_l = sysconf(_SC_PAGESIZE);
    const uint64_t page_sz = page_sz_l > 0 ? (uint64_t)page_sz_l : 4096u;
    const uintptr_t host_addr = (uintptr_t)((const char *)model_map + map_offset);
    const uintptr_t pre_addr = host_addr & ~(uintptr_t)(page_sz - 1u);
    const uint64_t pre_delta = (uint64_t)(host_addr - pre_addr);
    const uint64_t pre_bytes = (pre_delta + map_size + page_sz - 1u) & ~(page_sz - 1u);
    void *pre_ptr = (void *)pre_addr;

    const double t0 = cuda_wall_sec();
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetReadMostly, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model read-mostly advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    err = cudaMemAdvise(pre_ptr, (size_t)pre_bytes, cudaMemAdviseSetPreferredLocation, loc);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model preferred-location advise skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    if (!g_model_prefetch_stream) {
        err = cudaStreamCreateWithFlags(&g_model_prefetch_stream, cudaStreamNonBlocking);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model prefetch stream creation skipped: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }

    err = cudaMemPrefetchAsync(pre_ptr, (size_t)pre_bytes, loc, 0, g_model_prefetch_stream);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model prefetch skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    if (cuda_env_present("SF37_CUDA_MODEL_PREFETCH_SYNC", "DS4_CUDA_MODEL_PREFETCH_SYNC")) {
        err = cudaStreamSynchronize(g_model_prefetch_stream);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model prefetch sync failed: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
            return 0;
        }
    }
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "sf37: CUDA ATS/HMM prefetch queued %.2f GiB of model tensors in %.3fs\n",
            (double)map_size / 1073741824.0, t1 - t0);
    g_model_hmm_direct = 1;
    return 1;
}

static int cuda_model_copy_chunked(const void *model_map,
                                   uint64_t model_size,
                                   uint64_t map_offset,
                                   uint64_t map_size) {
    if (!model_map || model_size == 0 ||
        map_offset > model_size || map_size > model_size - map_offset) return 0;
    if (cuda_env_present("SF37_CUDA_NO_MODEL_COPY", "DS4_CUDA_NO_MODEL_COPY") ||
        cuda_env_present("SF37_CUDA_DIRECT_MODEL", "DS4_CUDA_DIRECT_MODEL") ||
        cuda_env_present("SF37_CUDA_WEIGHT_CACHE", "DS4_CUDA_WEIGHT_CACHE") ||
        cuda_env_present("SF37_CUDA_WEIGHT_PRELOAD", "DS4_CUDA_WEIGHT_PRELOAD")) {
        return 0;
    }
    if (g_model_device_owned || g_model_registered) return 1;

    void *dev = NULL;
    const double t0 = cuda_wall_sec();
    cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }

    fprintf(stderr, "sf37: CUDA chunk-copying %.2f GiB model image\n",
            (double)model_size / 1073741824.0);

    const uint64_t chunk = cuda_model_copy_chunk_bytes();
    void *stage = NULL;
    err = cudaMallocHost(&stage, (size_t)chunk);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA pinned model staging allocation failed: %s\n",
                cudaGetErrorString(err));
        (void)cudaFree(dev);
        (void)cudaGetLastError();
        return 0;
    }

    if (map_offset > 0) {
        uint64_t copied_header = 0;
        while (copied_header < map_offset) {
            const uint64_t n = (map_offset - copied_header < chunk) ? (map_offset - copied_header) : chunk;
            memcpy(stage, (const char *)model_map + copied_header, (size_t)n);
            err = cudaMemcpy((char *)dev + copied_header, stage, (size_t)n, cudaMemcpyHostToDevice);
            if (err != cudaSuccess) {
                fprintf(stderr, "sf37: CUDA model header copy failed: %s\n", cudaGetErrorString(err));
                (void)cudaFreeHost(stage);
                (void)cudaFree(dev);
                (void)cudaGetLastError();
                return 0;
            }
            copied_header += n;
        }
    }

    uint64_t copied = 0;
    double last_report = t0;
    while (copied < map_size) {
        const uint64_t n = (map_size - copied < chunk) ? (map_size - copied) : chunk;
        const uint64_t off = map_offset + copied;
        memcpy(stage, (const char *)model_map + off, (size_t)n);
        err = cudaMemcpy((char *)dev + off, stage, (size_t)n, cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "sf37: CUDA model chunk copy failed at %.2f GiB: %s\n",
                    (double)copied / 1073741824.0, cudaGetErrorString(err));
            (void)cudaFreeHost(stage);
            (void)cudaFree(dev);
            (void)cudaGetLastError();
            return 0;
        }
        cuda_model_discard_source_pages(model_map, model_size, off, n);
        copied += n;
        const double now = cuda_wall_sec();
        if (cuda_env_present("SF37_CUDA_MODEL_COPY_VERBOSE", "DS4_CUDA_MODEL_COPY_VERBOSE") &&
            now - last_report >= 2.0) {
            fprintf(stderr, "sf37: CUDA model chunk copy %.2f/%.2f GiB\n",
                    (double)copied / 1073741824.0,
                    (double)map_size / 1073741824.0);
            last_report = now;
        }
    }

    (void)cudaFreeHost(stage);
    g_model_device_base = (const char *)dev;
    g_model_device_owned = 1;
    g_model_hmm_direct = 0;
    const double t1 = cuda_wall_sec();
    fprintf(stderr,
            "sf37: CUDA model chunk copy complete in %.3fs (%.2f GiB tensors)\n",
            t1 - t0, (double)map_size / 1073741824.0);
    return 1;
}

static int cuda_model_set_host_map(const void *model_map, uint64_t model_size) {
    if (!model_map || model_size == 0) return 0;
    cuda_q8_expanded_cache_release_all();
    cuda_model_range_release_all();
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
        g_model_device_owned = 0;
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
        g_model_registered = 0;
    }
    g_model_host_base = model_map;
    g_model_device_base = (const char *)model_map;
    g_model_registered_size = model_size;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_cache_full = 0;
    g_model_preload_only = 0;
    g_model_mapping_failure_notice_printed = 0;
    cuda_model_cache_stats_reset();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    if (g_model_fd >= 0 && g_model_fd_host_base == NULL) {
        g_model_fd_host_base = model_map;
    }
    return 1;
}

static int cuda_model_set_model_map(const void *model_map, uint64_t model_size) {
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    if (cuda_env_present("SF37_CUDA_COPY_MODEL", "DS4_CUDA_COPY_MODEL")) {
        void *dev = NULL;
        const double t0 = cuda_wall_sec();
        cudaError_t err = cudaMalloc(&dev, (size_t)model_size);
        if (err == cudaSuccess) {
            fprintf(stderr, "sf37: CUDA copying %.2f GiB model to device memory\n",
                    (double)model_size / 1073741824.0);
            err = cudaMemcpy(dev, model_map, (size_t)model_size, cudaMemcpyHostToDevice);
            if (err == cudaSuccess) {
                g_model_device_base = (const char *)dev;
                g_model_device_owned = 1;
                const double t1 = cuda_wall_sec();
                fprintf(stderr, "sf37: CUDA model copy complete in %.3fs\n", t1 - t0);
                return 1;
            }
            fprintf(stderr, "sf37: CUDA model copy failed: %s\n", cudaGetErrorString(err));
            (void)cudaFree(dev);
            (void)cudaGetLastError();
        } else {
            fprintf(stderr, "sf37: CUDA model allocation skipped: %s\n", cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    }

    unsigned int flags = cudaHostRegisterMapped | cudaHostRegisterReadOnly;
    if (cuda_env_present("SF37_CUDA_HOST_REGISTER_PLAIN", "DS4_CUDA_HOST_REGISTER_PLAIN")) {
        flags = cudaHostRegisterMapped;
    }
    cudaError_t err = cudaHostRegister((void *)model_map, (size_t)model_size, flags);
    if (err == cudaSuccess) {
        void *dev = NULL;
        err = cudaHostGetDevicePointer(&dev, (void *)model_map, 0);
        if (err == cudaSuccess && dev) {
            g_model_device_base = (const char *)dev;
            g_model_registered = 1;
            fprintf(stderr, "sf37: CUDA registered %.2f GiB model mapping for device access\n",
                    (double)model_size / 1073741824.0);
        } else {
            fprintf(stderr, "sf37: CUDA host registration pointer lookup failed: %s\n",
                    cudaGetErrorString(err));
            (void)cudaGetLastError();
        }
    } else {
        fprintf(stderr, "sf37: CUDA host registration skipped: %s\n", cudaGetErrorString(err));
        (void)cudaGetLastError();
        const uint64_t limit = cuda_model_local_model_limit_bytes();
        if (!cuda_model_cache_limit_explicit() && model_size > limit) {
            fprintf(stderr,
                    "sf37: CUDA model %.2f GiB exceeds the default single-GPU "
                    "startup cache budget %.2f GiB; set SF37_CUDA_WEIGHT_CACHE_LIMIT_GB "
                    "or use a narrower preload span\n",
                    (double)model_size / 1073741824.0,
                    (double)limit / 1073741824.0);
            return 0;
        }
    }
    return 1;
}

extern "C" int sf37_cuda_set_model_fd(int fd) {
    g_model_fd = fd;
    g_model_fd_host_base = g_model_host_base;
    g_model_file_size = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    if (fd >= 0) {
        struct stat st;
        if (fstat(fd, &st) == 0 && st.st_size > 0) {
            g_model_file_size = (uint64_t)st.st_size;
            if (st.st_blksize > 1) g_model_direct_align = (uint64_t)st.st_blksize;
        }
#if defined(__linux__) && defined(O_DIRECT)
        if (!cuda_env_present("SF37_CUDA_NO_DIRECT_IO", "DS4_CUDA_NO_DIRECT_IO")) {
            char proc_path[64];
            snprintf(proc_path, sizeof(proc_path), "/proc/self/fd/%d", fd);
            int direct_fd = open(proc_path, O_RDONLY | O_DIRECT);
            if (direct_fd >= 0) {
                g_model_direct_fd = direct_fd;
                if (g_model_direct_align < 512) g_model_direct_align = 512;
                if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                                     "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                    fprintf(stderr, "sf37: CUDA model direct I/O enabled (align=%llu)\n",
                            (unsigned long long)g_model_direct_align);
                }
            } else if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE",
                                        "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
                fprintf(stderr, "sf37: CUDA model direct I/O unavailable: %s\n", strerror(errno));
            }
        }
#endif
    }
    return 1;
}

extern "C" int sf37_cuda_set_model_map_range(const void *model_map,
                                              uint64_t model_size,
                                              uint64_t map_offset,
                                              uint64_t map_size,
                                              uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!cuda_model_set_model_map(model_map, model_size)) return 0;
    if (cuda_env_present("SF37_CUDA_COPY_MODEL_CHUNKED", "DS4_CUDA_COPY_MODEL_CHUNKED") &&
        !cuda_model_copy_chunked(model_map, model_size, map_offset, map_size)) {
        (void)cuda_model_prefetch_range(model_map, model_size, map_offset, map_size);
    }
    return 1;
}

extern "C" int sf37_cuda_set_model_map_spans(const void *model_map,
                                              uint64_t model_size,
                                              const uint64_t *offsets,
                                              const uint64_t *sizes,
                                              uint32_t count,
                                              uint64_t max_tensor_bytes) {
    (void)max_tensor_bytes;
    if (!model_map || model_size == 0 || !offsets || !sizes || count == 0) return 0;
    for (uint32_t i = 0; i < count; i++) {
        if (offsets[i] > model_size || sizes[i] == 0 ||
            sizes[i] > model_size - offsets[i]) {
            return 0;
        }
    }
    if (!cuda_model_set_host_map(model_map, model_size)) return 0;

    if (cuda_env_present("SF37_CUDA_COPY_MODEL_CHUNKED", "DS4_CUDA_COPY_MODEL_CHUNKED")) {
        for (uint32_t i = 0; i < count; i++) {
            (void)cuda_model_prefetch_range(model_map, model_size, offsets[i], sizes[i]);
        }
    }
    return 1;
}

extern "C" int sf37_cuda_cache_model_range(const void *model_map,
                                            uint64_t model_size,
                                            uint64_t offset,
                                            uint64_t bytes,
                                            const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (offset > model_size || bytes > model_size - offset) return 0;
    if (cuda_model_range_is_cached(model_map, offset, bytes)) return 1;

    g_model_preload_only = 1;
    const char *ptr = cuda_model_range_ptr(model_map, offset, bytes,
                                           label ? label : "model_tensor");
    g_model_preload_only = 0;
    if (!ptr || !cuda_model_range_is_cached(model_map, offset, bytes)) {
        if (!g_model_mapping_failure_notice_printed) {
            fprintf(stderr, "sf37: CUDA failed to prepare model tensor spans for device access\n");
            g_model_mapping_failure_notice_printed = 1;
        }
        return 0;
    }
    return 1;
}

static void *cuda_tmp_alloc(uint64_t bytes, const char *what) {
    if (bytes == 0) return NULL;
    if (g_tmp_bytes >= bytes) return g_tmp;
    if (g_tmp) {
        (void)cudaFree(g_tmp);
        g_tmp = NULL;
        g_tmp_bytes = 0;
    }
    void *ptr = NULL;
    if (!cuda_ok(cudaMalloc(&ptr, (size_t)bytes), what ? what : "tmp alloc")) return NULL;
    g_tmp = ptr;
    g_tmp_bytes = bytes;
    return g_tmp;
}

extern "C" int sf37_cuda_device_count(void) {
    int n = 0;
    cudaError_t err = cudaGetDeviceCount(&n);
    if (err != cudaSuccess) return -1;
    return n;
}

extern "C" void sf37_cuda_print_devices(void) {
    int n = sf37_cuda_device_count();
    if (n < 0) {
        fprintf(stderr, "sf37: CUDA device query failed\n");
        return;
    }
    for (int i = 0; i < n; i++) {
        cudaDeviceProp prop;
        if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
            fprintf(stderr, "sf37: CUDA device %d: %s sm_%d%d memory %.2f GiB\n",
                    i, prop.name, prop.major, prop.minor,
                    (double)prop.totalGlobalMem / 1073741824.0);
        }
    }
}

extern "C" int sf37_cuda_init(void) {
    if (!cuda_ok(cudaSetDevice(0), "set device")) return 0;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) {
        fprintf(stderr, "sf37: CUDA backend initialized on %s (sm_%d%d)\n",
                prop.name, prop.major, prop.minor);
    }
    if (!g_cublas_ready) {
        if (!cublas_ok(cublasCreate(&g_cublas), "create handle")) return 0;
        const cublasMath_t math_mode =
            cuda_env_present("SF37_CUDA_NO_TF32", "DS4_CUDA_NO_TF32")
                ? CUBLAS_DEFAULT_MATH
                : CUBLAS_TF32_TENSOR_OP_MATH;
        (void)cublasSetMathMode(g_cublas, math_mode);
        g_cublas_ready = 1;
    }
    return 1;
}

extern "C" void sf37_cuda_cleanup(void) {
    (void)cudaDeviceSynchronize();
    if (g_cublas_ready) {
        (void)cublasDestroy(g_cublas);
        g_cublas = NULL;
        g_cublas_ready = 0;
    }
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr,
                "sf37: CUDA model cache stats: resident=%.2f GiB hits=%llu misses=%llu "
                "loaded=%.2f GiB evicts=%llu evicted=%.2f GiB q8_f16=%.2f GiB q8_f32=%.2f GiB\n",
                (double)g_model_range_bytes / 1073741824.0,
                (unsigned long long)g_model_cache_hits,
                (unsigned long long)g_model_cache_misses,
                (double)g_model_cache_load_bytes / 1073741824.0,
                (unsigned long long)g_model_cache_evict_count,
                (double)g_model_cache_evict_bytes / 1073741824.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                (double)g_q8_f32_bytes / 1073741824.0);
        cuda_residency_print_layer_stats();
    }
    cuda_q8_expanded_cache_release_all();
    g_q8_f16_disabled_after_oom = 0;
    g_q8_f16_budget_notice_printed = 0;
    cuda_model_range_release_all();
    if (g_tmp) {
        (void)cudaFree(g_tmp);
        g_tmp = NULL;
        g_tmp_bytes = 0;
    }
    for (size_t i = 0; i < 4; i++) {
        if (g_model_stage_event[i]) {
            (void)cudaEventDestroy(g_model_stage_event[i]);
            g_model_stage_event[i] = NULL;
        }
        if (g_model_stage_raw[i]) {
            (void)cudaFreeHost(g_model_stage_raw[i]);
            g_model_stage_raw[i] = NULL;
            g_model_stage[i] = NULL;
        }
    }
    g_model_stage_bytes = 0;
    if (g_model_upload_stream) {
        (void)cudaStreamDestroy(g_model_upload_stream);
        g_model_upload_stream = NULL;
    }
    if (g_model_device_owned && g_model_device_base) {
        (void)cudaFree((void *)g_model_device_base);
    }
    if (g_model_registered && g_model_host_base) {
        (void)cudaHostUnregister((void *)g_model_host_base);
    }
    g_model_host_base = NULL;
    g_model_device_base = NULL;
    g_model_registered_size = 0;
    g_model_registered = 0;
    g_model_device_owned = 0;
    g_model_range_mapping_supported = 1;
    g_model_hmm_direct = 0;
    g_model_fd = -1;
    g_model_fd_host_base = NULL;
    g_model_preload_only = 0;
    if (g_model_direct_fd >= 0) {
        (void)close(g_model_direct_fd);
        g_model_direct_fd = -1;
    }
    g_model_direct_align = 1;
    g_model_file_size = 0;
    g_model_cache_full = 0;
    g_model_mapping_failure_notice_printed = 0;
    if (g_model_prefetch_stream) {
        (void)cudaStreamDestroy(g_model_prefetch_stream);
        g_model_prefetch_stream = NULL;
    }
}

extern "C" int sf37_cuda_synchronize(void) {
    return cuda_ok(cudaDeviceSynchronize(), "synchronize");
}

extern "C" int sf37_cuda_memory_info(uint64_t *free_bytes, uint64_t *total_bytes) {
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }
    if (free_bytes) *free_bytes = (uint64_t)free_b;
    if (total_bytes) *total_bytes = (uint64_t)total_b;
    return 1;
}

extern "C" int sf37_cuda_begin_layer(uint32_t layer) {
    if (layer >= SF37_CUDA_MAX_STAT_LAYERS) {
        g_active_layer = -1;
        return 0;
    }
    g_active_layer = (int)layer;
    return 1;
}

extern "C" void sf37_cuda_end_layer(void) {
    g_active_layer = -1;
}

extern "C" sf37_cuda_tensor *sf37_cuda_tensor_alloc(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    sf37_cuda_tensor *t = (sf37_cuda_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMalloc(&t->ptr, (size_t)bytes), "tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    return t;
}

extern "C" sf37_cuda_tensor *sf37_cuda_tensor_alloc_managed(uint64_t bytes) {
    if (bytes == 0) bytes = 1;
    sf37_cuda_tensor *t = (sf37_cuda_tensor *)calloc(1, sizeof(*t));
    if (!t) return NULL;
    if (!cuda_ok(cudaMallocManaged(&t->ptr, (size_t)bytes), "managed tensor alloc")) {
        free(t);
        return NULL;
    }
    t->bytes = bytes;
    return t;
}

static uint64_t cuda_managed_kv_reserve_bytes(uint64_t total_bytes) {
    const uint64_t min_reserve = 8ull * 1073741824ull;
    const uint64_t max_reserve = 40ull * 1073741824ull;
    uint64_t reserve = total_bytes / 4u;
    if (reserve < min_reserve) reserve = min_reserve;
    if (reserve > max_reserve) reserve = max_reserve;
    return reserve;
}

extern "C" int sf37_cuda_should_use_managed_kv_cache(uint64_t kv_cache_bytes,
                                                      uint64_t context_bytes) {
    if (kv_cache_bytes == 0) return 0;

    const uint64_t huge_kv = 8ull * 1073741824ull;
    if (kv_cache_bytes >= huge_kv) return 1;

    const uint64_t large_context = 8ull * 1073741824ull;
    if (context_bytes < large_context) return 0;

    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        return 0;
    }

    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    const uint64_t reserve_bytes = cuda_managed_kv_reserve_bytes(total_bytes);
    if (context_bytes > free_bytes) return 1;
    return free_bytes - context_bytes < reserve_bytes;
}

extern "C" void sf37_cuda_tensor_free(sf37_cuda_tensor *tensor) {
    if (!tensor) return;
    if (tensor->ptr) (void)cudaFree(tensor->ptr);
    free(tensor);
}

extern "C" uint64_t sf37_cuda_tensor_bytes(const sf37_cuda_tensor *tensor) {
    return tensor ? tensor->bytes : 0;
}

extern "C" int sf37_cuda_tensor_write(sf37_cuda_tensor *tensor, uint64_t offset,
                                       const void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy((char *)tensor->ptr + offset, data, (size_t)bytes,
                              cudaMemcpyHostToDevice), "tensor write");
}

extern "C" int sf37_cuda_tensor_read(const sf37_cuda_tensor *tensor, uint64_t offset,
                                      void *data, uint64_t bytes) {
    if (!tensor || !data || offset > tensor->bytes || bytes > tensor->bytes - offset) return 0;
    return cuda_ok(cudaMemcpy(data, (const char *)tensor->ptr + offset, (size_t)bytes,
                              cudaMemcpyDeviceToHost), "tensor read");
}

extern "C" int sf37_cuda_tensor_copy(sf37_cuda_tensor *dst, uint64_t dst_offset,
                                      const sf37_cuda_tensor *src, uint64_t src_offset,
                                      uint64_t bytes) {
    if (!dst || !src ||
        dst_offset > dst->bytes || bytes > dst->bytes - dst_offset ||
        src_offset > src->bytes || bytes > src->bytes - src_offset) return 0;
    return cuda_ok(cudaMemcpy((char *)dst->ptr + dst_offset,
                              (const char *)src->ptr + src_offset,
                              (size_t)bytes,
                              cudaMemcpyDeviceToDevice), "tensor copy");
}

__global__ static void fill_f32_kernel(float *x, uint64_t n, float v) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) x[gid] = v;
}

extern "C" int sf37_cuda_tensor_fill_f32(sf37_cuda_tensor *tensor, float value, uint64_t count) {
    if (!tensor || count > tensor->bytes / sizeof(float)) return 0;
    if (count == 0) return 1;
    fill_f32_kernel<<<(unsigned)((count + 255u) / 256u), 256>>>((float *)tensor->ptr, count, value);
    return cuda_ok(cudaGetLastError(), "fill f32 launch");
}

static int mapped_range_ok(uint64_t model_size, uint64_t offset, uint64_t bytes);
__device__ static float sf37_bf16_to_f32_dev(uint16_t x);

__global__ static void embed_tokens_bf16_kernel(float *out,
                                                const uint16_t *emb,
                                                const int32_t *tokens,
                                                uint64_t dim,
                                                uint64_t vocab,
                                                uint32_t n_tok) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tok * dim;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / dim);
    const uint64_t d = gid - (uint64_t)t * dim;
    int32_t tok = tokens[t];
    if (tok < 0 || (uint64_t)tok >= vocab) {
        out[gid] = 0.0f;
        return;
    }
    out[gid] = sf37_bf16_to_f32_dev(emb[(uint64_t)(uint32_t)tok * dim + d]);
}

extern "C" int sf37_cuda_embed_tokens_bf16_mapped(sf37_cuda_tensor *out,
                                                   const void *model_map,
                                                   uint64_t model_size,
                                                   uint64_t weight_offset,
                                                   uint64_t dim,
                                                   uint64_t vocab,
                                                   const sf37_cuda_tensor *tokens,
                                                   uint32_t n_tok) {
    if (!out || !model_map || !tokens || dim == 0 || vocab == 0 || n_tok == 0) return 0;
    if (vocab > UINT64_MAX / dim / sizeof(uint16_t) ||
        n_tok > UINT64_MAX / dim / sizeof(float)) return 0;
    const uint64_t weight_bytes = vocab * dim * sizeof(uint16_t);
    const uint64_t out_bytes = (uint64_t)n_tok * dim * sizeof(float);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_bytes ||
        tokens->bytes < (uint64_t)n_tok * sizeof(int32_t)) {
        return 0;
    }
    const uint16_t *emb = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                                 weight_offset,
                                                                 weight_bytes,
                                                                 "embed_tokens_bf16");
    if (!emb) return 0;
    const uint64_t n = (uint64_t)n_tok * dim;
    embed_tokens_bf16_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            (float *)out->ptr,
            emb,
            (const int32_t *)tokens->ptr,
            dim,
            vocab,
            n_tok);
    return cuda_ok(cudaGetLastError(), "embed_tokens_bf16 mapped launch");
}

__device__ static float warp_sum_f32(float v) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    return v;
}

__device__ static float sf37_bf16_to_f32_dev(uint16_t x) {
    return __uint_as_float((uint32_t)x << 16);
}

__device__ static float sf37_sigmoid_dev(float x) {
    if (x >= 0.0f) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    }
    float z = expf(x);
    return z / (1.0f + z);
}

__global__ static void rms_norm_weight1_kernel(float *out, const float *x,
                                               const float *w, uint32_t n,
                                               float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * (w[i] + 1.0f);
    }
}

extern "C" int sf37_cuda_rms_norm_weight1_f32(sf37_cuda_tensor *out,
                                               const sf37_cuda_tensor *x,
                                               const sf37_cuda_tensor *weight,
                                               uint32_t n, float eps) {
    if (!out || !x || !weight) return 0;
    if (out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float) ||
        weight->bytes < (uint64_t)n * sizeof(float)) return 0;
    rms_norm_weight1_kernel<<<1, 256>>>((float *)out->ptr,
                                        (const float *)x->ptr,
                                        (const float *)weight->ptr,
                                        n, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight1 launch");
}

__global__ static void rms_norm_weight1_bf16_kernel(float *out, const float *x,
                                                    const uint16_t *w,
                                                    uint32_t n, float eps) {
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = x[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * (sf37_bf16_to_f32_dev(w[i]) + 1.0f);
    }
}

__global__ static void rms_norm_weight1_bf16_batch_kernel(float *out,
                                                          const float *x,
                                                          const uint16_t *w,
                                                          uint32_t n,
                                                          uint32_t n_tok,
                                                          float eps) {
    const uint32_t t = blockIdx.x;
    if (t >= n_tok) return;
    const float *xr = x + (uint64_t)t * n;
    float *orow = out + (uint64_t)t * n;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        float v = xr[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        orow[i] = xr[i] * scale * (sf37_bf16_to_f32_dev(w[i]) + 1.0f);
    }
}

extern "C" int sf37_cuda_rms_norm_weight1_bf16(sf37_cuda_tensor *out,
                                                const sf37_cuda_tensor *x,
                                                const sf37_cuda_tensor *weight,
                                                uint32_t n, float eps) {
    if (!out || !x || !weight) return 0;
    if (out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float) ||
        weight->bytes < (uint64_t)n * sizeof(uint16_t)) return 0;
    rms_norm_weight1_bf16_kernel<<<1, 256>>>((float *)out->ptr,
                                             (const float *)x->ptr,
                                             (const uint16_t *)weight->ptr,
                                             n, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight1_bf16 launch");
}

__global__ static void add_inplace_f32_kernel(float *dst, const float *src, uint64_t n) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < n) dst[gid] += src[gid];
}

extern "C" int sf37_cuda_add_inplace_f32(sf37_cuda_tensor *dst,
                                          const sf37_cuda_tensor *src,
                                          uint64_t n) {
    if (!dst || !src) return 0;
    const uint64_t bytes = n * sizeof(float);
    if (dst->bytes < bytes || src->bytes < bytes) return 0;
    add_inplace_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            (float *)dst->ptr, (const float *)src->ptr, n);
    return cuda_ok(cudaGetLastError(), "add inplace f32 launch");
}

__global__ static void swiglu_kernel(float *out, const float *gate,
                                     const float *up, uint64_t n,
                                     float clamp) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    float g = gate[gid];
    float u = up[gid];
    if (clamp > 1.0e-6f) {
        if (g > clamp) g = clamp;
        if (u > clamp) u = clamp;
        if (u < -clamp) u = -clamp;
    }
    out[gid] = g * sf37_sigmoid_dev(g) * u;
}

extern "C" int sf37_cuda_swiglu_f32(sf37_cuda_tensor *out,
                                     const sf37_cuda_tensor *gate,
                                     const sf37_cuda_tensor *up,
                                     uint64_t n, float clamp) {
    if (!out || !gate || !up) return 0;
    const uint64_t bytes = n * sizeof(float);
    if (out->bytes < bytes || gate->bytes < bytes || up->bytes < bytes) return 0;
    swiglu_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>((float *)out->ptr,
                                                          (const float *)gate->ptr,
                                                          (const float *)up->ptr,
                                                          n, clamp);
    return cuda_ok(cudaGetLastError(), "swiglu launch");
}

__global__ static void head_rms_norm_weight1_bf16_kernel(float *x,
                                                         const uint16_t *weight,
                                                         uint32_t n_head,
                                                         uint32_t head_dim,
                                                         float eps) {
    const uint32_t h = blockIdx.x;
    if (h >= n_head) return;
    float *head = x + (uint64_t)h * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = head[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] = head[i] * scale * (sf37_bf16_to_f32_dev(weight[i]) + 1.0f);
    }
}

extern "C" int sf37_cuda_head_rms_norm_weight1_bf16(sf37_cuda_tensor *x,
                                                     const sf37_cuda_tensor *weight,
                                                     uint32_t n_head,
                                                     uint32_t head_dim,
                                                     float eps) {
    if (!x || !weight) return 0;
    if (x->bytes < (uint64_t)n_head * head_dim * sizeof(float) ||
        weight->bytes < (uint64_t)head_dim * sizeof(uint16_t)) return 0;
    head_rms_norm_weight1_bf16_kernel<<<n_head, 256>>>((float *)x->ptr,
                                                       (const uint16_t *)weight->ptr,
                                                       n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_weight1_bf16 launch");
}

__device__ static double sf37_inv_freq_dev(uint32_t pair, uint32_t dim,
                                           double theta, int llama3) {
    double inv = 1.0 / pow(theta, (double)(pair * 2u) / (double)dim);
    if (!llama3) return inv;

    const double factor = 2.0;
    const double low_freq_factor = 1.0;
    const double high_freq_factor = 32.0;
    const double old_context_len = 131072.0;
    const double low_freq_wavelen = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;
    const double wavelen = 2.0 * 3.14159265358979323846 / inv;
    if (wavelen > low_freq_wavelen) return inv / factor;
    if (wavelen < high_freq_wavelen) return inv;

    const double smooth = (old_context_len / wavelen - low_freq_factor) /
                          (high_freq_factor - low_freq_factor);
    return (1.0 - smooth) * inv / factor + smooth * inv;
}

__global__ static void rope_split_half_kernel(float *x,
                                              uint32_t n_head,
                                              uint32_t head_dim,
                                              uint32_t rotary_dim,
                                              double theta,
                                              int llama3,
                                              uint32_t pos) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs = (uint64_t)n_head * (rotary_dim / 2u);
    if (gid >= pairs) return;
    const uint32_t pair = (uint32_t)(gid % (rotary_dim / 2u));
    const uint32_t h = (uint32_t)(gid / (rotary_dim / 2u));
    float *head = x + (uint64_t)h * head_dim;
    const uint32_t half = rotary_dim / 2u;
    const double inv = sf37_inv_freq_dev(pair, rotary_dim, theta, llama3);
    const float c = cosf((float)((double)pos * inv));
    const float s = sinf((float)((double)pos * inv));
    const float a = head[pair];
    const float b = head[pair + half];
    head[pair] = a * c - b * s;
    head[pair + half] = b * c + a * s;
}

extern "C" int sf37_cuda_rope_split_half(sf37_cuda_tensor *x,
                                          uint32_t n_head,
                                          uint32_t head_dim,
                                          uint32_t rotary_dim,
                                          double theta,
                                          int llama3,
                                          uint32_t pos) {
    if (!x || rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1u)) return 0;
    if (x->bytes < (uint64_t)n_head * head_dim * sizeof(float)) return 0;
    const uint64_t pairs = (uint64_t)n_head * (rotary_dim / 2u);
    rope_split_half_kernel<<<(unsigned)((pairs + 255u) / 256u), 256>>>(
            (float *)x->ptr, n_head, head_dim, rotary_dim, theta, llama3, pos);
    return cuda_ok(cudaGetLastError(), "rope_split_half launch");
}

__global__ static void gqa_single_token_heads_kernel(float *out_heads,
                                                     const float *v,
                                                     const float *head_gate,
                                                     uint32_t q_heads,
                                                     uint32_t kv_heads,
                                                     uint32_t head_dim) {
    uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t n = (uint64_t)q_heads * head_dim;
    if (gid >= n) return;
    const uint32_t h = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)h * head_dim);
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    out_heads[gid] = v[(uint64_t)kvh * head_dim + d] * sf37_sigmoid_dev(head_gate[h]);
}

extern "C" int sf37_cuda_gqa_single_token_heads(sf37_cuda_tensor *out_heads,
                                                 const sf37_cuda_tensor *v,
                                                 const sf37_cuda_tensor *head_gate,
                                                 uint32_t q_heads,
                                                 uint32_t kv_heads,
                                                 uint32_t head_dim) {
    if (!out_heads || !v || !head_gate || kv_heads == 0 || q_heads % kv_heads != 0) return 0;
    const uint64_t q_dim = (uint64_t)q_heads * head_dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * head_dim;
    if (out_heads->bytes < q_dim * sizeof(float) ||
        v->bytes < kv_dim * sizeof(float) ||
        head_gate->bytes < (uint64_t)q_heads * sizeof(float)) return 0;
    gqa_single_token_heads_kernel<<<(unsigned)((q_dim + 255u) / 256u), 256>>>(
            (float *)out_heads->ptr,
            (const float *)v->ptr,
            (const float *)head_gate->ptr,
            q_heads, kv_heads, head_dim);
    return cuda_ok(cudaGetLastError(), "gqa single-token heads launch");
}

__global__ static void attention_decode_heads_kernel(float *out_heads,
                                                     const float *q,
                                                     const float *k_cache,
                                                     const float *v_cache,
                                                     const float *head_gate,
                                                     uint32_t n_cache,
                                                     uint32_t cache_cap,
                                                     uint32_t q_heads,
                                                     uint32_t kv_heads,
                                                     uint32_t head_dim,
                                                     int sliding,
                                                     uint32_t window) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)q_heads * head_dim;
    if (gid >= n || n_cache == 0) return;
    const uint32_t h = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)h * head_dim);
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    const uint64_t kv_row = (uint64_t)kv_heads * head_dim;
    const float *qh = q + (uint64_t)h * head_dim;
    uint32_t start = 0;
    if (sliding && n_cache > window) start = n_cache - window;

    float max_score = -INFINITY;
    for (uint32_t r = start; r < n_cache; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        float score = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) score += qh[i] * kh[i];
        score *= rsqrtf((float)head_dim);
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (uint32_t r = start; r < n_cache; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        const float *vh = v_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        float score = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) score += qh[i] * kh[i];
        score *= rsqrtf((float)head_dim);
        const float w = expf(score - max_score);
        denom += w;
        acc += w * vh[d];
    }
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    out_heads[gid] = acc * inv * sf37_sigmoid_dev(head_gate[h]);
}

__global__ static void attention_decode_heads128_warp_kernel(float *out_heads,
                                                             const float *q,
                                                             const float *k_cache,
                                                             const float *v_cache,
                                                             const float *head_gate,
                                                             uint32_t n_cache,
                                                             uint32_t cache_cap,
                                                             uint32_t q_heads,
                                                             uint32_t kv_heads,
                                                             int sliding,
                                                             uint32_t window) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t h = blockIdx.x * 8u + warp;
    if (h >= q_heads || n_cache == 0) return;

    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    const uint64_t kv_row = (uint64_t)kv_heads * 128u;
    const float *qh = q + (uint64_t)h * 128u;
    const float q0 = qh[lane + 0u];
    const float q1 = qh[lane + 32u];
    const float q2 = qh[lane + 64u];
    const float q3 = qh[lane + 96u];
    const float scale = rsqrtf(128.0f);

    uint32_t start = 0;
    if (sliding && n_cache > window) start = n_cache - window;

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    const unsigned mask = 0xffffffffu;

    for (uint32_t r = start; r < n_cache; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        const float *vh = v_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        float dot = q0 * kh[lane + 0u] +
                    q1 * kh[lane + 32u] +
                    q2 * kh[lane + 64u] +
                    q3 * kh[lane + 96u];
        dot = warp_sum_f32(dot);
        const float score = __shfl_sync(mask, dot, 0) * scale;

        const float old_m = max_s;
        const float new_m = fmaxf(max_s, score);
        const float old_scale = expf(old_m - new_m);
        const float row_scale = expf(score - new_m);
        sum_s = sum_s * old_scale + row_scale;
        o0 = o0 * old_scale + row_scale * vh[lane + 0u];
        o1 = o1 * old_scale + row_scale * vh[lane + 32u];
        o2 = o2 * old_scale + row_scale * vh[lane + 64u];
        o3 = o3 * old_scale + row_scale * vh[lane + 96u];
        max_s = new_m;
    }

    const float gate = sf37_sigmoid_dev(head_gate[h]);
    const float inv = sum_s > 0.0f ? 1.0f / sum_s : 0.0f;
    float *oh = out_heads + (uint64_t)h * 128u;
    oh[lane + 0u] = o0 * inv * gate;
    oh[lane + 32u] = o1 * inv * gate;
    oh[lane + 64u] = o2 * inv * gate;
    oh[lane + 96u] = o3 * inv * gate;
}

__global__ static void attention_decode_heads_at_kernel(float *out_heads,
                                                        const float *q,
                                                        const float *k_cache,
                                                        const float *v_cache,
                                                        const float *head_gate,
                                                        uint32_t pos,
                                                        uint32_t cache_cap,
                                                        uint32_t q_heads,
                                                        uint32_t kv_heads,
                                                        uint32_t head_dim,
                                                        int sliding,
                                                        uint32_t window) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)q_heads * head_dim;
    if (gid >= n || cache_cap == 0) return;
    const uint32_t h = (uint32_t)(gid / head_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)h * head_dim);
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    const uint64_t kv_row = (uint64_t)kv_heads * head_dim;
    const float *qh = q + (uint64_t)h * head_dim;
    const uint32_t have = pos + 1u;
    uint32_t start = 0;
    if (have > cache_cap) start = have - cache_cap;
    if (sliding && have > window) {
        const uint32_t sw = have - window;
        if (sw > start) start = sw;
    }

    float max_score = -INFINITY;
    for (uint32_t r = start; r <= pos; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        float score = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) score += qh[i] * kh[i];
        score *= rsqrtf((float)head_dim);
        if (score > max_score) max_score = score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (uint32_t r = start; r <= pos; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        const float *vh = v_cache + (uint64_t)row * kv_row + (uint64_t)kvh * head_dim;
        float score = 0.0f;
        for (uint32_t i = 0; i < head_dim; i++) score += qh[i] * kh[i];
        score *= rsqrtf((float)head_dim);
        const float w = expf(score - max_score);
        denom += w;
        acc += w * vh[d];
    }
    const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
    out_heads[gid] = acc * inv * sf37_sigmoid_dev(head_gate[h]);
}

__global__ static void attention_decode_heads128_warp_at_kernel(float *out_heads,
                                                                const float *q,
                                                                const float *k_cache,
                                                                const float *v_cache,
                                                                const float *head_gate,
                                                                uint32_t pos,
                                                                uint32_t cache_cap,
                                                                uint32_t q_heads,
                                                                uint32_t kv_heads,
                                                                int sliding,
                                                                uint32_t window) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t h = blockIdx.x * 8u + warp;
    if (h >= q_heads || cache_cap == 0) return;

    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    const uint64_t kv_row = (uint64_t)kv_heads * 128u;
    const float *qh = q + (uint64_t)h * 128u;
    const float q0 = qh[lane + 0u];
    const float q1 = qh[lane + 32u];
    const float q2 = qh[lane + 64u];
    const float q3 = qh[lane + 96u];
    const float scale = rsqrtf(128.0f);
    const uint32_t have = pos + 1u;
    uint32_t start = 0;
    if (have > cache_cap) start = have - cache_cap;
    if (sliding && have > window) {
        const uint32_t sw = have - window;
        if (sw > start) start = sw;
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    const unsigned mask = 0xffffffffu;

    for (uint32_t r = start; r <= pos; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        const float *vh = v_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        float dot = q0 * kh[lane + 0u] +
                    q1 * kh[lane + 32u] +
                    q2 * kh[lane + 64u] +
                    q3 * kh[lane + 96u];
        dot = warp_sum_f32(dot);
        const float score = __shfl_sync(mask, dot, 0) * scale;

        const float old_m = max_s;
        const float new_m = fmaxf(max_s, score);
        const float old_scale = expf(old_m - new_m);
        const float row_scale = expf(score - new_m);
        sum_s = sum_s * old_scale + row_scale;
        o0 = o0 * old_scale + row_scale * vh[lane + 0u];
        o1 = o1 * old_scale + row_scale * vh[lane + 32u];
        o2 = o2 * old_scale + row_scale * vh[lane + 64u];
        o3 = o3 * old_scale + row_scale * vh[lane + 96u];
        max_s = new_m;
    }

    const float gate = sf37_sigmoid_dev(head_gate[h]);
    const float inv = sum_s > 0.0f ? 1.0f / sum_s : 0.0f;
    float *oh = out_heads + (uint64_t)h * 128u;
    oh[lane + 0u] = o0 * inv * gate;
    oh[lane + 32u] = o1 * inv * gate;
    oh[lane + 64u] = o2 * inv * gate;
    oh[lane + 96u] = o3 * inv * gate;
}

extern "C" int sf37_cuda_attention_decode_heads(sf37_cuda_tensor *out_heads,
                                                 const sf37_cuda_tensor *q,
                                                 const sf37_cuda_tensor *k_cache,
                                                 const sf37_cuda_tensor *v_cache,
                                                 const sf37_cuda_tensor *head_gate,
                                                 uint32_t n_cache,
                                                 uint32_t cache_cap,
                                                 uint32_t q_heads,
                                                 uint32_t kv_heads,
                                                 uint32_t head_dim,
                                                 int sliding,
                                                 uint32_t window) {
    if (!out_heads || !q || !k_cache || !v_cache || !head_gate ||
        kv_heads == 0 || q_heads % kv_heads != 0 || n_cache > cache_cap) return 0;
    const uint64_t q_dim = (uint64_t)q_heads * head_dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * head_dim;
    if (out_heads->bytes < q_dim * sizeof(float) ||
        q->bytes < q_dim * sizeof(float) ||
        k_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        v_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        head_gate->bytes < (uint64_t)q_heads * sizeof(float)) return 0;
    if (head_dim == 128u && !cuda_env_present("SF37_CUDA_NO_WARP_ATTENTION", NULL)) {
        attention_decode_heads128_warp_kernel<<<(unsigned)((q_heads + 7u) / 8u), 256>>>(
                (float *)out_heads->ptr,
                (const float *)q->ptr,
                (const float *)k_cache->ptr,
                (const float *)v_cache->ptr,
                (const float *)head_gate->ptr,
                n_cache, cache_cap, q_heads, kv_heads, sliding, window);
        return cuda_ok(cudaGetLastError(), "attention_decode_heads128 warp launch");
    }
    attention_decode_heads_kernel<<<(unsigned)((q_dim + 255u) / 256u), 256>>>(
            (float *)out_heads->ptr,
            (const float *)q->ptr,
            (const float *)k_cache->ptr,
            (const float *)v_cache->ptr,
            (const float *)head_gate->ptr,
            n_cache, cache_cap, q_heads, kv_heads, head_dim, sliding, window);
    return cuda_ok(cudaGetLastError(), "attention_decode_heads launch");
}

extern "C" int sf37_cuda_attention_decode_heads_at(sf37_cuda_tensor *out_heads,
                                                    const sf37_cuda_tensor *q,
                                                    const sf37_cuda_tensor *k_cache,
                                                    const sf37_cuda_tensor *v_cache,
                                                    const sf37_cuda_tensor *head_gate,
                                                    uint32_t pos,
                                                    uint32_t cache_cap,
                                                    uint32_t q_heads,
                                                    uint32_t kv_heads,
                                                    uint32_t head_dim,
                                                    int sliding,
                                                    uint32_t window) {
    if (!out_heads || !q || !k_cache || !v_cache || !head_gate ||
        cache_cap == 0 || kv_heads == 0 || q_heads % kv_heads != 0 ||
        head_dim == 0) {
        return 0;
    }
    const uint64_t q_dim = (uint64_t)q_heads * head_dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * head_dim;
    if (out_heads->bytes < q_dim * sizeof(float) ||
        q->bytes < q_dim * sizeof(float) ||
        k_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        v_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        head_gate->bytes < (uint64_t)q_heads * sizeof(float)) return 0;
    if (head_dim == 128u && !cuda_env_present("SF37_CUDA_NO_WARP_ATTENTION", NULL)) {
        attention_decode_heads128_warp_at_kernel<<<(unsigned)((q_heads + 7u) / 8u), 256>>>(
                (float *)out_heads->ptr,
                (const float *)q->ptr,
                (const float *)k_cache->ptr,
                (const float *)v_cache->ptr,
                (const float *)head_gate->ptr,
                pos, cache_cap, q_heads, kv_heads, sliding, window);
        return cuda_ok(cudaGetLastError(), "attention_decode_heads128_at warp launch");
    }
    attention_decode_heads_at_kernel<<<(unsigned)((q_dim + 255u) / 256u), 256>>>(
            (float *)out_heads->ptr,
            (const float *)q->ptr,
            (const float *)k_cache->ptr,
            (const float *)v_cache->ptr,
            (const float *)head_gate->ptr,
            pos, cache_cap, q_heads, kv_heads, head_dim, sliding, window);
    return cuda_ok(cudaGetLastError(), "attention_decode_heads_at launch");
}

__global__ static void attention_prefill_heads128_warp_kernel(float *out_heads,
                                                              const float *q,
                                                              const float *k_cache,
                                                              const float *v_cache,
                                                              const float *head_gate,
                                                              uint32_t pos0,
                                                              uint32_t n_tok,
                                                              uint32_t cache_cap,
                                                              uint32_t q_heads,
                                                              uint32_t kv_heads,
                                                              int sliding,
                                                              uint32_t window) {
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t global_warp = blockIdx.x * 8u + warp;
    const uint32_t total = n_tok * q_heads;
    if (global_warp >= total) return;
    const uint32_t t = global_warp / q_heads;
    const uint32_t h = global_warp - t * q_heads;
    const uint32_t repeat = q_heads / kv_heads;
    const uint32_t kvh = h / repeat;
    const uint64_t q_dim = (uint64_t)q_heads * 128u;
    const uint64_t kv_row = (uint64_t)kv_heads * 128u;
    const float *qh = q + (uint64_t)t * q_dim + (uint64_t)h * 128u;
    const float q0 = qh[lane + 0u];
    const float q1 = qh[lane + 32u];
    const float q2 = qh[lane + 64u];
    const float q3 = qh[lane + 96u];
    const float scale = rsqrtf(128.0f);
    const uint32_t pos = pos0 + t;
    uint32_t start = 0;
    const uint32_t have = pos + 1u;
    if (have > cache_cap) start = have - cache_cap;
    if (sliding && have > window) {
        const uint32_t sw = have - window;
        if (sw > start) start = sw;
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    float o0 = 0.0f;
    float o1 = 0.0f;
    float o2 = 0.0f;
    float o3 = 0.0f;
    const unsigned mask = 0xffffffffu;

    for (uint32_t r = start; r <= pos; r++) {
        const uint32_t row = r % cache_cap;
        const float *kh = k_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        const float *vh = v_cache + (uint64_t)row * kv_row + (uint64_t)kvh * 128u;
        float dot = q0 * kh[lane + 0u] +
                    q1 * kh[lane + 32u] +
                    q2 * kh[lane + 64u] +
                    q3 * kh[lane + 96u];
        dot = warp_sum_f32(dot);
        const float score = __shfl_sync(mask, dot, 0) * scale;

        const float old_m = max_s;
        const float new_m = fmaxf(max_s, score);
        const float old_scale = expf(old_m - new_m);
        const float row_scale = expf(score - new_m);
        sum_s = sum_s * old_scale + row_scale;
        o0 = o0 * old_scale + row_scale * vh[lane + 0u];
        o1 = o1 * old_scale + row_scale * vh[lane + 32u];
        o2 = o2 * old_scale + row_scale * vh[lane + 64u];
        o3 = o3 * old_scale + row_scale * vh[lane + 96u];
        max_s = new_m;
    }

    const float gate = sf37_sigmoid_dev(head_gate[(uint64_t)t * q_heads + h]);
    const float inv = sum_s > 0.0f ? 1.0f / sum_s : 0.0f;
    float *oh = out_heads + (uint64_t)t * q_dim + (uint64_t)h * 128u;
    oh[lane + 0u] = o0 * inv * gate;
    oh[lane + 32u] = o1 * inv * gate;
    oh[lane + 64u] = o2 * inv * gate;
    oh[lane + 96u] = o3 * inv * gate;
}

extern "C" int sf37_cuda_attention_prefill_heads(sf37_cuda_tensor *out_heads,
                                                  const sf37_cuda_tensor *q,
                                                  const sf37_cuda_tensor *k_cache,
                                                  const sf37_cuda_tensor *v_cache,
                                                  const sf37_cuda_tensor *head_gate,
                                                  uint32_t pos0,
                                                  uint32_t n_tok,
                                                  uint32_t cache_cap,
                                                  uint32_t q_heads,
                                                  uint32_t kv_heads,
                                                  uint32_t head_dim,
                                                  int sliding,
                                                  uint32_t window) {
    if (!out_heads || !q || !k_cache || !v_cache || !head_gate ||
        n_tok == 0 || cache_cap == 0 ||
        kv_heads == 0 || q_heads % kv_heads != 0 || head_dim == 0) {
        return 0;
    }
    const uint64_t q_dim = (uint64_t)q_heads * head_dim;
    const uint64_t kv_dim = (uint64_t)kv_heads * head_dim;
    if (n_tok > UINT64_MAX / q_dim / sizeof(float) ||
        out_heads->bytes < (uint64_t)n_tok * q_dim * sizeof(float) ||
        q->bytes < (uint64_t)n_tok * q_dim * sizeof(float) ||
        head_gate->bytes < (uint64_t)n_tok * q_heads * sizeof(float) ||
        k_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        v_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float)) {
        return 0;
    }
    if (head_dim == 128u && !cuda_env_present("SF37_CUDA_NO_BATCH_ATTENTION", NULL)) {
        const uint64_t warps = (uint64_t)n_tok * q_heads;
        attention_prefill_heads128_warp_kernel<<<(unsigned)((warps + 7u) / 8u), 256>>>(
                (float *)out_heads->ptr,
                (const float *)q->ptr,
                (const float *)k_cache->ptr,
                (const float *)v_cache->ptr,
                (const float *)head_gate->ptr,
                pos0,
                n_tok,
                cache_cap,
                q_heads,
                kv_heads,
                sliding,
                window);
        return cuda_ok(cudaGetLastError(), "attention_prefill_heads128 warp launch");
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint32_t pos = pos0 + t;
        sf37_cuda_tensor out_view = {
            (char *)out_heads->ptr + (uint64_t)t * q_dim * sizeof(float),
            q_dim * sizeof(float)
        };
        sf37_cuda_tensor q_view = {
            (char *)q->ptr + (uint64_t)t * q_dim * sizeof(float),
            q_dim * sizeof(float)
        };
        sf37_cuda_tensor gate_view = {
            (char *)head_gate->ptr + (uint64_t)t * q_heads * sizeof(float),
            (uint64_t)q_heads * sizeof(float)
        };
        if (!sf37_cuda_attention_decode_heads_at(&out_view,
                                                 &q_view,
                                                 k_cache,
                                                 v_cache,
                                                 &gate_view,
                                                 pos,
                                                 cache_cap,
                                                 q_heads,
                                                 kv_heads,
                                                 head_dim,
                                                 sliding,
                                                 window)) {
            return 0;
        }
    }
    return 1;
}

__global__ static void quantize_q8_0_f32_kernel(int8_t *xq, float *xscale,
                                                const float *x, uint64_t in_dim,
                                                uint64_t blocks) {
    const uint64_t b = blockIdx.x;
    const uint64_t tok = blockIdx.y;
    if (b >= blocks) return;
    const uint64_t i0 = b * 32u;
    const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
    const float *xr = x + tok * in_dim + i0;

    float a = 0.0f;
    if (threadIdx.x < bn) a = fabsf(xr[threadIdx.x]);
    __shared__ float vals[32];
    vals[threadIdx.x] = a;
    __syncthreads();
    for (uint32_t stride = 16; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) vals[threadIdx.x] = fmaxf(vals[threadIdx.x], vals[threadIdx.x + stride]);
        __syncthreads();
    }
    const float d = vals[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (threadIdx.x == 0) xscale[tok * blocks + b] = d;
    int8_t *dst = xq + (tok * blocks + b) * 32u;
    if (threadIdx.x < bn) {
        int v = (int)lrintf(xr[threadIdx.x] * id);
        v = v > 127 ? 127 : (v < -128 ? -128 : v);
        dst[threadIdx.x] = (int8_t)v;
    } else {
        dst[threadIdx.x] = 0;
    }
}

__global__ static void f32_to_f16_kernel(__half *out, const float *x, uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2half(x[i]);
}

__global__ static void dequant_q8_0_to_f16_kernel(__half *out,
                                                  const uint8_t *w,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const uint8_t *blk = w + (row * blocks + b) * SF37_Q8_BLOCK_SIZE;
    const __half scale = *(const __half *)blk;
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[gid] = __hmul(scale, __float2half((float)q));
}

__global__ static void dequant_q8_0_to_f32_kernel(float *out,
                                                  const uint8_t *w,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  uint64_t blocks) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = in_dim * out_dim;
    if (gid >= n) return;
    const uint64_t row = gid / in_dim;
    const uint64_t i = gid - row * in_dim;
    const uint64_t b = i / 32u;
    const uint64_t j = i - b * 32u;
    const uint8_t *blk = w + (row * blocks + b) * SF37_Q8_BLOCK_SIZE;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2u + j);
    out[gid] = scale * (float)q;
}

__device__ __forceinline__ static int32_t load_i8x4_i32_unaligned_dev(const void *p) {
    const uint8_t *u = (const uint8_t *)p;
    return (int32_t)((uint32_t)u[0] |
                     ((uint32_t)u[1] << 8) |
                     ((uint32_t)u[2] << 16) |
                     ((uint32_t)u[3] << 24));
}

__device__ __forceinline__ static int32_t load_i8x4_i32_aligned_dev(const void *p) {
    return *(const int32_t *)p;
}

__device__ __forceinline__ static int32_t dot_i8x32_dp4a_dev(const int8_t *a,
                                                             const int8_t *b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 610
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i += 4u) {
        dot = __dp4a(load_i8x4_i32_unaligned_dev(a + i),
                     load_i8x4_i32_aligned_dev(b + i),
                     dot);
    }
    return dot;
#else
    int32_t dot = 0;
#pragma unroll
    for (uint32_t i = 0; i < 32u; i++) dot += (int32_t)a[i] * (int32_t)b[i];
    return dot;
#endif
}

__device__ __forceinline__ static int dot_i8_block_dev(const int8_t *a,
                                                       const int8_t *b,
                                                       uint64_t n,
                                                       int use_dp4a) {
    if (use_dp4a && n == 32u) return dot_i8x32_dp4a_dev(a, b);
    int acc = 0;
    for (uint64_t i = 0; i < n; i++) acc += (int)a[i] * (int)b[i];
    return acc;
}

__global__ static void matvec_q8_0_preq_kernel(float *out,
                                               const uint8_t *w,
                                               const int8_t *xq,
                                               const float *xscale,
                                               uint64_t in_dim,
                                               uint64_t out_dim,
                                               uint64_t blocks,
                                               int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint64_t row_bytes = blocks * SF37_Q8_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const uint8_t *blk = wr + b * SF37_Q8_BLOCK_SIZE;
        const float ws = __half2float(*(const __half *)blk);
        const int8_t *wq = (const int8_t *)(blk + 2);
        const int8_t *xqb = xq + b * 32u;
        acc += ws * xscale[b] * (float)dot_i8_block_dev(wq, xqb, bn, use_dp4a);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_q8_0_preq_batch_warp8_kernel(float *out,
                                                           const uint8_t *w,
                                                           const int8_t *xq,
                                                           const float *xscale,
                                                           uint64_t in_dim,
                                                           uint64_t out_dim,
                                                           uint64_t n_tok,
                                                           uint64_t blocks,
                                                           int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = (uint64_t)blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;

    const uint64_t row_bytes = blocks * SF37_Q8_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    const int8_t *xqr = xq + tok * blocks * 32u;
    const float *xsr = xscale + tok * blocks;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const uint8_t *blk = wr + b * SF37_Q8_BLOCK_SIZE;
        const float ws = __half2float(*(const __half *)blk);
        const int8_t *wq = (const int8_t *)(blk + 2);
        const int8_t *xqb = xqr + b * 32u;
        acc += ws * xsr[b] * (float)dot_i8_block_dev(wq, xqb, bn, use_dp4a);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void matvec_q8_0_pair_preq_kernel(float *out0,
                                                    float *out1,
                                                    const uint8_t *w0,
                                                    const uint8_t *w1,
                                                    const int8_t *xq,
                                                    const float *xscale,
                                                    uint64_t in_dim,
                                                    uint64_t out_dim,
                                                    uint64_t blocks,
                                                    int use_dp4a) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint64_t row_bytes = blocks * SF37_Q8_BLOCK_SIZE;
    const uint8_t *wr0 = w0 + row * row_bytes;
    const uint8_t *wr1 = w1 + row * row_bytes;
    float acc0 = 0.0f;
    float acc1 = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        const uint64_t i0 = b * 32u;
        const uint64_t bn = in_dim - i0 < 32u ? in_dim - i0 : 32u;
        const int8_t *xqb = xq + b * 32u;
        const float xs = xscale[b];

        const uint8_t *blk0 = wr0 + b * SF37_Q8_BLOCK_SIZE;
        const float ws0 = __half2float(*(const __half *)blk0);
        const int8_t *wq0 = (const int8_t *)(blk0 + 2);
        acc0 += ws0 * xs * (float)dot_i8_block_dev(wq0, xqb, bn, use_dp4a);

        const uint8_t *blk1 = wr1 + b * SF37_Q8_BLOCK_SIZE;
        const float ws1 = __half2float(*(const __half *)blk1);
        const int8_t *wq1 = (const int8_t *)(blk1 + 2);
        acc1 += ws1 * xs * (float)dot_i8_block_dev(wq1, xqb, bn, use_dp4a);
    }
    acc0 = warp_sum_f32(acc0);
    acc1 = warp_sum_f32(acc1);
    if (lane == 0) {
        out0[row] = acc0;
        out1[row] = acc1;
    }
}

extern "C" int sf37_cuda_matvec_q8_0(sf37_cuda_tensor *out,
                                      const sf37_cuda_tensor *weights,
                                      uint64_t in_dim, uint64_t out_dim,
                                      const sf37_cuda_tensor *x) {
    if (!out || !weights || !x) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q8_BLOCK_SIZE;
    if (out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights->bytes < weight_bytes) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr,
                                                       in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 quantize launch")) return 0;
    const int use_dp4a = cuda_q8_use_dp4a();
    matvec_q8_0_preq_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr,
            (const uint8_t *)weights->ptr,
            xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "q8_0 matvec launch");
}

extern "C" int sf37_cuda_matvec_q8_0_pair(sf37_cuda_tensor *out0,
                                           sf37_cuda_tensor *out1,
                                           const sf37_cuda_tensor *weights0,
                                           const sf37_cuda_tensor *weights1,
                                           uint64_t in_dim, uint64_t out_dim,
                                           const sf37_cuda_tensor *x) {
    if (!out0 || !out1 || !weights0 || !weights1 || !x) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q8_BLOCK_SIZE;
    if (out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights0->bytes < weight_bytes ||
        weights1->bytes < weight_bytes) return 0;
    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale, (const float *)x->ptr,
                                                       in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 pair quantize launch")) return 0;
    const unsigned grid = (unsigned)((out_dim + 7u) / 8u);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!cuda_env_present("SF37_CUDA_NO_Q8_PAIR_FUSED", NULL)) {
        matvec_q8_0_pair_preq_kernel<<<grid, 256>>>(
                (float *)out0->ptr,
                (float *)out1->ptr,
                (const uint8_t *)weights0->ptr,
                (const uint8_t *)weights1->ptr,
                xq, xscale, in_dim, out_dim, blocks, use_dp4a);
        return cuda_ok(cudaGetLastError(), "q8_0 pair fused matvec launch");
    }
    matvec_q8_0_preq_kernel<<<grid, 256>>>(
            (float *)out0->ptr,
            (const uint8_t *)weights0->ptr,
            xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "q8_0 pair matvec0 launch")) return 0;
    matvec_q8_0_preq_kernel<<<grid, 256>>>(
            (float *)out1->ptr,
            (const uint8_t *)weights1->ptr,
            xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "q8_0 pair matvec1 launch");
}

__global__ static void matvec_bf16_kernel(float *out, const uint16_t *w,
                                          uint64_t in_dim, uint64_t out_dim,
                                          const float *x) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const uint16_t *wr = w + row * in_dim;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        acc += sf37_bf16_to_f32_dev(wr[i]) * x[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_bf16_kernel(float *out,
                                          const uint16_t *w,
                                          uint64_t in_dim,
                                          uint64_t out_dim,
                                          const float *x,
                                          uint64_t n_tok) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;
    const uint16_t *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        acc += sf37_bf16_to_f32_dev(wr[i]) * xr[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

__global__ static void matvec_bf16_argmax_stage1_kernel(float *part_val,
                                                        int32_t *part_idx,
                                                        const uint16_t *w,
                                                        uint64_t in_dim,
                                                        uint64_t out_dim,
                                                        const float *x,
                                                        int32_t excluded_id) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    __shared__ float warp_val[8];
    __shared__ int32_t warp_idx[8];

    float acc = -INFINITY;
    int32_t idx = INT32_MAX;
    if (row < out_dim && (int32_t)row != excluded_id) {
        const uint16_t *wr = w + row * in_dim;
        acc = 0.0f;
        for (uint64_t i = lane; i < in_dim; i += 32u) {
            acc += sf37_bf16_to_f32_dev(wr[i]) * x[i];
        }
        acc = warp_sum_f32(acc);
        idx = (int32_t)row;
    }
    if (lane == 0) {
        const uint32_t wr = threadIdx.x >> 5u;
        warp_val[wr] = acc;
        warp_idx[wr] = idx;
    }
    __syncthreads();

    if (threadIdx.x < 8u) {
        float best_v = warp_val[threadIdx.x];
        int32_t best_i = warp_idx[threadIdx.x];
        for (uint32_t i = threadIdx.x + 1u; i < 8u; i++) {
            const float v = warp_val[i];
            const int32_t id = warp_idx[i];
            if (v > best_v || (v == best_v && id < best_i)) {
                best_v = v;
                best_i = id;
            }
        }
        warp_val[threadIdx.x] = best_v;
        warp_idx[threadIdx.x] = best_i;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        part_val[blockIdx.x] = warp_val[0];
        part_idx[blockIdx.x] = warp_idx[0] == INT32_MAX ? 0 : warp_idx[0];
    }
}

__global__ static void argmax_partials_kernel(int32_t *out_idx,
                                              const float *part_val,
                                              const int32_t *part_idx,
                                              uint32_t n_part) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float best_v = -INFINITY;
    int32_t best_i = INT32_MAX;
    for (uint32_t i = tid; i < n_part; i += THREADS) {
        const float v = part_val[i];
        const int32_t id = part_idx[i];
        if (v > best_v || (v == best_v && id < best_i)) {
            best_v = v;
            best_i = id;
        }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float rv = sm_val[tid + s];
            const int32_t ri = sm_idx[tid + s];
            const float lv = sm_val[tid];
            const int32_t li = sm_idx[tid];
            if (rv > lv || (rv == lv && ri < li)) {
                sm_val[tid] = rv;
                sm_idx[tid] = ri;
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sm_idx[0] == INT32_MAX ? 0 : sm_idx[0];
}

__global__ static void argmax_f32_kernel(int32_t *out_idx,
                                         const float *logits,
                                         uint32_t n_vocab,
                                         int32_t excluded_id) {
    enum { THREADS = 1024 };
    __shared__ float sm_val[THREADS];
    __shared__ int32_t sm_idx[THREADS];

    const uint32_t tid = threadIdx.x;
    float best_v = -INFINITY;
    int32_t best_i = INT32_MAX;
    for (uint32_t i = tid; i < n_vocab; i += THREADS) {
        if ((int32_t)i == excluded_id) continue;
        const float v = logits[i];
        if (v > best_v || (v == best_v && (int32_t)i < best_i)) {
            best_v = v;
            best_i = (int32_t)i;
        }
    }
    sm_val[tid] = best_v;
    sm_idx[tid] = best_i;
    __syncthreads();

    for (uint32_t s = THREADS / 2u; s > 0u; s >>= 1u) {
        if (tid < s) {
            const float rv = sm_val[tid + s];
            const int32_t ri = sm_idx[tid + s];
            const float lv = sm_val[tid];
            const int32_t li = sm_idx[tid];
            if (rv > lv || (rv == lv && ri < li)) {
                sm_val[tid] = rv;
                sm_idx[tid] = ri;
            }
        }
        __syncthreads();
    }
    if (tid == 0) *out_idx = sm_idx[0] == INT32_MAX ? 0 : sm_idx[0];
}

__global__ static void f32_to_bf16_kernel(__nv_bfloat16 *out,
                                          const float *in,
                                          uint64_t n) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}

static int cuda_matvec_bf16_cublas(float *out,
                                   const uint16_t *w,
                                   uint64_t in_dim,
                                   uint64_t out_dim,
                                   const float *x,
                                   const char *label) {
    if (!g_cublas_ready || !cuda_bf16_cublas_enabled()) return 0;
    if (in_dim == 0 || out_dim == 0 ||
        out_dim < cuda_bf16_cublas_min_out_dim() ||
        in_dim > (uint64_t)INT_MAX ||
        out_dim > (uint64_t)INT_MAX) {
        return 0;
    }

    __nv_bfloat16 *xb = (__nv_bfloat16 *)cuda_tmp_alloc(
            in_dim * sizeof(__nv_bfloat16),
            label ? label : "bf16 cublas activation");
    if (!xb) return 0;

    f32_to_bf16_kernel<<<(unsigned)((in_dim + 255u) / 256u), 256>>>(xb, x, in_dim);
    if (!cuda_ok(cudaGetLastError(), "bf16 activation convert launch")) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     1,
                                     (int)in_dim,
                                     &alpha,
                                     w,
                                     CUDA_R_16BF,
                                     (int)in_dim,
                                     xb,
                                     CUDA_R_16BF,
                                     (int)in_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;

    fprintf(stderr,
            "sf37: cuBLAS bf16 matvec failed for %s: status %d; falling back to native kernel\n",
            label ? label : "bf16", (int)st);
    return 0;
}

static int cuda_matmul_bf16_cublas(float *out,
                                   const uint16_t *w,
                                   uint64_t in_dim,
                                   uint64_t out_dim,
                                   const float *x,
                                   uint64_t n_tok,
                                   const char *label) {
    if (!g_cublas_ready || !cuda_bf16_cublas_enabled() || n_tok == 0) return 0;
    if (in_dim == 0 || out_dim == 0 ||
        out_dim < cuda_bf16_cublas_min_out_dim() ||
        in_dim > (uint64_t)INT_MAX ||
        out_dim > (uint64_t)INT_MAX ||
        n_tok > (uint64_t)INT_MAX) {
        return 0;
    }

    const uint64_t x_count = n_tok * in_dim;
    __nv_bfloat16 *xb = (__nv_bfloat16 *)cuda_tmp_alloc(
            x_count * sizeof(__nv_bfloat16),
            label ? label : "bf16 cublas batch activations");
    if (!xb) return 0;

    f32_to_bf16_kernel<<<(unsigned)((x_count + 255u) / 256u), 256>>>(xb, x, x_count);
    if (!cuda_ok(cudaGetLastError(), "bf16 batch activation convert launch")) return 0;

    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasStatus_t st = cublasGemmEx(g_cublas,
                                     CUBLAS_OP_T,
                                     CUBLAS_OP_N,
                                     (int)out_dim,
                                     (int)n_tok,
                                     (int)in_dim,
                                     &alpha,
                                     w,
                                     CUDA_R_16BF,
                                     (int)in_dim,
                                     xb,
                                     CUDA_R_16BF,
                                     (int)in_dim,
                                     &beta,
                                     out,
                                     CUDA_R_32F,
                                     (int)out_dim,
                                     CUBLAS_COMPUTE_32F,
                                     CUBLAS_GEMM_DEFAULT);
    if (st == CUBLAS_STATUS_SUCCESS) return 1;

    fprintf(stderr,
            "sf37: cuBLAS bf16 matmul failed for %s: status %d; falling back to native kernel\n",
            label ? label : "bf16", (int)st);
    return 0;
}

extern "C" int sf37_cuda_matvec_bf16(sf37_cuda_tensor *out,
                                      const sf37_cuda_tensor *weights,
                                      uint64_t in_dim, uint64_t out_dim,
                                      const sf37_cuda_tensor *x) {
    if (!out || !weights || !x) return 0;
    if (out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights->bytes < out_dim * in_dim * sizeof(uint16_t)) return 0;
    if (cuda_matvec_bf16_cublas((float *)out->ptr,
                                (const uint16_t *)weights->ptr,
                                in_dim, out_dim,
                                (const float *)x->ptr,
                                "bf16 matvec")) {
        return 1;
    }
    matvec_bf16_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr,
            (const uint16_t *)weights->ptr,
            in_dim, out_dim,
            (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "bf16 matvec launch");
}

extern "C" int sf37_cuda_argmax_f32(int32_t *out_token,
                                     const sf37_cuda_tensor *logits,
                                     uint32_t n_vocab,
                                     int32_t excluded_id) {
    if (!out_token || !logits || n_vocab == 0 ||
        logits->bytes < (uint64_t)n_vocab * sizeof(float)) {
        return 0;
    }
    int32_t *dev_out = (int32_t *)cuda_tmp_alloc(sizeof(int32_t), "f32 argmax");
    if (!dev_out) return 0;
    argmax_f32_kernel<<<1, 1024>>>(dev_out,
                                   (const float *)logits->ptr,
                                   n_vocab,
                                   excluded_id);
    if (!cuda_ok(cudaGetLastError(), "f32 argmax launch")) return 0;
    return cuda_ok(cudaMemcpy(out_token, dev_out, sizeof(*out_token),
                              cudaMemcpyDeviceToHost),
                   "f32 argmax read");
}

__global__ static void matvec_f32_kernel(float *out, const float *w,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const float *x) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const float *wr = w + row * in_dim;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) acc += wr[i] * x[i];
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matmul_f32_kernel(float *out,
                                         const float *w,
                                         uint64_t in_dim,
                                         uint64_t out_dim,
                                         const float *x,
                                         uint64_t n_tok) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint64_t tok = blockIdx.y;
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim || tok >= n_tok) return;
    const float *wr = w + row * in_dim;
    const float *xr = x + tok * in_dim;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) acc += wr[i] * xr[i];
    acc = warp_sum_f32(acc);
    if (lane == 0) out[tok * out_dim + row] = acc;
}

extern "C" int sf37_cuda_matvec_f32(sf37_cuda_tensor *out,
                                     const sf37_cuda_tensor *weights,
                                     uint64_t in_dim, uint64_t out_dim,
                                     const sf37_cuda_tensor *x) {
    if (!out || !weights || !x) return 0;
    if (out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights->bytes < out_dim * in_dim * sizeof(float)) return 0;
    matvec_f32_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr,
            (const float *)weights->ptr,
            in_dim, out_dim,
            (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "f32 matvec launch");
}

__device__ static uint32_t q3_payload_get(const uint8_t *payload, uint32_t idx) {
    const uint32_t bit = idx * 3u;
    const uint32_t byte = bit >> 3u;
    const uint32_t shift = bit & 7u;
    uint32_t word = payload[byte];
    if (byte + 1u < 24u) word |= (uint32_t)payload[byte + 1u] << 8u;
    return (word >> shift) & 7u;
}

__device__ static uint32_t q2_payload_get(const uint8_t *payload, uint32_t idx) {
    return (payload[idx >> 2u] >> ((idx & 3u) * 2u)) & 3u;
}

typedef struct {
    float d;
    int8_t qs[SF37_QK_K];
    int16_t gsums[4];
} sf37_cuda_block_q8_k;

__global__ static void q8_k_quantize_sf37_kernel(sf37_cuda_block_q8_k *out,
                                                 const float *x,
                                                 uint32_t in_dim,
                                                 uint32_t n_rows) {
    const uint32_t b = blockIdx.x;
    const uint32_t row = blockIdx.y;
    const uint32_t blocks = in_dim / SF37_QK_K;
    if (row >= n_rows || b >= blocks) return;

    const uint32_t tid = threadIdx.x;
    const float *xr = x + (uint64_t)row * in_dim + (uint64_t)b * SF37_QK_K;
    sf37_cuda_block_q8_k *yb = out + (uint64_t)row * blocks + b;

    __shared__ float abs_part[SF37_QK_K];
    const float xv = tid < SF37_QK_K ? xr[tid] : 0.0f;
    abs_part[tid] = tid < SF37_QK_K ? fabsf(xv) : 0.0f;
    __syncthreads();
    for (uint32_t stride = SF37_QK_K >> 1u; stride > 0; stride >>= 1u) {
        if (tid < stride) abs_part[tid] = fmaxf(abs_part[tid], abs_part[tid + stride]);
        __syncthreads();
    }

    const float d = abs_part[0] / 127.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    if (tid == 0) yb->d = d;
    if (tid < SF37_QK_K) {
        int q = (int)lrintf(xv * id);
        q = q > 127 ? 127 : (q < -128 ? -128 : q);
        yb->qs[tid] = (int8_t)q;
    }
    __syncthreads();
    if (tid < 4u) {
        int sum = 0;
        const uint32_t off = tid * 64u;
        for (uint32_t i = 0; i < 64u; i++) sum += (int)yb->qs[off + i];
        yb->gsums[tid] = (int16_t)sum;
    }
}

__device__ __forceinline__ static int32_t q3_i8x4_pack_dev(const uint8_t *payload,
                                                           uint32_t idx) {
    const uint32_t q0 = q3_payload_get(payload, idx + 0u);
    const uint32_t q1 = q3_payload_get(payload, idx + 1u);
    const uint32_t q2 = q3_payload_get(payload, idx + 2u);
    const uint32_t q3 = q3_payload_get(payload, idx + 3u);
    return (int32_t)(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
}

__device__ __forceinline__ static int32_t q2_i8x4_pack_dev(const uint8_t *payload,
                                                           uint32_t idx) {
    const uint32_t v = payload[idx >> 2u];
    const uint32_t q0 = v & 3u;
    const uint32_t q1 = (v >> 2u) & 3u;
    const uint32_t q2 = (v >> 4u) & 3u;
    const uint32_t q3 = (v >> 6u) & 3u;
    return (int32_t)(q0 | (q1 << 8u) | (q2 << 16u) | (q3 << 24u));
}

__device__ static float dot_q3_asym_q8k_block_dev(const uint8_t *blk,
                                                  const sf37_cuda_block_q8_k *xq) {
    float acc = 0.0f;
    const int8_t *xqs = xq->qs;
    const float xd = xq->d;
    if (xd == 0.0f) return 0.0f;
    for (uint32_t g = 0; g < 4u; g++) {
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const int32_t zp = (int32_t)(blk[8u + g] & 7u);
        const uint8_t *payload = blk + 12u + g * 24u;
        int32_t dot = 0;
        const uint32_t off = g * 64u;
        for (uint32_t i = 0; i < 64u; i += 4u) {
            dot = __dp4a(q3_i8x4_pack_dev(payload, i),
                         *(const int32_t *)(xqs + off + i),
                         dot);
        }
        dot -= zp * (int32_t)xq->gsums[g];
        acc += scale * xd * (float)dot;
    }
    return acc;
}

__device__ static float dot_q2_asym_q8k_block_dev(const uint8_t *blk,
                                                  const sf37_cuda_block_q8_k *xq) {
    float acc = 0.0f;
    const int8_t *xqs = xq->qs;
    const float xd = xq->d;
    if (xd == 0.0f) return 0.0f;
    for (uint32_t g = 0; g < 4u; g++) {
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const int32_t zp = (int32_t)(blk[8u + g] & 3u);
        const uint8_t *payload = blk + 12u + g * 16u;
        int32_t dot = 0;
        const uint32_t off = g * 64u;
        for (uint32_t i = 0; i < 64u; i += 4u) {
            dot = __dp4a(q2_i8x4_pack_dev(payload, i),
                         *(const int32_t *)(xqs + off + i),
                         dot);
        }
        dot -= zp * (int32_t)xq->gsums[g];
        acc += scale * xd * (float)dot;
    }
    return acc;
}

__device__ static void dot_q3_asym_q8k_block8_dev(
        const uint8_t *blk,
        const sf37_cuda_block_q8_k *x0,
        const sf37_cuda_block_q8_k *x1,
        const sf37_cuda_block_q8_k *x2,
        const sf37_cuda_block_q8_k *x3,
        const sf37_cuda_block_q8_k *x4,
        const sf37_cuda_block_q8_k *x5,
        const sf37_cuda_block_q8_k *x6,
        const sf37_cuda_block_q8_k *x7,
        uint32_t np,
        float acc[8]) {
    const sf37_cuda_block_q8_k *xs[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
    for (uint32_t g = 0; g < 4u; g++) {
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const int32_t zp = (int32_t)(blk[8u + g] & 7u);
        const uint8_t *payload = blk + 12u + g * 24u;
        const uint32_t off = g * 64u;
        int32_t dot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t i = 0; i < 64u; i += 4u) {
            const int32_t q = q3_i8x4_pack_dev(payload, i);
#pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                if (p < np && xs[p]) {
                    dot[p] = __dp4a(q, *(const int32_t *)(xs[p]->qs + off + i), dot[p]);
                }
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < np && xs[p]) {
                const float xd = xs[p]->d;
                if (xd != 0.0f) {
                    const int32_t corrected = dot[p] - zp * (int32_t)xs[p]->gsums[g];
                    acc[p] += scale * xd * (float)corrected;
                }
            }
        }
    }
}

__device__ static void dot_q2_asym_q8k_block8_dev(
        const uint8_t *blk,
        const sf37_cuda_block_q8_k *x0,
        const sf37_cuda_block_q8_k *x1,
        const sf37_cuda_block_q8_k *x2,
        const sf37_cuda_block_q8_k *x3,
        const sf37_cuda_block_q8_k *x4,
        const sf37_cuda_block_q8_k *x5,
        const sf37_cuda_block_q8_k *x6,
        const sf37_cuda_block_q8_k *x7,
        uint32_t np,
        float acc[8]) {
    const sf37_cuda_block_q8_k *xs[8] = {x0, x1, x2, x3, x4, x5, x6, x7};
    for (uint32_t g = 0; g < 4u; g++) {
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const int32_t zp = (int32_t)(blk[8u + g] & 3u);
        const uint8_t *payload = blk + 12u + g * 16u;
        const uint32_t off = g * 64u;
        int32_t dot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (uint32_t i = 0; i < 64u; i += 4u) {
            const int32_t q = q2_i8x4_pack_dev(payload, i);
#pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                if (p < np && xs[p]) {
                    dot[p] = __dp4a(q, *(const int32_t *)(xs[p]->qs + off + i), dot[p]);
                }
            }
        }
#pragma unroll
        for (uint32_t p = 0; p < 8u; p++) {
            if (p < np && xs[p]) {
                const float xd = xs[p]->d;
                if (xd != 0.0f) {
                    const int32_t corrected = dot[p] - zp * (int32_t)xs[p]->gsums[g];
                    acc[p] += scale * xd * (float)corrected;
                }
            }
        }
    }
}

__device__ __forceinline__ static float quarter_warp_sum_f32(float v) {
    v += __shfl_down_sync(0xffffffffu, v, 4, 8);
    v += __shfl_down_sync(0xffffffffu, v, 2, 8);
    v += __shfl_down_sync(0xffffffffu, v, 1, 8);
    return v;
}

__global__ static void matvec_q3_asym_q8k_kernel(float *out,
                                                const uint8_t *w,
                                                const sf37_cuda_block_q8_k *xq,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const uint64_t row_bytes = blocks * SF37_Q3_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        acc += dot_q3_asym_q8k_block_dev(wr + b * SF37_Q3_BLOCK_SIZE, xq + b);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matvec_q2_asym_q8k_kernel(float *out,
                                                const uint8_t *w,
                                                const sf37_cuda_block_q8_k *xq,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                uint64_t blocks) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;
    const uint64_t row_bytes = blocks * SF37_Q2_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t b = lane; b < blocks; b += 32u) {
        acc += dot_q2_asym_q8k_block_dev(wr + b * SF37_Q2_BLOCK_SIZE, xq + b);
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void moe_gate_up_mid_q3_asym_q8k_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const uint8_t *gate_w,
        const uint8_t *up_w,
        const sf37_cuda_block_q8_k *xq,
        const int32_t *selected,
        const float *weights,
        uint32_t topk,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        float clamp,
        int write_gate_up) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= topk) return;

    int32_t expert_i = selected[slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;
    const uint64_t row_q3_bytes = (uint64_t)xq_blocks * SF37_Q3_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)expert_mid_dim * row_q3_bytes;
    const uint8_t *gr = gate_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                        (uint64_t)row * row_q3_bytes;
    const uint8_t *ur = up_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                        (uint64_t)row * row_q3_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dot_q3_asym_q8k_block_dev(gr + (uint64_t)b * SF37_Q3_BLOCK_SIZE, xq + b);
        up += dot_q3_asym_q8k_block_dev(ur + (uint64_t)b * SF37_Q3_BLOCK_SIZE, xq + b);
    }
    gate = quarter_warp_sum_f32(gate);
    up = quarter_warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)slot * expert_mid_dim + row;
        if (write_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = gate * sf37_sigmoid_dev(gate) * up * weights[slot];
    }
}

__global__ static void moe_down_q2_asym_sum_q8k_kernel(
        float *out,
        const uint8_t *down_w,
        const sf37_cuda_block_q8_k *midq,
        const int32_t *selected,
        uint32_t topk,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_total_expert) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const int use_shared_midq = topk <= 8u && midq_blocks <= 8u;
    __shared__ sf37_cuda_block_q8_k shared_midq[8][8];
    if (use_shared_midq) {
        const uint32_t n = topk * midq_blocks;
        for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
            const uint32_t slot = i / midq_blocks;
            const uint32_t b = i - slot * midq_blocks;
            shared_midq[slot][b] = midq[(uint64_t)slot * midq_blocks + b];
        }
        __syncthreads();
    }
    if (row >= out_dim) return;

    const uint64_t row_q2_bytes = (uint64_t)midq_blocks * SF37_Q2_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_q2_bytes;
    float total = 0.0f;
    for (uint32_t slot = 0; slot < topk; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;
        const uint8_t *wr = down_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                            (uint64_t)row * row_q2_bytes;
        const sf37_cuda_block_q8_k *xq =
            use_shared_midq ? shared_midq[slot] : midq + (uint64_t)slot * midq_blocks;
        float acc = 0.0f;
        for (uint32_t b = lane; b < midq_blocks; b += 8u) {
            acc += dot_q2_asym_q8k_block_dev(wr + (uint64_t)b * SF37_Q2_BLOCK_SIZE,
                                             xq + b);
        }
        total += quarter_warp_sum_f32(acc);
    }
    if (lane == 0) out[row] = total;
}

__global__ static void moe_count_sorted_pairs_sf37_kernel(
        uint32_t *counts,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    const uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;
    atomicAdd(counts + (uint32_t)expert_i, 1u);
}

__global__ static void moe_prefix_sorted_pairs_sf37_kernel(
        uint32_t *offsets,
        uint32_t *cursors,
        const uint32_t *counts,
        uint32_t n_total_expert) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t sum = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        offsets[e] = sum;
        cursors[e] = sum;
        sum += counts[e];
    }
    offsets[n_total_expert] = sum;
}

__global__ static void moe_scatter_sorted_pairs_sf37_kernel(
        uint32_t *sorted_pairs,
        uint32_t *cursors,
        const int32_t *selected,
        uint32_t pair_count,
        uint32_t n_total_expert) {
    const uint32_t pair = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (pair >= pair_count) return;
    int32_t expert_i = selected[pair];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;
    const uint32_t pos = atomicAdd(cursors + (uint32_t)expert_i, 1u);
    sorted_pairs[pos] = pair;
}

__global__ static void moe_build_expert_tile_offsets_sf37_kernel(
        uint32_t *tile_offsets,
        uint32_t *tile_total,
        const uint32_t *counts,
        uint32_t n_total_expert,
        uint32_t tile_m) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    uint32_t sum = 0;
    for (uint32_t e = 0; e < n_total_expert; e++) {
        tile_offsets[e] = sum;
        sum += (counts[e] + tile_m - 1u) / tile_m;
    }
    tile_offsets[n_total_expert] = sum;
    *tile_total = sum;
}

__global__ static void moe_build_expert_tiles_sf37_kernel(
        uint32_t *tile_experts,
        uint32_t *tile_starts,
        const uint32_t *tile_offsets,
        const uint32_t *counts,
        uint32_t n_total_expert,
        uint32_t tile_m) {
    const uint32_t e = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    if (e >= n_total_expert) return;
    const uint32_t ntiles = (counts[e] + tile_m - 1u) / tile_m;
    const uint32_t off = tile_offsets[e];
    for (uint32_t t = 0; t < ntiles; t++) {
        tile_experts[off + t] = e;
        tile_starts[off + t] = t * tile_m;
    }
}

__global__ static void moe_gate_up_mid_q3_asym_q8k_sorted_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const uint8_t *gate_w,
        const uint8_t *up_w,
        const sf37_cuda_block_q8_k *xq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        const float *weights,
        uint32_t topk,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        uint32_t n_total_expert,
        float clamp,
        int write_gate_up) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= expert_mid_dim) return;

    const uint32_t tok = pair / topk;
    const uint32_t slot = pair - tok * topk;
    int32_t expert_i = selected[(uint64_t)tok * topk + slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;

    const uint64_t row_q3_bytes = (uint64_t)xq_blocks * SF37_Q3_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)expert_mid_dim * row_q3_bytes;
    const uint8_t *gr = gate_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                        (uint64_t)row * row_q3_bytes;
    const uint8_t *ur = up_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                        (uint64_t)row * row_q3_bytes;
    const sf37_cuda_block_q8_k *xqb = xq + (uint64_t)tok * xq_blocks;

    float gate = 0.0f;
    float up = 0.0f;
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        gate += dot_q3_asym_q8k_block_dev(gr + (uint64_t)b * SF37_Q3_BLOCK_SIZE, xqb + b);
        up += dot_q3_asym_q8k_block_dev(ur + (uint64_t)b * SF37_Q3_BLOCK_SIZE, xqb + b);
    }
    gate = quarter_warp_sum_f32(gate);
    up = quarter_warp_sum_f32(up);
    if (lane == 0) {
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)pair * expert_mid_dim + row;
        if (write_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = gate * sf37_sigmoid_dev(gate) * up *
                       weights[(uint64_t)tok * topk + slot];
    }
}

__global__ static void moe_gate_up_mid_q3_asym_q8k_expert_tile8_kernel(
        float *gate_out,
        float *up_out,
        float *mid_out,
        const uint8_t *gate_w,
        const uint8_t *up_w,
        const sf37_cuda_block_q8_k *xq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        const float *weights,
        uint32_t topk,
        uint32_t xq_blocks,
        uint32_t expert_mid_dim,
        float clamp,
        int write_gate_up) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    __shared__ sf37_cuda_block_q8_k sxq[8][16];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t tok[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    uint32_t slot[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const sf37_cuda_block_q8_k *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        const uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        tok[np] = pair[np] / topk;
        slot[np] = pair[np] - tok[np] * topk;
        xqb[np] = xq + (uint64_t)tok[np] * xq_blocks;
    }
    if (xq_blocks <= 16u) {
        for (uint32_t i = threadIdx.x; i < np * xq_blocks; i += blockDim.x) {
            const uint32_t p = i / xq_blocks;
            const uint32_t b = i - p * xq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= expert_mid_dim) return;

    const uint64_t row_q3_bytes = (uint64_t)xq_blocks * SF37_Q3_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)expert_mid_dim * row_q3_bytes;
    const uint8_t *gr = gate_w + (uint64_t)expert * expert_bytes +
                        (uint64_t)row * row_q3_bytes;
    const uint8_t *ur = up_w + (uint64_t)expert * expert_bytes +
                        (uint64_t)row * row_q3_bytes;
    float gate[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float up[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < xq_blocks; b += 8u) {
        const uint8_t *gb = gr + (uint64_t)b * SF37_Q3_BLOCK_SIZE;
        const uint8_t *ub = ur + (uint64_t)b * SF37_Q3_BLOCK_SIZE;
        dot_q3_asym_q8k_block8_dev(gb,
                                    xqb[0] ? xqb[0] + b : NULL,
                                    xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL,
                                    xqb[3] ? xqb[3] + b : NULL,
                                    xqb[4] ? xqb[4] + b : NULL,
                                    xqb[5] ? xqb[5] + b : NULL,
                                    xqb[6] ? xqb[6] + b : NULL,
                                    xqb[7] ? xqb[7] + b : NULL,
                                    np, gate);
        dot_q3_asym_q8k_block8_dev(ub,
                                    xqb[0] ? xqb[0] + b : NULL,
                                    xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL,
                                    xqb[3] ? xqb[3] + b : NULL,
                                    xqb[4] ? xqb[4] + b : NULL,
                                    xqb[5] ? xqb[5] + b : NULL,
                                    xqb[6] ? xqb[6] + b : NULL,
                                    xqb[7] ? xqb[7] + b : NULL,
                                    np, up);
    }
    for (uint32_t p = 0; p < np; p++) {
        gate[p] = quarter_warp_sum_f32(gate[p]);
        up[p] = quarter_warp_sum_f32(up[p]);
        if (lane == 0) {
            if (clamp > 1.0e-6f) {
                if (gate[p] > clamp) gate[p] = clamp;
                if (up[p] > clamp) up[p] = clamp;
                if (up[p] < -clamp) up[p] = -clamp;
            }
            const uint64_t off = (uint64_t)pair[p] * expert_mid_dim + row;
            if (write_gate_up) {
                gate_out[off] = gate[p];
                up_out[off] = up[p];
            }
            mid_out[off] = gate[p] * sf37_sigmoid_dev(gate[p]) * up[p] *
                           weights[(uint64_t)tok[p] * topk + slot[p]];
        }
    }
}

__global__ static void moe_down_q2_asym_q8k_sorted_kernel(
        float *down_out,
        const uint8_t *down_w,
        const sf37_cuda_block_q8_k *midq,
        const uint32_t *sorted_pairs,
        const int32_t *selected,
        uint32_t topk,
        uint32_t midq_blocks,
        uint32_t out_dim,
        uint32_t n_total_expert,
        int atomic_out) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row_lane = threadIdx.x >> 3u;
    const uint32_t row = blockIdx.x * 32u + row_lane;
    const uint32_t pair = sorted_pairs[blockIdx.y];
    if (row >= out_dim) return;

    const uint32_t tok = pair / topk;
    const uint32_t slot = pair - tok * topk;
    int32_t expert_i = selected[(uint64_t)tok * topk + slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;

    const uint64_t row_q2_bytes = (uint64_t)midq_blocks * SF37_Q2_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_q2_bytes;
    const uint8_t *wr = down_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                        (uint64_t)row * row_q2_bytes;
    const sf37_cuda_block_q8_k *xq = midq + (uint64_t)pair * midq_blocks;
    float acc = 0.0f;
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        acc += dot_q2_asym_q8k_block_dev(wr + (uint64_t)b * SF37_Q2_BLOCK_SIZE, xq + b);
    }
    acc = quarter_warp_sum_f32(acc);
    if (lane == 0) {
        if (atomic_out) atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc);
        else down_out[(uint64_t)pair * out_dim + row] = acc;
    }
}

__global__ static void moe_down_q2_asym_q8k_expert_tile8_kernel(
        float *down_out,
        const uint8_t *down_w,
        const sf37_cuda_block_q8_k *midq,
        const uint32_t *sorted_pairs,
        const uint32_t *offsets,
        const uint32_t *counts,
        const uint32_t *tile_total,
        const uint32_t *tile_experts,
        const uint32_t *tile_starts,
        uint32_t topk,
        uint32_t midq_blocks,
        uint32_t out_dim,
        int atomic_out) {
    const uint32_t tile = blockIdx.y;
    if (tile >= *tile_total) return;
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t row = blockIdx.x * 32u + (threadIdx.x >> 3u);
    const uint32_t expert = tile_experts[tile];
    const uint32_t local_start = tile_starts[tile];
    __shared__ sf37_cuda_block_q8_k sxq[8][8];
    uint32_t pair[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const sf37_cuda_block_q8_k *xqb[8] = {NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL};
    uint32_t np = 0;
    for (; np < 8u; np++) {
        const uint32_t local_pair = local_start + np;
        if (local_pair >= counts[expert]) break;
        pair[np] = sorted_pairs[offsets[expert] + local_pair];
        xqb[np] = midq + (uint64_t)pair[np] * midq_blocks;
    }
    if (midq_blocks <= 8u) {
        for (uint32_t i = threadIdx.x; i < np * midq_blocks; i += blockDim.x) {
            const uint32_t p = i / midq_blocks;
            const uint32_t b = i - p * midq_blocks;
            sxq[p][b] = xqb[p][b];
        }
        __syncthreads();
        for (uint32_t p = 0; p < np; p++) xqb[p] = sxq[p];
    }
    if (row >= out_dim) return;

    const uint64_t row_q2_bytes = (uint64_t)midq_blocks * SF37_Q2_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_q2_bytes;
    const uint8_t *wr = down_w + (uint64_t)expert * expert_bytes +
                        (uint64_t)row * row_q2_bytes;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    for (uint32_t b = lane; b < midq_blocks; b += 8u) {
        const uint8_t *wb = wr + (uint64_t)b * SF37_Q2_BLOCK_SIZE;
        dot_q2_asym_q8k_block8_dev(wb,
                                    xqb[0] ? xqb[0] + b : NULL,
                                    xqb[1] ? xqb[1] + b : NULL,
                                    xqb[2] ? xqb[2] + b : NULL,
                                    xqb[3] ? xqb[3] + b : NULL,
                                    xqb[4] ? xqb[4] + b : NULL,
                                    xqb[5] ? xqb[5] + b : NULL,
                                    xqb[6] ? xqb[6] + b : NULL,
                                    xqb[7] ? xqb[7] + b : NULL,
                                    np, acc);
    }
    for (uint32_t p = 0; p < np; p++) {
        acc[p] = quarter_warp_sum_f32(acc[p]);
        if (lane == 0) {
            if (atomic_out) {
                const uint32_t tok = pair[p] / topk;
                atomicAdd(down_out + (uint64_t)tok * out_dim + row, acc[p]);
            } else {
                down_out[(uint64_t)pair[p] * out_dim + row] = acc[p];
            }
        }
    }
}

__global__ static void moe_sum_pairs_kernel(float *out,
                                            const float *down,
                                            uint32_t out_dim,
                                            uint32_t topk,
                                            uint32_t n_tok) {
    const uint32_t row = (uint32_t)((uint64_t)blockIdx.x * blockDim.x + threadIdx.x);
    const uint32_t tok = blockIdx.y;
    if (row >= out_dim || tok >= n_tok) return;
    float acc = 0.0f;
    const uint64_t base = (uint64_t)tok * topk;
    for (uint32_t slot = 0; slot < topk; slot++) {
        acc += down[(base + slot) * out_dim + row];
    }
    out[(uint64_t)tok * out_dim + row] = acc;
}

__global__ static void router_select_sf37_kernel(int32_t *selected,
                                                 float *weights,
                                                 float *probs,
                                                 const float *logits,
                                                 const float *bias,
                                                 uint32_t n_experts,
                                                 uint32_t topk,
                                                 float scale) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    int32_t best_idx[16];
    float best_score[16];
    float best_prob[16];
    for (uint32_t j = 0; j < topk; j++) {
        best_idx[j] = -1;
        best_score[j] = -INFINITY;
        best_prob[j] = 0.0f;
    }

    for (uint32_t e = 0; e < n_experts; e++) {
        const float p = sf37_sigmoid_dev(logits[e]);
        if (probs) probs[e] = p;
        const float score = p + bias[e];
        for (uint32_t j = 0; j < topk; j++) {
            if (best_idx[j] < 0 || score > best_score[j]) {
                for (uint32_t k = topk - 1u; k > j; k--) {
                    best_idx[k] = best_idx[k - 1u];
                    best_score[k] = best_score[k - 1u];
                    best_prob[k] = best_prob[k - 1u];
                }
                best_idx[j] = (int32_t)e;
                best_score[j] = score;
                best_prob[j] = p;
                break;
            }
        }
    }

    float sum = 0.0f;
    for (uint32_t j = 0; j < topk; j++) sum += best_prob[j];
    if (sum < 1.0e-20f) sum = 1.0e-20f;
    for (uint32_t j = 0; j < topk; j++) {
        selected[j] = best_idx[j];
        weights[j] = best_prob[j] / sum * scale;
    }
}

__device__ __forceinline__ static bool router_score_better_sf37(float av,
                                                                uint32_t ai,
                                                                float bv,
                                                                uint32_t bi) {
    return av > bv || (av == bv && ai < bi);
}

__global__ static void router_select_sf37_warp_topk_kernel(int32_t *selected,
                                                           float *weights,
                                                           float *probs,
                                                           const float *logits,
                                                           const float *bias,
                                                           float scale) {
    const uint32_t lane = threadIdx.x;
    if (blockIdx.x != 0 || lane >= 32u) return;

    uint32_t local_idx[9];
    float local_prob[9];
    float local_score[9];
    uint32_t local_n = 0;
#pragma unroll
    for (uint32_t j = 0; j < 9u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < 288u) {
            const float p = sf37_sigmoid_dev(logits[e]);
            local_idx[local_n] = e;
            local_prob[local_n] = p;
            local_score[local_n] = p + bias[e];
            if (probs) probs[e] = p;
            local_n++;
        }
    }

    uint32_t sel_idx[8];
    float sel_prob[8];
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
#pragma unroll
        for (uint32_t j = 0; j < 9u; j++) {
            if (j >= local_n) continue;
            const uint32_t e = local_idx[j];
            bool used = false;
#pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                if (p >= k) continue;
                if (sel_idx[p] == e) used = true;
            }
            if (!used && router_score_better_sf37(local_score[j], e,
                                                  best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float os = __shfl_down_sync(mask, best_score, off);
            const float op = __shfl_down_sync(mask, best_prob, off);
            const uint32_t oi = __shfl_down_sync(mask, best_idx, off);
            if (router_score_better_sf37(os, oi, best_score, best_idx)) {
                best_score = os;
                best_prob = op;
                best_idx = oi;
            }
        }
        best_idx = __shfl_sync(mask, best_idx, 0);
        best_prob = __shfl_sync(mask, best_prob, 0);
        sel_idx[k] = best_idx;
        sel_prob[k] = best_prob;
        if (lane == 0) selected[k] = (int32_t)best_idx;
    }

    if (lane == 0) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) sum += sel_prob[k];
        if (sum < 1.0e-20f) sum = 1.0e-20f;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) weights[k] = sel_prob[k] / sum * scale;
    }
}

extern "C" int sf37_cuda_router_select(sf37_cuda_tensor *selected,
                                        sf37_cuda_tensor *weights,
                                        sf37_cuda_tensor *probs,
                                        const sf37_cuda_tensor *logits,
                                        const sf37_cuda_tensor *bias,
                                        uint32_t n_experts,
                                        uint32_t topk,
                                        float scale) {
    if (!selected || !weights || !logits || !bias ||
        n_experts == 0 || topk == 0 || topk > 16u) return 0;
    if (selected->bytes < (uint64_t)topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)topk * sizeof(float) ||
        logits->bytes < (uint64_t)n_experts * sizeof(float) ||
        bias->bytes < (uint64_t)n_experts * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_experts * sizeof(float))) return 0;
    if (n_experts == 288u && topk == 8u &&
        !cuda_env_present("SF37_CUDA_NO_WARP_ROUTER_SELECT", NULL)) {
        router_select_sf37_warp_topk_kernel<<<1, 32>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                probs ? (float *)probs->ptr : NULL,
                (const float *)logits->ptr,
                (const float *)bias->ptr,
                scale);
        return cuda_ok(cudaGetLastError(), "router_select warp topk launch");
    }
    router_select_sf37_kernel<<<1, 1>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            probs ? (float *)probs->ptr : NULL,
            (const float *)logits->ptr,
            (const float *)bias->ptr,
            n_experts, topk, scale);
    return cuda_ok(cudaGetLastError(), "router_select launch");
}

__global__ static void matvec_q3_asym_kernel(float *out, const uint8_t *w,
                                             uint64_t in_dim, uint64_t out_dim,
                                             const float *x) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q3_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        const uint64_t b = i / SF37_QK_K;
        const uint32_t j = (uint32_t)(i - b * SF37_QK_K);
        const uint32_t g = j >> 6u;
        const uint32_t gi = j & 63u;
        const uint8_t *blk = wr + b * SF37_Q3_BLOCK_SIZE;
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const float zp = (float)blk[8u + g];
        const uint8_t *payload = blk + 12u + g * 24u;
        const float q = (float)q3_payload_get(payload, gi);
        acc += (q - zp) * scale * x[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void matvec_q2_asym_kernel(float *out, const uint8_t *w,
                                             uint64_t in_dim, uint64_t out_dim,
                                             const float *x) {
    const uint64_t row = (uint64_t)blockIdx.x * 8u + (threadIdx.x >> 5u);
    const uint32_t lane = threadIdx.x & 31u;
    if (row >= out_dim) return;

    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q2_BLOCK_SIZE;
    const uint8_t *wr = w + row * row_bytes;
    float acc = 0.0f;
    for (uint64_t i = lane; i < in_dim; i += 32u) {
        const uint64_t b = i / SF37_QK_K;
        const uint32_t j = (uint32_t)(i - b * SF37_QK_K);
        const uint32_t g = j >> 6u;
        const uint32_t gi = j & 63u;
        const uint8_t *blk = wr + b * SF37_Q2_BLOCK_SIZE;
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const float zp = (float)blk[8u + g];
        const uint8_t *payload = blk + 12u + g * 16u;
        const float q = (float)q2_payload_get(payload, gi);
        acc += (q - zp) * scale * x[i];
    }
    acc = warp_sum_f32(acc);
    if (lane == 0) out[row] = acc;
}

__global__ static void moe_gate_up_mid_q3_asym_kernel(float *gate_out,
                                                      float *up_out,
                                                      float *mid_out,
                                                      const uint8_t *gate_w,
                                                      const uint8_t *up_w,
                                                      const float *x,
                                                      const int32_t *selected,
                                                      const float *weights,
                                                      uint32_t topk,
                                                      uint32_t in_dim,
                                                      uint32_t expert_mid_dim,
                                                      uint32_t n_total_expert,
                                                      float clamp,
                                                      int write_gate_up) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= expert_mid_dim || slot >= topk) return;
    int32_t expert_i = selected[slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;

    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q3_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)expert_mid_dim * row_bytes;
    const uint8_t *gr = gate_w + (uint64_t)(uint32_t)expert_i * expert_bytes + (uint64_t)row * row_bytes;
    const uint8_t *ur = up_w + (uint64_t)(uint32_t)expert_i * expert_bytes + (uint64_t)row * row_bytes;

    float gate = 0.0f;
    float up = 0.0f;
    for (uint64_t i = threadIdx.x; i < in_dim; i += blockDim.x) {
        const uint64_t b = i / SF37_QK_K;
        const uint32_t j = (uint32_t)(i - b * SF37_QK_K);
        const uint32_t g = j >> 6u;
        const uint32_t gi = j & 63u;
        const uint8_t *gb = gr + b * SF37_Q3_BLOCK_SIZE;
        const uint8_t *ub = ur + b * SF37_Q3_BLOCK_SIZE;
        const float gscale = __half2float(*(const __half *)(gb + g * 2u));
        const float uscale = __half2float(*(const __half *)(ub + g * 2u));
        const float gzp = (float)gb[8u + g];
        const float uzp = (float)ub[8u + g];
        const float gq = (float)q3_payload_get(gb + 12u + g * 24u, gi);
        const float uq = (float)q3_payload_get(ub + 12u + g * 24u, gi);
        const float xv = x[i];
        gate += (gq - gzp) * gscale * xv;
        up += (uq - uzp) * uscale * xv;
    }

    __shared__ float partial_gate[256];
    __shared__ float partial_up[256];
    partial_gate[threadIdx.x] = gate;
    partial_up[threadIdx.x] = up;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            partial_gate[threadIdx.x] += partial_gate[threadIdx.x + stride];
            partial_up[threadIdx.x] += partial_up[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        gate = partial_gate[0];
        up = partial_up[0];
        if (clamp > 1.0e-6f) {
            if (gate > clamp) gate = clamp;
            if (up > clamp) up = clamp;
            if (up < -clamp) up = -clamp;
        }
        const uint64_t off = (uint64_t)slot * expert_mid_dim + row;
        if (write_gate_up) {
            gate_out[off] = gate;
            up_out[off] = up;
        }
        mid_out[off] = gate * sf37_sigmoid_dev(gate) * up * weights[slot];
    }
}

__global__ static void moe_down_q2_asym_kernel(float *down_out,
                                               const uint8_t *down_w,
                                               const float *mid,
                                               const int32_t *selected,
                                               uint32_t topk,
                                               uint32_t expert_mid_dim,
                                               uint32_t out_dim,
                                               uint32_t n_total_expert) {
    const uint32_t row = blockIdx.x;
    const uint32_t slot = blockIdx.y;
    if (row >= out_dim || slot >= topk) return;
    int32_t expert_i = selected[slot];
    if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;

    const uint64_t blocks = expert_mid_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q2_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_bytes;
    const uint8_t *wr = down_w + (uint64_t)(uint32_t)expert_i * expert_bytes + (uint64_t)row * row_bytes;
    const float *xr = mid + (uint64_t)slot * expert_mid_dim;

    float acc = 0.0f;
    for (uint64_t i = threadIdx.x; i < expert_mid_dim; i += blockDim.x) {
        const uint64_t b = i / SF37_QK_K;
        const uint32_t j = (uint32_t)(i - b * SF37_QK_K);
        const uint32_t g = j >> 6u;
        const uint32_t gi = j & 63u;
        const uint8_t *blk = wr + b * SF37_Q2_BLOCK_SIZE;
        const float scale = __half2float(*(const __half *)(blk + g * 2u));
        const float zp = (float)blk[8u + g];
        const float q = (float)q2_payload_get(blk + 12u + g * 16u, gi);
        acc += (q - zp) * scale * xr[i];
    }

    __shared__ float partial[256];
    partial[threadIdx.x] = acc;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) down_out[(uint64_t)slot * out_dim + row] = partial[0];
}

__global__ static void moe_down_q2_asym_sum_kernel(float *out,
                                                   const uint8_t *down_w,
                                                   const float *mid,
                                                   const int32_t *selected,
                                                   uint32_t topk,
                                                   uint32_t expert_mid_dim,
                                                   uint32_t out_dim,
                                                   uint32_t n_total_expert) {
    const uint32_t row = blockIdx.x;
    if (row >= out_dim) return;

    const uint64_t blocks = expert_mid_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * SF37_Q2_BLOCK_SIZE;
    const uint64_t expert_bytes = (uint64_t)out_dim * row_bytes;
    __shared__ float partial[256];
    float total = 0.0f;

    for (uint32_t slot = 0; slot < topk; slot++) {
        int32_t expert_i = selected[slot];
        if (expert_i < 0 || (uint32_t)expert_i >= n_total_expert) expert_i = 0;
        const uint8_t *wr = down_w + (uint64_t)(uint32_t)expert_i * expert_bytes +
                            (uint64_t)row * row_bytes;
        const float *xr = mid + (uint64_t)slot * expert_mid_dim;

        float acc = 0.0f;
        for (uint64_t i = threadIdx.x; i < expert_mid_dim; i += blockDim.x) {
            const uint64_t b = i / SF37_QK_K;
            const uint32_t j = (uint32_t)(i - b * SF37_QK_K);
            const uint32_t g = j >> 6u;
            const uint32_t gi = j & 63u;
            const uint8_t *blk = wr + b * SF37_Q2_BLOCK_SIZE;
            const float scale = __half2float(*(const __half *)(blk + g * 2u));
            const float zp = (float)blk[8u + g];
            const float q = (float)q2_payload_get(blk + 12u + g * 16u, gi);
            acc += (q - zp) * scale * xr[i];
        }

        partial[threadIdx.x] = acc;
        __syncthreads();
        for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) total += partial[0];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[row] = total;
}

__global__ static void moe_sum_slots_kernel(float *out,
                                            const float *down,
                                            uint32_t out_dim,
                                            uint32_t topk) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= out_dim) return;
    float acc = 0.0f;
    for (uint32_t slot = 0; slot < topk; slot++) {
        acc += down[(uint64_t)slot * out_dim + row];
    }
    out[row] = acc;
}

extern "C" int sf37_cuda_routed_moe_one(sf37_cuda_tensor *out,
                                         sf37_cuda_tensor *gate,
                                         sf37_cuda_tensor *up,
                                         sf37_cuda_tensor *mid,
                                         sf37_cuda_tensor *down,
                                         const sf37_cuda_tensor *moe_gate,
                                         const sf37_cuda_tensor *moe_up,
                                         const sf37_cuda_tensor *moe_down,
                                         const sf37_cuda_tensor *selected,
                                         const sf37_cuda_tensor *weights,
                                         uint32_t n_total_expert,
                                         uint32_t topk,
                                         uint32_t in_dim,
                                         uint32_t expert_mid_dim,
                                         uint32_t out_dim,
                                         float clamp,
                                         const sf37_cuda_tensor *x) {
    if (!out || !gate || !up || !mid || !down ||
        !moe_gate || !moe_up || !moe_down || !selected || !weights || !x ||
        n_total_expert == 0 || topk == 0 ||
        in_dim % SF37_QK_K != 0 || expert_mid_dim % SF37_QK_K != 0) {
        return 0;
    }
    const uint64_t gate_row_bytes = (uint64_t)(in_dim / SF37_QK_K) * SF37_Q3_BLOCK_SIZE;
    const uint64_t down_row_bytes = (uint64_t)(expert_mid_dim / SF37_QK_K) * SF37_Q2_BLOCK_SIZE;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * expert_mid_dim * gate_row_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * out_dim * down_row_bytes;
    const uint64_t pair_mid_bytes = (uint64_t)topk * expert_mid_dim * sizeof(float);
    const uint64_t pair_down_bytes = (uint64_t)topk * out_dim * sizeof(float);
    if (out->bytes < (uint64_t)out_dim * sizeof(float) ||
        gate->bytes < pair_mid_bytes ||
        up->bytes < pair_mid_bytes ||
        mid->bytes < pair_mid_bytes ||
        down->bytes < pair_down_bytes ||
        moe_gate->bytes < gate_bytes ||
        moe_up->bytes < gate_bytes ||
        moe_down->bytes < down_bytes ||
        selected->bytes < (uint64_t)topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)topk * sizeof(float) ||
        x->bytes < (uint64_t)in_dim * sizeof(float)) {
        return 0;
    }

    cudaEvent_t prof_ev[5] = {NULL, NULL, NULL, NULL, NULL};
    const int profile_moe = cuda_moe_profile_start(prof_ev);
    const int write_gate_up = cuda_moe_write_gate_up();
    const int direct_down_sum = cuda_moe_direct_down_sum_enabled();

    if (direct_down_sum &&
        cuda_qlow_q8k_use_for(expert_mid_dim) &&
        cuda_qlow_q8k_use_for(out_dim) &&
        in_dim <= (uint32_t)UINT32_MAX &&
        expert_mid_dim <= (uint32_t)UINT32_MAX) {
        const uint32_t xq_blocks = in_dim / SF37_QK_K;
        const uint32_t midq_blocks = expert_mid_dim / SF37_QK_K;
        const uint64_t xq_count = (uint64_t)xq_blocks;
        const uint64_t midq_count = (uint64_t)topk * midq_blocks;
        const uint64_t xq_bytes = xq_count * sizeof(sf37_cuda_block_q8_k);
        const uint64_t midq_bytes = midq_count * sizeof(sf37_cuda_block_q8_k);
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                xq_bytes + midq_bytes, "routed_moe q8k activations");
        if (xq) {
            sf37_cuda_block_q8_k *midq = (sf37_cuda_block_q8_k *)((char *)xq + xq_bytes);
            q8_k_quantize_sf37_kernel<<<dim3(xq_blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q8k input quantize launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);

            dim3 q8_mgrid((expert_mid_dim + 31u) / 32u, topk, 1u);
            moe_gate_up_mid_q3_asym_q8k_kernel<<<q8_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    (const uint8_t *)moe_gate->ptr,
                    (const uint8_t *)moe_up->ptr,
                    xq,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    topk, xq_blocks, expert_mid_dim, n_total_expert,
                    clamp, write_gate_up);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q8k gate/up launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);

            q8_k_quantize_sf37_kernel<<<dim3(midq_blocks, topk, 1u), 256>>>(
                    midq, (const float *)mid->ptr, expert_mid_dim, topk);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q8k mid quantize launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);

            moe_down_q2_asym_sum_q8k_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    (const uint8_t *)moe_down->ptr,
                    midq,
                    (const int32_t *)selected->ptr,
                    topk, midq_blocks, out_dim, n_total_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe q8k direct down-sum launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) {
                (void)cudaEventRecord(prof_ev[4], 0);
                cuda_moe_profile_finish(prof_ev, 1, topk, in_dim, expert_mid_dim, out_dim);
            }
            return 1;
        }
    }

    if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);
    dim3 mgrid(expert_mid_dim, topk, 1);
    moe_gate_up_mid_q3_asym_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            (const uint8_t *)moe_gate->ptr,
            (const uint8_t *)moe_up->ptr,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            topk, in_dim, expert_mid_dim, n_total_expert, clamp,
            write_gate_up);
    if (!cuda_ok(cudaGetLastError(), "routed_moe gate/up launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);

    if (direct_down_sum) {
        moe_down_q2_asym_sum_kernel<<<out_dim, 256>>>(
                (float *)out->ptr,
                (const uint8_t *)moe_down->ptr,
                (const float *)mid->ptr,
                (const int32_t *)selected->ptr,
                topk, expert_mid_dim, out_dim, n_total_expert);
        if (!cuda_ok(cudaGetLastError(), "routed_moe direct down-sum launch")) {
            if (profile_moe) cuda_moe_profile_abort(prof_ev);
            return 0;
        }
        if (profile_moe) {
            (void)cudaEventRecord(prof_ev[4], 0);
            cuda_moe_profile_finish(prof_ev, 0, topk, in_dim, expert_mid_dim, out_dim);
        }
        return 1;
    }

    dim3 dgrid(out_dim, topk, 1);
    moe_down_q2_asym_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            (const uint8_t *)moe_down->ptr,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            topk, expert_mid_dim, out_dim, n_total_expert);
    if (!cuda_ok(cudaGetLastError(), "routed_moe down launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);

    moe_sum_slots_kernel<<<(out_dim + 255u) / 256u, 256>>>(
            (float *)out->ptr,
            (const float *)down->ptr,
            out_dim, topk);
    if (!cuda_ok(cudaGetLastError(), "routed_moe sum launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) {
        (void)cudaEventRecord(prof_ev[4], 0);
        cuda_moe_profile_finish(prof_ev, 0, topk, in_dim, expert_mid_dim, out_dim);
    }
    return 1;
}

extern "C" int sf37_cuda_routed_moe_batch_mapped(sf37_cuda_tensor *out,
                                                  sf37_cuda_tensor *gate,
                                                  sf37_cuda_tensor *up,
                                                  sf37_cuda_tensor *mid,
                                                  sf37_cuda_tensor *down,
                                                  const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t gate_offset,
                                                  uint64_t up_offset,
                                                  uint64_t down_offset,
                                                  const sf37_cuda_tensor *selected,
                                                  const sf37_cuda_tensor *weights,
                                                  uint32_t n_total_expert,
                                                  uint32_t topk,
                                                  uint32_t in_dim,
                                                  uint32_t expert_mid_dim,
                                                  uint32_t out_dim,
                                                  float clamp,
                                                  const sf37_cuda_tensor *x,
                                                  uint32_t n_tok) {
    if (!out || !gate || !up || !mid || !down || !model_map ||
        !selected || !weights || !x ||
        n_total_expert == 0 || topk == 0 || in_dim == 0 ||
        expert_mid_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float) ||
        n_tok > UINT64_MAX / topk / sizeof(float) ||
        x->bytes < (uint64_t)n_tok * in_dim * sizeof(float) ||
        out->bytes < (uint64_t)n_tok * out_dim * sizeof(float) ||
        selected->bytes < (uint64_t)n_tok * topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tok * topk * sizeof(float) ||
        gate->bytes < (uint64_t)topk * expert_mid_dim * sizeof(float) ||
        up->bytes < (uint64_t)topk * expert_mid_dim * sizeof(float) ||
        mid->bytes < (uint64_t)topk * expert_mid_dim * sizeof(float) ||
        down->bytes < (uint64_t)topk * out_dim * sizeof(float)) {
        return 0;
    }

    const uint64_t pair_count64 = (uint64_t)n_tok * topk;
    if (n_tok > 1u &&
        pair_count64 <= (uint64_t)UINT32_MAX &&
        cuda_moe_sorted_pairs_enabled() &&
        cuda_qlow_q8k_use_for(expert_mid_dim) &&
        cuda_qlow_q8k_use_for(out_dim) &&
        in_dim % SF37_QK_K == 0 &&
        expert_mid_dim % SF37_QK_K == 0 &&
        in_dim <= (uint32_t)UINT32_MAX &&
        expert_mid_dim <= (uint32_t)UINT32_MAX &&
        n_total_expert > 0u) {
        const uint32_t pair_count = (uint32_t)pair_count64;
        const uint32_t xq_blocks = in_dim / SF37_QK_K;
        const uint32_t midq_blocks = expert_mid_dim / SF37_QK_K;
        const uint64_t gate_row_bytes = (uint64_t)xq_blocks * SF37_Q3_BLOCK_SIZE;
        const uint64_t down_row_bytes = (uint64_t)midq_blocks * SF37_Q2_BLOCK_SIZE;
        if (expert_mid_dim != 0 &&
            n_total_expert <= UINT64_MAX / expert_mid_dim / gate_row_bytes &&
            out_dim != 0 &&
            n_total_expert <= UINT64_MAX / out_dim / down_row_bytes) {
            const uint64_t gate_bytes = (uint64_t)n_total_expert * expert_mid_dim * gate_row_bytes;
            const uint64_t down_bytes = (uint64_t)n_total_expert * out_dim * down_row_bytes;
            const uint8_t *gate_w = (const uint8_t *)cuda_model_range_ptr(
                    model_map, gate_offset, gate_bytes, "moe_gate");
            const uint8_t *up_w = (const uint8_t *)cuda_model_range_ptr(
                    model_map, up_offset, gate_bytes, "moe_up");
            const uint8_t *down_w = (const uint8_t *)cuda_model_range_ptr(
                    model_map, down_offset, down_bytes, "moe_down");
            if (!gate_w || !up_w || !down_w) return 0;

            const int write_gate_up = cuda_moe_write_gate_up();
            const int direct_down = cuda_moe_direct_down_sum_enabled();
            const int use_tiles = cuda_moe_expert_tiles_enabled();
            const uint32_t tile_m = 8u;
            const uint64_t tile_capacity =
                    (pair_count64 + tile_m - 1u) / tile_m + (uint64_t)n_total_expert;
            uint64_t scratch_bytes = 0;
            uint64_t xq_off = 0;
            uint64_t counts_off = 0;
            uint64_t offsets_off = 0;
            uint64_t cursors_off = 0;
            uint64_t sorted_off = 0;
            uint64_t tile_offsets_off = 0;
            uint64_t tile_total_off = 0;
            uint64_t tile_experts_off = 0;
            uint64_t tile_starts_off = 0;
            uint64_t gate_pair_off = 0;
            uint64_t up_pair_off = 0;
            uint64_t mid_pair_off = 0;
            uint64_t midq_off = 0;
            uint64_t down_pair_off = 0;
            int scratch_ok = 1;
#define SF37_SCRATCH_RESERVE(name, count, type) do { \
                if (!scratch_ok) break; \
                const uint64_t _cnt = (uint64_t)(count); \
                if (_cnt > UINT64_MAX / (uint64_t)sizeof(type)) { scratch_ok = 0; break; } \
                scratch_bytes = cuda_align_u64(scratch_bytes, 256u); \
                name##_off = scratch_bytes; \
                const uint64_t _bytes = _cnt * (uint64_t)sizeof(type); \
                if (_bytes > UINT64_MAX - scratch_bytes) { scratch_ok = 0; break; } \
                scratch_bytes += _bytes; \
            } while (0)
            SF37_SCRATCH_RESERVE(xq, (uint64_t)n_tok * xq_blocks, sf37_cuda_block_q8_k);
            SF37_SCRATCH_RESERVE(counts, n_total_expert, uint32_t);
            SF37_SCRATCH_RESERVE(offsets, (uint64_t)n_total_expert + 1u, uint32_t);
            SF37_SCRATCH_RESERVE(cursors, n_total_expert, uint32_t);
            SF37_SCRATCH_RESERVE(sorted, pair_count64, uint32_t);
            if (use_tiles) {
                SF37_SCRATCH_RESERVE(tile_offsets, (uint64_t)n_total_expert + 1u, uint32_t);
                SF37_SCRATCH_RESERVE(tile_total, 1u, uint32_t);
                SF37_SCRATCH_RESERVE(tile_experts, tile_capacity, uint32_t);
                SF37_SCRATCH_RESERVE(tile_starts, tile_capacity, uint32_t);
            }
            if (write_gate_up) {
                SF37_SCRATCH_RESERVE(gate_pair, pair_count64 * expert_mid_dim, float);
                SF37_SCRATCH_RESERVE(up_pair, pair_count64 * expert_mid_dim, float);
            }
            SF37_SCRATCH_RESERVE(mid_pair, pair_count64 * expert_mid_dim, float);
            SF37_SCRATCH_RESERVE(midq, pair_count64 * midq_blocks, sf37_cuda_block_q8_k);
            if (!direct_down) {
                SF37_SCRATCH_RESERVE(down_pair, pair_count64 * out_dim, float);
            }
#undef SF37_SCRATCH_RESERVE
            if (scratch_ok) {
                char *scratch = (char *)cuda_tmp_alloc(scratch_bytes, "routed_moe batch sorted scratch");
                if (scratch) {
                    sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)(scratch + xq_off);
                    uint32_t *counts = (uint32_t *)(scratch + counts_off);
                    uint32_t *offsets = (uint32_t *)(scratch + offsets_off);
                    uint32_t *cursors = (uint32_t *)(scratch + cursors_off);
                    uint32_t *sorted_pairs = (uint32_t *)(scratch + sorted_off);
                    uint32_t *tile_offsets = use_tiles ? (uint32_t *)(scratch + tile_offsets_off) : NULL;
                    uint32_t *tile_total = use_tiles ? (uint32_t *)(scratch + tile_total_off) : NULL;
                    uint32_t *tile_experts = use_tiles ? (uint32_t *)(scratch + tile_experts_off) : NULL;
                    uint32_t *tile_starts = use_tiles ? (uint32_t *)(scratch + tile_starts_off) : NULL;
                    float *gate_pair = write_gate_up ? (float *)(scratch + gate_pair_off) : (float *)gate->ptr;
                    float *up_pair = write_gate_up ? (float *)(scratch + up_pair_off) : (float *)up->ptr;
                    float *mid_pair = (float *)(scratch + mid_pair_off);
                    sf37_cuda_block_q8_k *midq = (sf37_cuda_block_q8_k *)(scratch + midq_off);
                    float *down_pair = direct_down ? NULL : (float *)(scratch + down_pair_off);
                    cudaEvent_t prof_ev[6] = {NULL, NULL, NULL, NULL, NULL, NULL};
                    const int profile_moe = cuda_moe_batch_profile_start(prof_ev);

                    q8_k_quantize_sf37_kernel<<<dim3(xq_blocks, n_tok, 1u), 256>>>(
                            xq, (const float *)x->ptr, in_dim, n_tok);
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch q8k input quantize launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);

                    if (!cuda_ok(cudaMemset(counts, 0, (size_t)n_total_expert * sizeof(uint32_t)),
                                 "routed_moe batch sorted counts memset")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    moe_count_sorted_pairs_sf37_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                            counts, (const int32_t *)selected->ptr, pair_count, n_total_expert);
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch sorted count launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    moe_prefix_sorted_pairs_sf37_kernel<<<1, 1>>>(
                            offsets, cursors, counts, n_total_expert);
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch sorted prefix launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    moe_scatter_sorted_pairs_sf37_kernel<<<(pair_count + 255u) / 256u, 256>>>(
                            sorted_pairs, cursors, (const int32_t *)selected->ptr,
                            pair_count, n_total_expert);
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch sorted scatter launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    if (use_tiles) {
                        moe_build_expert_tile_offsets_sf37_kernel<<<1, 1>>>(
                                tile_offsets, tile_total, counts, n_total_expert, tile_m);
                        if (!cuda_ok(cudaGetLastError(), "routed_moe batch tile offsets launch")) {
                            if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                            return 0;
                        }
                        moe_build_expert_tiles_sf37_kernel<<<(n_total_expert + 255u) / 256u, 256>>>(
                                tile_experts, tile_starts, tile_offsets, counts,
                                n_total_expert, tile_m);
                        if (!cuda_ok(cudaGetLastError(), "routed_moe batch tile build launch")) {
                            if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                            return 0;
                        }
                    }
                    if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);

                    dim3 mgrid((expert_mid_dim + 31u) / 32u,
                               use_tiles ? (unsigned)tile_capacity : pair_count,
                               1u);
                    if (use_tiles) {
                        moe_gate_up_mid_q3_asym_q8k_expert_tile8_kernel<<<mgrid, 256>>>(
                                gate_pair, up_pair, mid_pair, gate_w, up_w, xq,
                                sorted_pairs, offsets, counts, tile_total,
                                tile_experts, tile_starts,
                                (const float *)weights->ptr,
                                topk, xq_blocks, expert_mid_dim,
                                clamp, write_gate_up);
                    } else {
                        moe_gate_up_mid_q3_asym_q8k_sorted_kernel<<<mgrid, 256>>>(
                                gate_pair, up_pair, mid_pair, gate_w, up_w, xq,
                                sorted_pairs,
                                (const int32_t *)selected->ptr,
                                (const float *)weights->ptr,
                                topk, xq_blocks, expert_mid_dim, n_total_expert,
                                clamp, write_gate_up);
                    }
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch q8k gate/up launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    const uint64_t full_mid_bytes =
                            pair_count64 * (uint64_t)expert_mid_dim * sizeof(float);
                    if (mid->bytes >= full_mid_bytes) {
                        if (!cuda_ok(cudaMemcpy(mid->ptr, mid_pair, (size_t)full_mid_bytes,
                                                cudaMemcpyDeviceToDevice),
                                     "routed_moe batch mid export")) {
                            if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                            return 0;
                        }
                    }
                    if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);

                    q8_k_quantize_sf37_kernel<<<dim3(midq_blocks, pair_count, 1u), 256>>>(
                            midq, mid_pair, expert_mid_dim, pair_count);
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch q8k mid quantize launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    if (profile_moe) (void)cudaEventRecord(prof_ev[4], 0);

                    if (direct_down) {
                        if (!cuda_ok(cudaMemset(out->ptr, 0,
                                                (size_t)((uint64_t)n_tok * out_dim * sizeof(float))),
                                     "routed_moe batch output memset")) {
                            if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                            return 0;
                        }
                    }
                    dim3 dgrid((out_dim + 31u) / 32u,
                               use_tiles ? (unsigned)tile_capacity : pair_count,
                               1u);
                    if (use_tiles) {
                        moe_down_q2_asym_q8k_expert_tile8_kernel<<<dgrid, 256>>>(
                                direct_down ? (float *)out->ptr : down_pair,
                                down_w, midq, sorted_pairs, offsets, counts,
                                tile_total, tile_experts, tile_starts,
                                topk, midq_blocks, out_dim, direct_down);
                    } else {
                        moe_down_q2_asym_q8k_sorted_kernel<<<dgrid, 256>>>(
                                direct_down ? (float *)out->ptr : down_pair,
                                down_w, midq, sorted_pairs,
                                (const int32_t *)selected->ptr,
                                topk, midq_blocks, out_dim, n_total_expert,
                                direct_down);
                    }
                    if (!cuda_ok(cudaGetLastError(), "routed_moe batch q8k down launch")) {
                        if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                        return 0;
                    }
                    if (!direct_down) {
                        moe_sum_pairs_kernel<<<dim3((out_dim + 255u) / 256u, n_tok, 1u), 256>>>(
                                (float *)out->ptr, down_pair, out_dim, topk, n_tok);
                        if (!cuda_ok(cudaGetLastError(), "routed_moe batch q8k down sum launch")) {
                            if (profile_moe) cuda_moe_batch_profile_abort(prof_ev);
                            return 0;
                        }
                    }
                    if (profile_moe) {
                        (void)cudaEventRecord(prof_ev[5], 0);
                        cuda_moe_batch_profile_finish(prof_ev, n_tok, topk,
                                                      in_dim, expert_mid_dim, out_dim,
                                                      use_tiles, direct_down);
                    }
                    return 1;
                }
            }
        }
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const uint64_t mid_pair_bytes = (uint64_t)topk * expert_mid_dim * sizeof(float);
        sf37_cuda_tensor out_view = {
            (char *)out->ptr + (uint64_t)t * out_dim * sizeof(float),
            (uint64_t)out_dim * sizeof(float)
        };
        sf37_cuda_tensor x_view = {
            (char *)x->ptr + (uint64_t)t * in_dim * sizeof(float),
            (uint64_t)in_dim * sizeof(float)
        };
        sf37_cuda_tensor selected_view = {
            (char *)selected->ptr + (uint64_t)t * topk * sizeof(int32_t),
            (uint64_t)topk * sizeof(int32_t)
        };
        sf37_cuda_tensor weights_view = {
            (char *)weights->ptr + (uint64_t)t * topk * sizeof(float),
            (uint64_t)topk * sizeof(float)
        };
        sf37_cuda_tensor mid_view = {
            mid->bytes >= (uint64_t)n_tok * mid_pair_bytes ?
                (char *)mid->ptr + (uint64_t)t * mid_pair_bytes :
                mid->ptr,
            mid_pair_bytes
        };
        if (!sf37_cuda_routed_moe_one_mapped(&out_view,
                                             gate,
                                             up,
                                             &mid_view,
                                             down,
                                             model_map,
                                             model_size,
                                             gate_offset,
                                             up_offset,
                                             down_offset,
                                             &selected_view,
                                             &weights_view,
                                             n_total_expert,
                                             topk,
                                             in_dim,
                                             expert_mid_dim,
                                             out_dim,
                                             clamp,
                                             &x_view)) {
            return 0;
        }
    }
    return 1;
}

extern "C" int sf37_cuda_matvec_q3_asym(sf37_cuda_tensor *out,
                                         const sf37_cuda_tensor *weights,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const sf37_cuda_tensor *x) {
    if (!out || !weights || !x || in_dim % SF37_QK_K != 0) return 0;
    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q3_BLOCK_SIZE;
    if (out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights->bytes < weight_bytes) return 0;
    if (cuda_qlow_q8k_use_for(out_dim) && in_dim <= (uint64_t)UINT32_MAX &&
        blocks <= (uint64_t)UINT32_MAX) {
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                blocks * sizeof(sf37_cuda_block_q8_k), "q3 asym q8k activation");
        if (xq) {
            q8_k_quantize_sf37_kernel<<<dim3((unsigned)blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, (uint32_t)in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "q3 asym q8k quantize launch")) return 0;
            matvec_q3_asym_q8k_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
                    (float *)out->ptr, (const uint8_t *)weights->ptr,
                    xq, in_dim, out_dim, blocks);
            return cuda_ok(cudaGetLastError(), "q3 asym q8k matvec launch");
        }
    }
    matvec_q3_asym_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, (const uint8_t *)weights->ptr,
            in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "q3 asym matvec launch");
}

extern "C" int sf37_cuda_matvec_q2_asym(sf37_cuda_tensor *out,
                                         const sf37_cuda_tensor *weights,
                                         uint64_t in_dim, uint64_t out_dim,
                                         const sf37_cuda_tensor *x) {
    if (!out || !weights || !x || in_dim % SF37_QK_K != 0) return 0;
    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q2_BLOCK_SIZE;
    if (out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float) ||
        weights->bytes < weight_bytes) return 0;
    if (cuda_qlow_q8k_use_for(out_dim) && in_dim <= (uint64_t)UINT32_MAX &&
        blocks <= (uint64_t)UINT32_MAX) {
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                blocks * sizeof(sf37_cuda_block_q8_k), "q2 asym q8k activation");
        if (xq) {
            q8_k_quantize_sf37_kernel<<<dim3((unsigned)blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, (uint32_t)in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "q2 asym q8k quantize launch")) return 0;
            matvec_q2_asym_q8k_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
                    (float *)out->ptr, (const uint8_t *)weights->ptr,
                    xq, in_dim, out_dim, blocks);
            return cuda_ok(cudaGetLastError(), "q2 asym q8k matvec launch");
        }
    }
    matvec_q2_asym_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, (const uint8_t *)weights->ptr,
            in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "q2 asym matvec launch");
}

static int mapped_range_ok(uint64_t model_size, uint64_t offset, uint64_t bytes) {
    return offset <= model_size && bytes <= model_size - offset;
}

static uint64_t cuda_parse_mib_env2(const char *sf37_name,
                                    const char *ds4_name,
                                    int *present) {
    const char *env = cuda_env_value(sf37_name, ds4_name);
    if (present) *present = 0;
    if (!env || !env[0]) return 0;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env || *end != '\0') return 0;
    if (present) *present = 1;
    if (v > UINT64_MAX / 1048576ull) return UINT64_MAX;
    return (uint64_t)v * 1048576ull;
}

static uint64_t cuda_q8_f16_cache_limit_bytes(void) {
    int present = 0;
    const uint64_t limit = cuda_parse_mib_env2("SF37_CUDA_Q8_F16_CACHE_MB",
                                               "DS4_CUDA_Q8_F16_CACHE_MB",
                                               &present);
    return present ? limit : UINT64_MAX;
}

static uint64_t cuda_q8_f16_cache_reserve_bytes(uint64_t total_bytes) {
    int present = 0;
    const uint64_t reserve = cuda_parse_mib_env2("SF37_CUDA_Q8_F16_CACHE_RESERVE_MB",
                                                 "DS4_CUDA_Q8_F16_CACHE_RESERVE_MB",
                                                 &present);
    if (present) return reserve;
    const uint64_t min_reserve = 4096ull * 1048576ull;
    const uint64_t pct_reserve = total_bytes / 20u;
    return pct_reserve > min_reserve ? pct_reserve : min_reserve;
}

static void cuda_q8_f16_cache_budget_notice(const char *reason,
                                            uint64_t request_bytes,
                                            uint64_t free_bytes,
                                            uint64_t total_bytes,
                                            uint64_t reserve_bytes,
                                            uint64_t limit_bytes) {
    if (g_q8_f16_budget_notice_printed &&
        !cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        return;
    }
    g_q8_f16_budget_notice_printed = 1;
    fprintf(stderr,
            "sf37: CUDA q8 fp16 cache %s; using q8 kernels "
            "(request=%.2f MiB cached=%.2f GiB limit=%.2f GiB free=%.2f GiB reserve=%.2f GiB total=%.2f GiB)\n",
            reason ? reason : "skipped",
            (double)request_bytes / 1048576.0,
            (double)g_q8_f16_bytes / 1073741824.0,
            limit_bytes == UINT64_MAX ? -1.0 : (double)limit_bytes / 1073741824.0,
            (double)free_bytes / 1073741824.0,
            (double)reserve_bytes / 1073741824.0,
            (double)total_bytes / 1073741824.0);
}

static int cuda_q8_f16_cache_has_budget(uint64_t request_bytes, const char *label) {
    (void)label;
    uint64_t limit = cuda_q8_f16_cache_limit_bytes();
    if (limit == 0) return 0;
    if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
        cuda_q8_f16_cache_budget_notice("limit reached", request_bytes, 0, 0, 0, limit);
        return 0;
    }
    size_t free_b = 0;
    size_t total_b = 0;
    cudaError_t err = cudaMemGetInfo(&free_b, &total_b);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA q8 fp16 cache memory query failed: %s; using q8 kernels\n",
                cudaGetErrorString(err));
        (void)cudaGetLastError();
        return 0;
    }
    const uint64_t free_bytes = (uint64_t)free_b;
    const uint64_t total_bytes = (uint64_t)total_b;
    if (limit == UINT64_MAX &&
        total_bytes <= 128ull * 1073741824ull &&
        (g_model_range_bytes >= 64ull * 1073741824ull ||
         g_model_registered_size >= 64ull * 1073741824ull)) {
        if (g_model_registered_size >= 88ull * 1073741824ull ||
            g_model_range_bytes >= 88ull * 1073741824ull) {
            limit = 16ull * 1073741824ull;
        } else {
            limit = 12ull * 1073741824ull;
        }
        if (g_q8_f16_bytes > limit || request_bytes > limit - g_q8_f16_bytes) {
            cuda_q8_f16_cache_budget_notice("limit reached", request_bytes,
                                            free_bytes, total_bytes, 0, limit);
            return 0;
        }
    }
    const uint64_t reserve_bytes = cuda_q8_f16_cache_reserve_bytes(total_bytes);
    if (request_bytes > free_bytes || free_bytes - request_bytes < reserve_bytes) {
        cuda_q8_f16_cache_budget_notice("budget exhausted", request_bytes,
                                        free_bytes, total_bytes,
                                        reserve_bytes, limit);
        return 0;
    }
    return 1;
}

static void cuda_q8_f16_cache_disable_after_failure(const char *what,
                                                    uint64_t request_bytes) {
    if (!g_q8_f16_disabled_after_oom) {
        fprintf(stderr,
                "sf37: CUDA q8 fp16 cache disabled after %s "
                "(request=%.2f MiB cached=%.2f GiB); using q8 kernels\n",
                what ? what : "allocation failure",
                (double)request_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0);
    }
    g_q8_f16_disabled_after_oom = 1;
    if (!g_q8_f16_ranges.empty()) {
        (void)cudaDeviceSynchronize();
        for (const cuda_q8_f16_range &r : g_q8_f16_ranges) {
            if (r.device_ptr) (void)cudaFree(r.device_ptr);
        }
        g_q8_f16_ranges.clear();
        g_q8_f16_by_offset.clear();
        g_q8_f16_bytes = 0;
    }
    (void)cudaGetLastError();
}

static int cuda_q8_f16_cache_allowed(const char *label,
                                     uint64_t in_dim,
                                     uint64_t out_dim) {
    (void)label;
    (void)in_dim;
    (void)out_dim;
    if (g_q8_f16_disabled_after_oom) return 0;
    if (cuda_env_present("SF37_CUDA_NO_Q8_F16_CACHE", "DS4_CUDA_NO_Q8_F16_CACHE")) return 0;
    if (cuda_q8_f16_cache_limit_bytes() == 0) return 0;
    return 1;
}

static int cuda_q8_f32_cache_allowed(const char *label,
                                     uint64_t in_dim,
                                     uint64_t out_dim) {
    (void)label;
    if (cuda_env_present("SF37_CUDA_NO_Q8_F32_CACHE", "DS4_CUDA_NO_Q8_F32_CACHE")) return 0;
    if (cuda_env_present("SF37_CUDA_Q8_F32_ALL", "DS4_CUDA_Q8_F32_ALL")) return 1;
    return cuda_env_present("SF37_CUDA_Q8_F32_LARGE", "DS4_CUDA_Q8_F32_LARGE") &&
           in_dim * out_dim >= 16ull * 1024ull * 1024ull;
}

static const __half *cuda_q8_f16_ptr(const void *model_map,
                                     uint64_t offset,
                                     uint64_t weight_bytes,
                                     uint64_t in_dim,
                                     uint64_t out_dim,
                                     const char *label) {
    auto exact = g_q8_f16_by_offset.find(offset);
    if (exact != g_q8_f16_by_offset.end()) {
        const cuda_q8_f16_range &r = g_q8_f16_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            cuda_residency_note_q8_expanded(1);
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f16_cache_allowed(label, in_dim, out_dim)) return NULL;
    cuda_residency_note_q8_expanded(0);
    const uint8_t *q8 = (const uint8_t *)cuda_model_range_ptr(model_map, offset,
                                                              weight_bytes,
                                                              label ? label : "q8_0");
    if (!q8) return NULL;
    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(__half)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(__half);
    if (!cuda_q8_f16_cache_has_budget(out_bytes, label)) return NULL;
    __half *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA q8 fp16 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        cuda_q8_f16_cache_disable_after_failure("allocation failure", out_bytes);
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f16_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            dev, q8, in_dim, out_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp16 dequant launch")) {
        (void)cudaFree(dev);
        cuda_q8_f16_cache_disable_after_failure("dequant launch failure", out_bytes);
        return NULL;
    }
    g_q8_f16_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f16_by_offset[offset] = g_q8_f16_ranges.size() - 1u;
    g_q8_f16_bytes += out_bytes;
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "sf37: CUDA cached q8 fp16 %.2f MiB (total %.2f GiB) for %s\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f16_bytes / 1073741824.0,
                label ? label : "q8_0");
    }
    return dev;
}

static float *cuda_q8_f32_ptr(const void *model_map,
                              uint64_t offset,
                              uint64_t weight_bytes,
                              uint64_t in_dim,
                              uint64_t out_dim,
                              const char *label) {
    auto exact = g_q8_f32_by_offset.find(offset);
    if (exact != g_q8_f32_by_offset.end()) {
        const cuda_q8_f32_range &r = g_q8_f32_ranges[exact->second];
        if (r.host_base == model_map && r.weight_bytes == weight_bytes &&
            r.in_dim == in_dim && r.out_dim == out_dim) {
            cuda_residency_note_q8_expanded(1);
            return r.device_ptr;
        }
    }
    if (!cuda_q8_f32_cache_allowed(label, in_dim, out_dim)) return NULL;
    cuda_residency_note_q8_expanded(0);
    const uint8_t *q8 = (const uint8_t *)cuda_model_range_ptr(model_map, offset,
                                                              weight_bytes,
                                                              label ? label : "q8_0");
    if (!q8) return NULL;
    if (in_dim != 0 && out_dim > UINT64_MAX / in_dim / sizeof(float)) return NULL;
    const uint64_t out_bytes = in_dim * out_dim * sizeof(float);
    float *dev = NULL;
    cudaError_t err = cudaMalloc(&dev, (size_t)out_bytes);
    if (err != cudaSuccess) {
        fprintf(stderr, "sf37: CUDA q8 fp32 cache alloc failed (%.2f MiB): %s\n",
                (double)out_bytes / 1048576.0, cudaGetErrorString(err));
        (void)cudaGetLastError();
        return NULL;
    }
    const uint64_t blocks = (in_dim + 31u) / 32u;
    const uint64_t n = in_dim * out_dim;
    dequant_q8_0_to_f32_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            dev, q8, in_dim, out_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8 fp32 dequant launch")) {
        (void)cudaFree(dev);
        return NULL;
    }
    g_q8_f32_ranges.push_back({model_map, offset, weight_bytes, in_dim, out_dim, dev});
    g_q8_f32_by_offset[offset] = g_q8_f32_ranges.size() - 1u;
    g_q8_f32_bytes += out_bytes;
    if (cuda_env_present("SF37_CUDA_WEIGHT_CACHE_VERBOSE", "DS4_CUDA_WEIGHT_CACHE_VERBOSE")) {
        fprintf(stderr, "sf37: CUDA cached q8 fp32 %.2f MiB (total %.2f GiB) for %s\n",
                (double)out_bytes / 1048576.0,
                (double)g_q8_f32_bytes / 1073741824.0,
                label ? label : "q8_0");
    }
    return dev;
}

extern "C" int sf37_cuda_cache_q8_f16_range(const void *model_map,
                                             uint64_t model_size,
                                             uint64_t offset,
                                             uint64_t bytes,
                                             uint64_t in_dim,
                                             uint64_t out_dim,
                                             const char *label) {
    if (!model_map || bytes == 0) return 1;
    if (!mapped_range_ok(model_size, offset, bytes)) return 0;
    const char *cache_label = label ? label : "q8_0";
    if (cuda_env_present("SF37_CUDA_Q8_F32_PRELOAD", "DS4_CUDA_Q8_F32_PRELOAD") &&
        cuda_q8_f32_cache_allowed(cache_label, in_dim, out_dim)) {
        (void)cuda_q8_f32_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
        return 1;
    }
    if (!cuda_q8_f16_cache_allowed(cache_label, in_dim, out_dim)) return 1;
    (void)cuda_q8_f16_ptr(model_map, offset, bytes, in_dim, out_dim, cache_label);
    return 1;
}

extern "C" int sf37_cuda_rms_norm_weight1_bf16_mapped(sf37_cuda_tensor *out,
                                                       const sf37_cuda_tensor *x,
                                                       const void *model_map,
                                                       uint64_t model_size,
                                                       uint64_t weight_offset,
                                                       uint32_t n, float eps) {
    if (!out || !x || !model_map) return 0;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < (uint64_t)n * sizeof(float) ||
        x->bytes < (uint64_t)n * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes, "rms_weight_bf16");
    if (!w) return 0;
    rms_norm_weight1_bf16_kernel<<<1, 256>>>((float *)out->ptr,
                                             (const float *)x->ptr,
                                             w, n, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight1_bf16 mapped launch");
}

extern "C" int sf37_cuda_rms_norm_weight1_bf16_batch_mapped(sf37_cuda_tensor *out,
                                                             const sf37_cuda_tensor *x,
                                                             const void *model_map,
                                                             uint64_t model_size,
                                                             uint64_t weight_offset,
                                                             uint32_t n,
                                                             uint32_t n_tok,
                                                             float eps) {
    if (!out || !x || !model_map || n == 0 || n_tok == 0) return 0;
    const uint64_t weight_bytes = (uint64_t)n * sizeof(uint16_t);
    const uint64_t elem_count = (uint64_t)n * n_tok;
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        elem_count > UINT64_MAX / sizeof(float) ||
        out->bytes < elem_count * sizeof(float) ||
        x->bytes < elem_count * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes,
                                                               "rms_weight_bf16_batch");
    if (!w) return 0;
    rms_norm_weight1_bf16_batch_kernel<<<n_tok, 256>>>((float *)out->ptr,
                                                       (const float *)x->ptr,
                                                       w, n, n_tok, eps);
    return cuda_ok(cudaGetLastError(), "rms_norm_weight1_bf16 batch mapped launch");
}

extern "C" int sf37_cuda_head_rms_norm_weight1_bf16_mapped(sf37_cuda_tensor *x,
                                                            const void *model_map,
                                                            uint64_t model_size,
                                                            uint64_t weight_offset,
                                                            uint32_t n_head,
                                                            uint32_t head_dim,
                                                            float eps) {
    if (!x || !model_map) return 0;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        x->bytes < (uint64_t)n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes, "head_rms_bf16");
    if (!w) return 0;
    head_rms_norm_weight1_bf16_kernel<<<n_head, 256>>>((float *)x->ptr, w,
                                                       n_head, head_dim, eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_weight1_bf16 mapped launch");
}

__global__ static void head_rms_norm_weight1_bf16_batch_kernel(float *x,
                                                               const uint16_t *weight,
                                                               uint32_t n_tok,
                                                               uint32_t n_head,
                                                               uint32_t head_dim,
                                                               float eps) {
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    if (h >= n_head || t >= n_tok) return;
    float *head = x + ((uint64_t)t * n_head + h) * head_dim;
    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        const float v = head[i];
        sum += v * v;
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)head_dim + eps);
    for (uint32_t i = threadIdx.x; i < head_dim; i += blockDim.x) {
        head[i] = head[i] * scale * (sf37_bf16_to_f32_dev(weight[i]) + 1.0f);
    }
}

extern "C" int sf37_cuda_head_rms_norm_weight1_bf16_batch_mapped(sf37_cuda_tensor *x,
                                                                  const void *model_map,
                                                                  uint64_t model_size,
                                                                  uint64_t weight_offset,
                                                                  uint32_t n_tok,
                                                                  uint32_t n_head,
                                                                  uint32_t head_dim,
                                                                  float eps) {
    if (!x || !model_map || n_tok == 0 || n_head == 0 || head_dim == 0) return 0;
    const uint64_t weight_bytes = (uint64_t)head_dim * sizeof(uint16_t);
    if (n_tok > UINT64_MAX / n_head / head_dim / sizeof(float) ||
        !mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes,
                                                               "head_rms_bf16_batch");
    if (!w) return 0;
    dim3 grid(n_head, n_tok, 1u);
    head_rms_norm_weight1_bf16_batch_kernel<<<grid, 256>>>((float *)x->ptr,
                                                           w,
                                                           n_tok,
                                                           n_head,
                                                           head_dim,
                                                           eps);
    return cuda_ok(cudaGetLastError(), "head_rms_norm_weight1_bf16 batch mapped launch");
}

__global__ static void rope_split_half_batch_kernel(float *x,
                                                    uint32_t n_tok,
                                                    uint32_t n_head,
                                                    uint32_t head_dim,
                                                    uint32_t rotary_dim,
                                                    double theta,
                                                    int llama3,
                                                    uint32_t pos0) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t pairs_per_tok = (uint64_t)n_head * (rotary_dim / 2u);
    const uint64_t total = (uint64_t)n_tok * pairs_per_tok;
    if (gid >= total) return;
    const uint32_t t = (uint32_t)(gid / pairs_per_tok);
    const uint64_t local = gid - (uint64_t)t * pairs_per_tok;
    const uint32_t pair = (uint32_t)(local % (rotary_dim / 2u));
    const uint32_t h = (uint32_t)(local / (rotary_dim / 2u));
    float *head = x + ((uint64_t)t * n_head + h) * head_dim;
    const uint32_t half = rotary_dim / 2u;
    const double inv = sf37_inv_freq_dev(pair, rotary_dim, theta, llama3);
    const float c = cosf((float)((double)(pos0 + t) * inv));
    const float s = sinf((float)((double)(pos0 + t) * inv));
    const float a = head[pair];
    const float b = head[pair + half];
    head[pair] = a * c - b * s;
    head[pair + half] = b * c + a * s;
}

extern "C" int sf37_cuda_rope_split_half_batch(sf37_cuda_tensor *x,
                                                uint32_t n_tok,
                                                uint32_t n_head,
                                                uint32_t head_dim,
                                                uint32_t rotary_dim,
                                                double theta,
                                                int llama3,
                                                uint32_t pos0) {
    if (!x || n_tok == 0 || n_head == 0 ||
        rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1u)) {
        return 0;
    }
    if (n_tok > UINT64_MAX / n_head / head_dim / sizeof(float) ||
        x->bytes < (uint64_t)n_tok * n_head * head_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t pairs = (uint64_t)n_tok * n_head * (rotary_dim / 2u);
    rope_split_half_batch_kernel<<<(unsigned)((pairs + 255u) / 256u), 256>>>(
            (float *)x->ptr,
            n_tok,
            n_head,
            head_dim,
            rotary_dim,
            theta,
            llama3,
            pos0);
    return cuda_ok(cudaGetLastError(), "rope_split_half batch launch");
}

__global__ static void store_kv_cache_batch_kernel(float *k_cache,
                                                   float *v_cache,
                                                   const float *k,
                                                   const float *v,
                                                   uint32_t pos0,
                                                   uint32_t n_tok,
                                                   uint32_t cache_cap,
                                                   uint32_t kv_dim) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n = (uint64_t)n_tok * kv_dim;
    if (gid >= n) return;
    const uint32_t t = (uint32_t)(gid / kv_dim);
    const uint32_t d = (uint32_t)(gid - (uint64_t)t * kv_dim);
    const uint32_t slot = (pos0 + t) % cache_cap;
    const uint64_t dst = (uint64_t)slot * kv_dim + d;
    k_cache[dst] = k[gid];
    v_cache[dst] = v[gid];
}

extern "C" int sf37_cuda_store_kv_cache_batch(sf37_cuda_tensor *k_cache,
                                               sf37_cuda_tensor *v_cache,
                                               const sf37_cuda_tensor *k,
                                               const sf37_cuda_tensor *v,
                                               uint32_t pos0,
                                               uint32_t n_tok,
                                               uint32_t cache_cap,
                                               uint32_t kv_dim) {
    if (!k_cache || !v_cache || !k || !v || n_tok == 0 || cache_cap == 0 || kv_dim == 0) {
        return 0;
    }
    if (n_tok > UINT64_MAX / kv_dim / sizeof(float) ||
        k->bytes < (uint64_t)n_tok * kv_dim * sizeof(float) ||
        v->bytes < (uint64_t)n_tok * kv_dim * sizeof(float) ||
        k_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float) ||
        v_cache->bytes < (uint64_t)cache_cap * kv_dim * sizeof(float)) {
        return 0;
    }
    const uint64_t n = (uint64_t)n_tok * kv_dim;
    store_kv_cache_batch_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            (float *)k_cache->ptr,
            (float *)v_cache->ptr,
            (const float *)k->ptr,
            (const float *)v->ptr,
            pos0,
            n_tok,
            cache_cap,
            kv_dim);
    return cuda_ok(cudaGetLastError(), "store kv cache batch launch");
}

extern "C" int sf37_cuda_matvec_q8_0_mapped(sf37_cuda_tensor *out,
                                             const void *model_map,
                                             uint64_t model_size,
                                             uint64_t weight_offset,
                                             uint64_t in_dim,
                                             uint64_t out_dim,
                                             const sf37_cuda_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks != 0 && out_dim > UINT64_MAX / blocks / SF37_Q8_BLOCK_SIZE) return 0;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q8_BLOCK_SIZE;
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                             weight_bytes, "q8_0");
    if (!w) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 mapped prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale,
                                                       (const float *)x->ptr,
                                                       in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 mapped quantize launch")) return 0;
    const int use_dp4a = cuda_q8_use_dp4a();
    matvec_q8_0_preq_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, w, xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "q8_0 mapped matvec launch");
}

extern "C" int sf37_cuda_matmul_q8_0_mapped(sf37_cuda_tensor *out,
                                             const void *model_map,
                                             uint64_t model_size,
                                             uint64_t weight_offset,
                                             uint64_t in_dim,
                                             uint64_t out_dim,
                                             const sf37_cuda_tensor *x,
                                             uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks != 0 && out_dim > UINT64_MAX / blocks / SF37_Q8_BLOCK_SIZE) return 0;
    if (n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float)) return 0;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q8_BLOCK_SIZE;
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        x->bytes < n_tok * in_dim * sizeof(float) ||
        out->bytes < n_tok * out_dim * sizeof(float)) {
        return 0;
    }

    if (g_cublas_ready && n_tok > 1 &&
        in_dim <= (uint64_t)INT_MAX &&
        out_dim <= (uint64_t)INT_MAX &&
        n_tok <= (uint64_t)INT_MAX) {
        const float *w_f32 = cuda_q8_f32_ptr(model_map, weight_offset, weight_bytes,
                                             in_dim, out_dim, "q8_0");
        if (w_f32) {
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasSgemm(g_cublas,
                                            CUBLAS_OP_T,
                                            CUBLAS_OP_N,
                                            (int)out_dim,
                                            (int)n_tok,
                                            (int)in_dim,
                                            &alpha,
                                            w_f32,
                                            (int)in_dim,
                                            (const float *)x->ptr,
                                            (int)in_dim,
                                            &beta,
                                            (float *)out->ptr,
                                            (int)out_dim);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "sf37: cuBLAS q8 fp32 matmul failed: status %d; falling back to q8 kernel\n",
                    (int)st);
        }

        const __half *w_f16 = cuda_q8_f16_ptr(model_map, weight_offset, weight_bytes,
                                              in_dim, out_dim, "q8_0");
        if (w_f16) {
            const uint64_t xh_count = n_tok * in_dim;
            __half *xh = (__half *)cuda_tmp_alloc(xh_count * sizeof(__half),
                                                  "q8 f16 gemm activations");
            if (!xh) return 0;
            f32_to_f16_kernel<<<(unsigned)((xh_count + 255u) / 256u), 256>>>(
                    xh, (const float *)x->ptr, xh_count);
            if (!cuda_ok(cudaGetLastError(), "q8 f16 activation convert launch")) return 0;
            const float alpha = 1.0f;
            const float beta = 0.0f;
            cublasStatus_t st = cublasGemmEx(g_cublas,
                                             CUBLAS_OP_T,
                                             CUBLAS_OP_N,
                                             (int)out_dim,
                                             (int)n_tok,
                                             (int)in_dim,
                                             &alpha,
                                             w_f16,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             xh,
                                             CUDA_R_16F,
                                             (int)in_dim,
                                             &beta,
                                             out->ptr,
                                             CUDA_R_32F,
                                             (int)out_dim,
                                             CUBLAS_COMPUTE_32F,
                                             CUBLAS_GEMM_DEFAULT);
            if (st == CUBLAS_STATUS_SUCCESS) return 1;
            fprintf(stderr, "sf37: cuBLAS q8 fp16 matmul failed: status %d; falling back to q8 kernel\n",
                    (int)st);
            cuda_q8_f16_cache_disable_after_failure("cuBLAS f16 matmul failure",
                                                    in_dim * out_dim * sizeof(__half));
        }
    }

    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                             weight_bytes, "q8_0");
    if (!w) return 0;
    const uint64_t xq_bytes = n_tok * blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + n_tok * blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 mapped matmul prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    dim3 qgrid((unsigned)blocks, (unsigned)n_tok, 1u);
    quantize_q8_0_f32_kernel<<<qgrid, 32>>>(xq, xscale,
                                            (const float *)x->ptr,
                                            in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 mapped matmul quantize launch")) return 0;
    const int use_dp4a = cuda_q8_use_dp4a();
    if (n_tok == 1) {
        matvec_q8_0_preq_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
                (float *)out->ptr, w, xq, xscale, in_dim, out_dim, blocks, use_dp4a);
        return cuda_ok(cudaGetLastError(), "q8_0 mapped matmul single launch");
    }
    dim3 grid((unsigned)((out_dim + 7u) / 8u), (unsigned)n_tok, 1u);
    matmul_q8_0_preq_batch_warp8_kernel<<<grid, 256>>>(
            (float *)out->ptr, w, xq, xscale,
            in_dim, out_dim, n_tok, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "q8_0 mapped matmul batch launch");
}

extern "C" int sf37_cuda_matmul_q8_0_pair_mapped(sf37_cuda_tensor *out0,
                                                  sf37_cuda_tensor *out1,
                                                  const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight0_offset,
                                                  uint64_t weight1_offset,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  const sf37_cuda_tensor *x,
                                                  uint64_t n_tok) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) {
        return 0;
    }
    if (n_tok == 1 && !cuda_env_present("SF37_CUDA_NO_Q8_PAIR_FUSED", NULL)) {
        return sf37_cuda_matvec_q8_0_pair_mapped(out0,
                                                 out1,
                                                 model_map,
                                                 model_size,
                                                 weight0_offset,
                                                 weight1_offset,
                                                 in_dim,
                                                 out_dim,
                                                 x);
    }
    if (!sf37_cuda_matmul_q8_0_mapped(out0,
                                      model_map,
                                      model_size,
                                      weight0_offset,
                                      in_dim,
                                      out_dim,
                                      x,
                                      n_tok)) {
        return 0;
    }
    return sf37_cuda_matmul_q8_0_mapped(out1,
                                        model_map,
                                        model_size,
                                        weight1_offset,
                                        in_dim,
                                        out_dim,
                                        x,
                                        n_tok);
}

extern "C" int sf37_cuda_matvec_q8_0_pair_mapped(sf37_cuda_tensor *out0,
                                                  sf37_cuda_tensor *out1,
                                                  const void *model_map,
                                                  uint64_t model_size,
                                                  uint64_t weight0_offset,
                                                  uint64_t weight1_offset,
                                                  uint64_t in_dim,
                                                  uint64_t out_dim,
                                                  const sf37_cuda_tensor *x) {
    if (!out0 || !out1 || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    const uint64_t blocks = (in_dim + 31u) / 32u;
    if (blocks != 0 && out_dim > UINT64_MAX / blocks / SF37_Q8_BLOCK_SIZE) return 0;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q8_BLOCK_SIZE;
    if (!mapped_range_ok(model_size, weight0_offset, weight_bytes) ||
        !mapped_range_ok(model_size, weight1_offset, weight_bytes) ||
        out0->bytes < out_dim * sizeof(float) ||
        out1->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint8_t *w0 = (const uint8_t *)cuda_model_range_ptr(model_map, weight0_offset,
                                                              weight_bytes, "q8_0_pair0");
    const uint8_t *w1 = (const uint8_t *)cuda_model_range_ptr(model_map, weight1_offset,
                                                              weight_bytes, "q8_0_pair1");
    if (!w0 || !w1) return 0;

    const uint64_t xq_bytes = blocks * 32u;
    const uint64_t scale_offset = (xq_bytes + 15u) & ~15ull;
    const uint64_t tmp_bytes = scale_offset + blocks * sizeof(float);
    void *tmp = cuda_tmp_alloc(tmp_bytes, "q8_0 pair mapped prequant");
    if (!tmp) return 0;
    int8_t *xq = (int8_t *)tmp;
    float *xscale = (float *)((char *)tmp + scale_offset);
    quantize_q8_0_f32_kernel<<<(unsigned)blocks, 32>>>(xq, xscale,
                                                       (const float *)x->ptr,
                                                       in_dim, blocks);
    if (!cuda_ok(cudaGetLastError(), "q8_0 pair mapped quantize launch")) return 0;
    const unsigned grid = (unsigned)((out_dim + 7u) / 8u);
    const int use_dp4a = cuda_q8_use_dp4a();
    if (!cuda_env_present("SF37_CUDA_NO_Q8_PAIR_FUSED", NULL)) {
        matvec_q8_0_pair_preq_kernel<<<grid, 256>>>((float *)out0->ptr,
                                                   (float *)out1->ptr,
                                                   w0, w1,
                                                   xq, xscale,
                                                   in_dim, out_dim, blocks,
                                                   use_dp4a);
        return cuda_ok(cudaGetLastError(), "q8_0 pair mapped fused matvec launch");
    }
    matvec_q8_0_preq_kernel<<<grid, 256>>>((float *)out0->ptr, w0,
                                           xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    if (!cuda_ok(cudaGetLastError(), "q8_0 pair mapped matvec0 launch")) return 0;
    matvec_q8_0_preq_kernel<<<grid, 256>>>((float *)out1->ptr, w1,
                                           xq, xscale, in_dim, out_dim, blocks, use_dp4a);
    return cuda_ok(cudaGetLastError(), "q8_0 pair mapped matvec1 launch");
}

extern "C" int sf37_cuda_matvec_bf16_mapped(sf37_cuda_tensor *out,
                                             const void *model_map,
                                             uint64_t model_size,
                                             uint64_t weight_offset,
                                             uint64_t in_dim,
                                             uint64_t out_dim,
                                             const sf37_cuda_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim / sizeof(uint16_t)) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes, "bf16");
    if (!w) return 0;
    if (cuda_matvec_bf16_cublas((float *)out->ptr, w,
                                in_dim, out_dim,
                                (const float *)x->ptr,
                                "bf16 mapped matvec")) {
        return 1;
    }
    matvec_bf16_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, w, in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "bf16 mapped matvec launch");
}

extern "C" int sf37_cuda_matvec_bf16_argmax_mapped(int32_t *out_token,
                                                    const void *model_map,
                                                    uint64_t model_size,
                                                    uint64_t weight_offset,
                                                    uint64_t in_dim,
                                                    uint64_t out_dim,
                                                    const sf37_cuda_tensor *x,
                                                    int32_t excluded_id) {
    if (!out_token || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim / sizeof(uint16_t)) return 0;
    if (out_dim > (uint64_t)INT32_MAX) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes, "bf16_lm_head_argmax");
    if (!w) return 0;
    const uint32_t n_part = (uint32_t)((out_dim + 7u) / 8u);
    const uint64_t val_off = 0;
    const uint64_t idx_off = cuda_align_u64(n_part * sizeof(float), 256u);
    const uint64_t out_off = cuda_align_u64(idx_off + (uint64_t)n_part * sizeof(int32_t), 256u);
    const uint64_t tmp_bytes = out_off + sizeof(int32_t);
    char *tmp = (char *)cuda_tmp_alloc(tmp_bytes, "bf16 lm_head argmax");
    if (!tmp) return 0;
    float *part_val = (float *)(tmp + val_off);
    int32_t *part_idx = (int32_t *)(tmp + idx_off);
    int32_t *dev_out = (int32_t *)(tmp + out_off);

    matvec_bf16_argmax_stage1_kernel<<<n_part, 256>>>(
            part_val, part_idx, w, in_dim, out_dim,
            (const float *)x->ptr, excluded_id);
    if (!cuda_ok(cudaGetLastError(), "bf16 lm_head argmax stage1 launch")) return 0;
    argmax_partials_kernel<<<1, 1024>>>(dev_out, part_val, part_idx, n_part);
    if (!cuda_ok(cudaGetLastError(), "bf16 lm_head argmax reduce launch")) return 0;
    if (!cuda_ok(cudaMemcpy(out_token, dev_out, sizeof(*out_token),
                            cudaMemcpyDeviceToHost),
                 "bf16 lm_head argmax read")) {
        return 0;
    }
    return 1;
}

extern "C" int sf37_cuda_matmul_bf16_mapped(sf37_cuda_tensor *out,
                                             const void *model_map,
                                             uint64_t model_size,
                                             uint64_t weight_offset,
                                             uint64_t in_dim,
                                             uint64_t out_dim,
                                             const sf37_cuda_tensor *x,
                                             uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim / sizeof(uint16_t) ||
        n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < n_tok * out_dim * sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                               weight_bytes,
                                                               "bf16_batch");
    if (!w) return 0;
    if (cuda_matmul_bf16_cublas((float *)out->ptr,
                                w,
                                in_dim,
                                out_dim,
                                (const float *)x->ptr,
                                n_tok,
                                "bf16 mapped matmul")) {
        return 1;
    }
    dim3 grid((unsigned)((out_dim + 7u) / 8u), (unsigned)n_tok, 1u);
    matmul_bf16_kernel<<<grid, 256>>>((float *)out->ptr,
                                      w,
                                      in_dim,
                                      out_dim,
                                      (const float *)x->ptr,
                                      n_tok);
    return cuda_ok(cudaGetLastError(), "bf16 mapped matmul launch");
}

extern "C" int sf37_cuda_matvec_f32_mapped(sf37_cuda_tensor *out,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t weight_offset,
                                            uint64_t in_dim,
                                            uint64_t out_dim,
                                            const sf37_cuda_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim / sizeof(float)) return 0;
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(float);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         weight_bytes, "f32");
    if (!w) return 0;
    matvec_f32_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, w, in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "f32 mapped matvec launch");
}

extern "C" int sf37_cuda_matmul_f32_mapped(sf37_cuda_tensor *out,
                                            const void *model_map,
                                            uint64_t model_size,
                                            uint64_t weight_offset,
                                            uint64_t in_dim,
                                            uint64_t out_dim,
                                            const sf37_cuda_tensor *x,
                                            uint64_t n_tok) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 || n_tok == 0) return 0;
    if (out_dim > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / in_dim / sizeof(float) ||
        n_tok > UINT64_MAX / out_dim / sizeof(float)) {
        return 0;
    }
    const uint64_t weight_bytes = out_dim * in_dim * sizeof(float);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < n_tok * out_dim * sizeof(float) ||
        x->bytes < n_tok * in_dim * sizeof(float)) {
        return 0;
    }
    const float *w = (const float *)cuda_model_range_ptr(model_map, weight_offset,
                                                         weight_bytes, "f32_batch");
    if (!w) return 0;
    if (g_cublas_ready &&
        in_dim <= (uint64_t)INT_MAX &&
        out_dim <= (uint64_t)INT_MAX &&
        n_tok <= (uint64_t)INT_MAX) {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasStatus_t st = cublasSgemm(g_cublas,
                                        CUBLAS_OP_T,
                                        CUBLAS_OP_N,
                                        (int)out_dim,
                                        (int)n_tok,
                                        (int)in_dim,
                                        &alpha,
                                        w,
                                        (int)in_dim,
                                        (const float *)x->ptr,
                                        (int)in_dim,
                                        &beta,
                                        (float *)out->ptr,
                                        (int)out_dim);
        if (st == CUBLAS_STATUS_SUCCESS) return 1;
        fprintf(stderr, "sf37: cuBLAS f32 matmul failed: status %d; falling back to native kernel\n",
                (int)st);
    }
    dim3 grid((unsigned)((out_dim + 7u) / 8u), (unsigned)n_tok, 1u);
    matmul_f32_kernel<<<grid, 256>>>((float *)out->ptr,
                                     w,
                                     in_dim,
                                     out_dim,
                                     (const float *)x->ptr,
                                     n_tok);
    return cuda_ok(cudaGetLastError(), "f32 mapped matmul launch");
}

__global__ static void layer_norm_bf16_batch_kernel(float *out,
                                                    const float *x,
                                                    const uint16_t *weight,
                                                    const uint16_t *bias,
                                                    uint32_t n,
                                                    uint32_t n_rows,
                                                    float eps) {
    const uint32_t row = blockIdx.x;
    if (row >= n_rows) return;
    const float *xr = x + (uint64_t)row * n;
    float *orow = out + (uint64_t)row * n;

    float sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) sum += xr[i];
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float mean = partial[0] / (float)n;

    float var = 0.0f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float d = xr[i] - mean;
        var += d * d;
    }
    partial[threadIdx.x] = var;
    __syncthreads();
    for (uint32_t stride = blockDim.x >> 1u; stride > 0; stride >>= 1u) {
        if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
        __syncthreads();
    }
    const float scale = rsqrtf(partial[0] / (float)n + eps);

    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        const float w = sf37_bf16_to_f32_dev(weight[i]);
        const float b = bias ? sf37_bf16_to_f32_dev(bias[i]) : 0.0f;
        orow[i] = (xr[i] - mean) * scale * w + b;
    }
}

extern "C" int sf37_cuda_layer_norm_bf16_mapped(sf37_cuda_tensor *out,
                                                 const sf37_cuda_tensor *x,
                                                 const void *model_map,
                                                 uint64_t model_size,
                                                 uint64_t weight_offset,
                                                 uint64_t bias_offset,
                                                 uint32_t n,
                                                 uint32_t n_rows,
                                                 float eps) {
    if (!out || !x || !model_map || n == 0 || n_rows == 0) return 0;
    const uint64_t elems = (uint64_t)n * n_rows;
    const uint64_t vec_bytes = (uint64_t)n * sizeof(uint16_t);
    if (elems > UINT64_MAX / sizeof(float) ||
        out->bytes < elems * sizeof(float) ||
        x->bytes < elems * sizeof(float) ||
        !mapped_range_ok(model_size, weight_offset, vec_bytes)) {
        return 0;
    }
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                               weight_offset,
                                                               vec_bytes,
                                                               "vision_layernorm_weight");
    if (!w) return 0;
    const uint16_t *b = NULL;
    if (bias_offset != UINT64_MAX) {
        if (!mapped_range_ok(model_size, bias_offset, vec_bytes)) return 0;
        b = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                   bias_offset,
                                                   vec_bytes,
                                                   "vision_layernorm_bias");
        if (!b) return 0;
    }
    layer_norm_bf16_batch_kernel<<<n_rows, 256>>>((float *)out->ptr,
                                                  (const float *)x->ptr,
                                                  w, b, n, n_rows, eps);
    return cuda_ok(cudaGetLastError(), "vision layer_norm bf16 mapped launch");
}

__global__ static void add_bias_bf16_kernel(float *x,
                                            const uint16_t *bias,
                                            uint32_t n,
                                            uint32_t n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n * n_rows;
    if (gid >= total) return;
    const uint32_t d = (uint32_t)(gid % n);
    x[gid] += sf37_bf16_to_f32_dev(bias[d]);
}

extern "C" int sf37_cuda_add_bias_bf16_mapped(sf37_cuda_tensor *x,
                                               const void *model_map,
                                               uint64_t model_size,
                                               uint64_t bias_offset,
                                               uint32_t n,
                                               uint32_t n_rows) {
    if (!x || !model_map || n == 0 || n_rows == 0) return 0;
    const uint64_t elems = (uint64_t)n * n_rows;
    const uint64_t bias_bytes = (uint64_t)n * sizeof(uint16_t);
    if (elems > UINT64_MAX / sizeof(float) ||
        x->bytes < elems * sizeof(float) ||
        !mapped_range_ok(model_size, bias_offset, bias_bytes)) {
        return 0;
    }
    const uint16_t *bias = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                                  bias_offset,
                                                                  bias_bytes,
                                                                  "vision_bias_bf16");
    if (!bias) return 0;
    add_bias_bf16_kernel<<<(unsigned)((elems + 255u) / 256u), 256>>>(
            (float *)x->ptr, bias, n, n_rows);
    return cuda_ok(cudaGetLastError(), "vision add bias bf16 launch");
}

__global__ static void add_scaled_bf16_kernel(float *dst,
                                              const float *src,
                                              const uint16_t *gamma,
                                              uint32_t n,
                                              uint32_t n_rows) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n * n_rows;
    if (gid >= total) return;
    const uint32_t d = (uint32_t)(gid % n);
    dst[gid] += src[gid] * sf37_bf16_to_f32_dev(gamma[d]);
}

extern "C" int sf37_cuda_add_scaled_bf16_mapped(sf37_cuda_tensor *dst,
                                                 const sf37_cuda_tensor *src,
                                                 const void *model_map,
                                                 uint64_t model_size,
                                                 uint64_t gamma_offset,
                                                 uint32_t n,
                                                 uint32_t n_rows) {
    if (!dst || !src || !model_map || n == 0 || n_rows == 0) return 0;
    const uint64_t elems = (uint64_t)n * n_rows;
    const uint64_t gamma_bytes = (uint64_t)n * sizeof(uint16_t);
    if (elems > UINT64_MAX / sizeof(float) ||
        dst->bytes < elems * sizeof(float) ||
        src->bytes < elems * sizeof(float) ||
        !mapped_range_ok(model_size, gamma_offset, gamma_bytes)) {
        return 0;
    }
    const uint16_t *gamma = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                                   gamma_offset,
                                                                   gamma_bytes,
                                                                   "vision_layer_scale");
    if (!gamma) return 0;
    add_scaled_bf16_kernel<<<(unsigned)((elems + 255u) / 256u), 256>>>(
            (float *)dst->ptr, (const float *)src->ptr, gamma, n, n_rows);
    return cuda_ok(cudaGetLastError(), "vision add scaled bf16 launch");
}

__global__ static void quick_gelu_kernel(float *x, uint64_t n) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;
    const float v = x[gid];
    x[gid] = v * sf37_sigmoid_dev(1.702f * v);
}

extern "C" int sf37_cuda_quick_gelu_f32(sf37_cuda_tensor *x, uint64_t n) {
    if (!x || n > x->bytes / sizeof(float)) return 0;
    if (n == 0) return 1;
    quick_gelu_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>((float *)x->ptr, n);
    return cuda_ok(cudaGetLastError(), "vision quick_gelu launch");
}

__device__ static float vision_input_at(const float *x,
                                        uint32_t n,
                                        uint32_t c,
                                        uint32_t y,
                                        uint32_t z,
                                        uint32_t in_c,
                                        uint32_t in_h,
                                        uint32_t in_w,
                                        int input_nlc) {
    if (input_nlc) {
        const uint64_t pos = (uint64_t)y * in_w + z;
        return x[((uint64_t)n * in_h * in_w + pos) * in_c + c];
    }
    return x[((uint64_t)n * in_c + c) * in_h * in_w + (uint64_t)y * in_w + z];
}

__global__ static void vision_conv2d_bf16_kernel(float *out,
                                                 const float *x,
                                                 const uint16_t *w,
                                                 const uint16_t *bias,
                                                 uint32_t n_img,
                                                 uint32_t in_c,
                                                 uint32_t in_h,
                                                 uint32_t in_w,
                                                 uint32_t out_c,
                                                 uint32_t out_h,
                                                 uint32_t out_w,
                                                 uint32_t kernel,
                                                 uint32_t stride,
                                                 uint32_t pad,
                                                 int input_nlc) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t total = (uint64_t)n_img * out_h * out_w * out_c;
    if (gid >= total) return;
    uint64_t t = gid;
    const uint32_t oc = (uint32_t)(t % out_c); t /= out_c;
    const uint32_t ox = (uint32_t)(t % out_w); t /= out_w;
    const uint32_t oy = (uint32_t)(t % out_h); t /= out_h;
    const uint32_t n = (uint32_t)t;

    float acc = bias ? sf37_bf16_to_f32_dev(bias[oc]) : 0.0f;
    for (uint32_t ic = 0; ic < in_c; ic++) {
        for (uint32_t ky = 0; ky < kernel; ky++) {
            const int iy = (int)(oy * stride + ky) - (int)pad;
            if (iy < 0 || iy >= (int)in_h) continue;
            for (uint32_t kx = 0; kx < kernel; kx++) {
                const int ix = (int)(ox * stride + kx) - (int)pad;
                if (ix < 0 || ix >= (int)in_w) continue;
                const uint64_t wi = (((uint64_t)oc * in_c + ic) * kernel + ky) * kernel + kx;
                acc += vision_input_at(x, n, ic, (uint32_t)iy, (uint32_t)ix,
                                       in_c, in_h, in_w, input_nlc) *
                       sf37_bf16_to_f32_dev(w[wi]);
            }
        }
    }
    out[gid] = acc;
}

extern "C" int sf37_cuda_vision_conv2d_bf16_mapped(sf37_cuda_tensor *out_nlc,
                                                    const sf37_cuda_tensor *x,
                                                    const void *model_map,
                                                    uint64_t model_size,
                                                    uint64_t weight_offset,
                                                    uint64_t bias_offset,
                                                    uint32_t n_img,
                                                    uint32_t in_c,
                                                    uint32_t in_h,
                                                    uint32_t in_w,
                                                    uint32_t out_c,
                                                    uint32_t kernel,
                                                    uint32_t stride,
                                                    uint32_t pad,
                                                    int input_nlc) {
    if (!out_nlc || !x || !model_map || n_img == 0 || in_c == 0 ||
        in_h == 0 || in_w == 0 || out_c == 0 || kernel == 0 || stride == 0) {
        return 0;
    }
    if (in_h + 2u * pad < kernel || in_w + 2u * pad < kernel) return 0;
    const uint32_t out_h = (in_h + 2u * pad - kernel) / stride + 1u;
    const uint32_t out_w = (in_w + 2u * pad - kernel) / stride + 1u;
    const uint64_t in_elems = (uint64_t)n_img * in_c * in_h * in_w;
    const uint64_t out_elems = (uint64_t)n_img * out_h * out_w * out_c;
    const uint64_t weight_elems = (uint64_t)out_c * in_c * kernel * kernel;
    if (in_elems > UINT64_MAX / sizeof(float) ||
        out_elems > UINT64_MAX / sizeof(float) ||
        weight_elems > UINT64_MAX / sizeof(uint16_t) ||
        x->bytes < in_elems * sizeof(float) ||
        out_nlc->bytes < out_elems * sizeof(float)) {
        return 0;
    }
    const uint64_t weight_bytes = weight_elems * sizeof(uint16_t);
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes)) return 0;
    const uint16_t *w = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                               weight_offset,
                                                               weight_bytes,
                                                               "vision_conv_bf16");
    if (!w) return 0;
    const uint16_t *bias = NULL;
    if (bias_offset != UINT64_MAX) {
        const uint64_t bias_bytes = (uint64_t)out_c * sizeof(uint16_t);
        if (!mapped_range_ok(model_size, bias_offset, bias_bytes)) return 0;
        bias = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                      bias_offset,
                                                      bias_bytes,
                                                      "vision_conv_bias_bf16");
        if (!bias) return 0;
    }
    vision_conv2d_bf16_kernel<<<(unsigned)((out_elems + 255u) / 256u), 256>>>(
            (float *)out_nlc->ptr,
            (const float *)x->ptr,
            w,
            bias,
            n_img, in_c, in_h, in_w, out_c, out_h, out_w,
            kernel, stride, pad, input_nlc);
    return cuda_ok(cudaGetLastError(), "vision conv2d bf16 mapped launch");
}

__device__ static float q8_0_pos_value_dev(const uint8_t *pos,
                                           uint32_t width,
                                           uint32_t row,
                                           uint32_t d) {
    const uint32_t blocks = (width + 31u) / 32u;
    const uint8_t *blk = pos + ((uint64_t)row * blocks + d / 32u) * SF37_Q8_BLOCK_SIZE;
    const float scale = __half2float(*(const __half *)blk);
    const int8_t q = *(const int8_t *)(blk + 2u + (d & 31u));
    return scale * (float)q;
}

__global__ static void vision_add_pos_q8_0_kernel(float *hidden,
                                                  const uint8_t *pos,
                                                  uint32_t width,
                                                  uint32_t grid_h,
                                                  uint32_t grid_w,
                                                  uint32_t base_grid) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t n_tok = (uint64_t)grid_h * grid_w;
    const uint64_t total = n_tok * width;
    if (gid >= total) return;
    const uint32_t d = (uint32_t)(gid % width);
    const uint32_t t = (uint32_t)(gid / width);
    const uint32_t y = t / grid_w;
    const uint32_t x = t - y * grid_w;
    float v;
    if (grid_h == base_grid && grid_w == base_grid) {
        v = q8_0_pos_value_dev(pos, width, t, d);
    } else {
        const float fy = ((float)y + 0.5f) * ((float)base_grid / (float)grid_h) - 0.5f;
        const float fx = ((float)x + 0.5f) * ((float)base_grid / (float)grid_w) - 0.5f;
        int y0 = (int)floorf(fy);
        int x0 = (int)floorf(fx);
        float wy = fy - (float)y0;
        float wx = fx - (float)x0;
        if (y0 < 0) { y0 = 0; wy = 0.0f; }
        if (x0 < 0) { x0 = 0; wx = 0.0f; }
        int y1 = y0 + 1;
        int x1 = x0 + 1;
        if (y1 >= (int)base_grid) y1 = (int)base_grid - 1;
        if (x1 >= (int)base_grid) x1 = (int)base_grid - 1;
        const uint32_t r00 = (uint32_t)y0 * base_grid + (uint32_t)x0;
        const uint32_t r01 = (uint32_t)y0 * base_grid + (uint32_t)x1;
        const uint32_t r10 = (uint32_t)y1 * base_grid + (uint32_t)x0;
        const uint32_t r11 = (uint32_t)y1 * base_grid + (uint32_t)x1;
        const float v00 = q8_0_pos_value_dev(pos, width, r00, d);
        const float v01 = q8_0_pos_value_dev(pos, width, r01, d);
        const float v10 = q8_0_pos_value_dev(pos, width, r10, d);
        const float v11 = q8_0_pos_value_dev(pos, width, r11, d);
        v = (1.0f - wy) * ((1.0f - wx) * v00 + wx * v01) +
            wy * ((1.0f - wx) * v10 + wx * v11);
    }
    hidden[gid] += v;
}

extern "C" int sf37_cuda_vision_add_pos_q8_0_mapped(sf37_cuda_tensor *hidden,
                                                     const void *model_map,
                                                     uint64_t model_size,
                                                     uint64_t pos_offset,
                                                     uint32_t width,
                                                     uint32_t grid_h,
                                                     uint32_t grid_w,
                                                     uint32_t base_grid) {
    if (!hidden || !model_map || width == 0 || grid_h == 0 || grid_w == 0 ||
        base_grid == 0) {
        return 0;
    }
    const uint64_t blocks = (width + 31u) / 32u;
    const uint64_t pos_rows = (uint64_t)base_grid * base_grid;
    const uint64_t pos_bytes = pos_rows * blocks * SF37_Q8_BLOCK_SIZE;
    const uint64_t n = (uint64_t)grid_h * grid_w * width;
    if (hidden->bytes < n * sizeof(float) ||
        !mapped_range_ok(model_size, pos_offset, pos_bytes)) {
        return 0;
    }
    const uint8_t *pos = (const uint8_t *)cuda_model_range_ptr(model_map,
                                                               pos_offset,
                                                               pos_bytes,
                                                               "vision_pos_q8");
    if (!pos) return 0;
    vision_add_pos_q8_0_kernel<<<(unsigned)((n + 255u) / 256u), 256>>>(
            (float *)hidden->ptr, pos, width, grid_h, grid_w, base_grid);
    return cuda_ok(cudaGetLastError(), "vision add positional q8 launch");
}

__device__ static float vision_rope_value(float self,
                                          float mate,
                                          uint32_t d,
                                          uint32_t head_dim,
                                          uint32_t row,
                                          uint32_t col,
                                          uint32_t max_grid_w,
                                          double theta) {
    (void)max_grid_w;
    const uint32_t half = head_dim / 2u;
    const uint32_t axis_d = d < half ? d : d - half;
    const uint32_t pair = axis_d / 2u;
    const float coord = d < half ? (float)col : (float)row;
    const double inv = 1.0 / pow(theta, (double)(pair * 2u) / (double)half);
    const float freq = (float)((double)coord * inv);
    const float c = cosf(freq);
    const float s = sinf(freq);
    if ((d & 1u) == 0) return self * c - mate * s;
    return self * c + mate * s;
}

__global__ static void vision_qkv_split_rope_bf16_kernel(float *q,
                                                         float *k,
                                                         float *v,
                                                         const float *qkv,
                                                         const uint16_t *bias,
                                                         uint32_t n_tok,
                                                         uint32_t grid_h,
                                                         uint32_t grid_w,
                                                         uint32_t n_heads,
                                                         uint32_t head_dim,
                                                         uint32_t max_grid_w,
                                                         double theta) {
    const uint64_t gid = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t width = (uint64_t)n_heads * head_dim;
    const uint64_t total = (uint64_t)n_tok * width;
    if (gid >= total) return;
    const uint32_t d_global = (uint32_t)(gid % width);
    const uint32_t tok = (uint32_t)(gid / width);
    const uint32_t h = d_global / head_dim;
    const uint32_t d = d_global - h * head_dim;
    const uint32_t mate_d = d ^ 1u;
    const uint32_t mate_global = h * head_dim + mate_d;
    const uint32_t row = tok / grid_w;
    const uint32_t col = tok - row * grid_w;
    const uint64_t base = (uint64_t)tok * width * 3u;
    const uint64_t out = (uint64_t)tok * width + d_global;

    const float q_self = qkv[base + d_global] + sf37_bf16_to_f32_dev(bias[d_global]);
    const float q_mate = qkv[base + mate_global] + sf37_bf16_to_f32_dev(bias[mate_global]);
    const uint64_t k_base = base + width;
    const float k_self = qkv[k_base + d_global] + sf37_bf16_to_f32_dev(bias[width + d_global]);
    const float k_mate = qkv[k_base + mate_global] + sf37_bf16_to_f32_dev(bias[width + mate_global]);
    const uint64_t v_base = base + width * 2u;
    q[out] = vision_rope_value(q_self, q_mate, d, head_dim, row, col, max_grid_w, theta);
    k[out] = vision_rope_value(k_self, k_mate, d, head_dim, row, col, max_grid_w, theta);
    v[out] = qkv[v_base + d_global] + sf37_bf16_to_f32_dev(bias[width * 2u + d_global]);
}

extern "C" int sf37_cuda_vision_qkv_split_rope_bf16_mapped(sf37_cuda_tensor *q,
                                                            sf37_cuda_tensor *k,
                                                            sf37_cuda_tensor *v,
                                                            const sf37_cuda_tensor *qkv,
                                                            const void *model_map,
                                                            uint64_t model_size,
                                                            uint64_t bias_offset,
                                                            uint32_t n_tok,
                                                            uint32_t grid_h,
                                                            uint32_t grid_w,
                                                            uint32_t n_heads,
                                                            uint32_t head_dim,
                                                            uint32_t max_grid_w,
                                                            double theta) {
    if (!q || !k || !v || !qkv || !model_map || n_tok == 0 ||
        grid_h == 0 || grid_w == 0 || n_heads == 0 || head_dim == 0 ||
        (head_dim & 1u) != 0 || head_dim < 4u || max_grid_w == 0) {
        return 0;
    }
    const uint64_t width = (uint64_t)n_heads * head_dim;
    const uint64_t total = (uint64_t)n_tok * width;
    const uint64_t qkv_elems = total * 3u;
    const uint64_t bias_bytes = width * 3u * sizeof(uint16_t);
    if (q->bytes < total * sizeof(float) ||
        k->bytes < total * sizeof(float) ||
        v->bytes < total * sizeof(float) ||
        qkv->bytes < qkv_elems * sizeof(float) ||
        !mapped_range_ok(model_size, bias_offset, bias_bytes)) {
        return 0;
    }
    const uint16_t *bias = (const uint16_t *)cuda_model_range_ptr(model_map,
                                                                  bias_offset,
                                                                  bias_bytes,
                                                                  "vision_qkv_bias");
    if (!bias) return 0;
    vision_qkv_split_rope_bf16_kernel<<<(unsigned)((total + 255u) / 256u), 256>>>(
            (float *)q->ptr,
            (float *)k->ptr,
            (float *)v->ptr,
            (const float *)qkv->ptr,
            bias,
            n_tok, grid_h, grid_w, n_heads, head_dim, max_grid_w, theta);
    return cuda_ok(cudaGetLastError(), "vision qkv split rope launch");
}

__global__ static void vision_attention_warp_kernel(float *out,
                                                    const float *q,
                                                    const float *k,
                                                    const float *v,
                                                    uint32_t n_tok,
                                                    uint32_t n_heads,
                                                    uint32_t head_dim) {
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5u;
    const uint32_t hw = blockIdx.x * 8u + warp;
    const uint32_t total_hw = n_tok * n_heads;
    if (hw >= total_hw) return;
    const uint32_t tok = hw / n_heads;
    const uint32_t h = hw - tok * n_heads;
    const uint64_t width = (uint64_t)n_heads * head_dim;
    const float *qh = q + (uint64_t)tok * width + (uint64_t)h * head_dim;
    const float scale = rsqrtf((float)head_dim);
    float qv[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float ov[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
    for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t d = lane + s * 32u;
        if (d < head_dim) qv[s] = qh[d];
    }

    float max_s = -INFINITY;
    float sum_s = 0.0f;
    const unsigned mask = 0xffffffffu;
    for (uint32_t key = 0; key < n_tok; key++) {
        const float *kh = k + (uint64_t)key * width + (uint64_t)h * head_dim;
        const float *vh = v + (uint64_t)key * width + (uint64_t)h * head_dim;
        float dot = 0.0f;
#pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const uint32_t d = lane + s * 32u;
            if (d < head_dim) dot += qv[s] * kh[d];
        }
        dot = warp_sum_f32(dot);
        const float score = __shfl_sync(mask, dot, 0) * scale;
        const float old_m = max_s;
        const float new_m = fmaxf(max_s, score);
        const float old_scale = expf(old_m - new_m);
        const float row_scale = expf(score - new_m);
        sum_s = sum_s * old_scale + row_scale;
#pragma unroll
        for (uint32_t s = 0; s < 4u; s++) {
            const uint32_t d = lane + s * 32u;
            if (d < head_dim) ov[s] = ov[s] * old_scale + row_scale * vh[d];
        }
        max_s = new_m;
    }
    const float inv = sum_s > 0.0f ? 1.0f / sum_s : 0.0f;
    float *oh = out + (uint64_t)tok * width + (uint64_t)h * head_dim;
#pragma unroll
    for (uint32_t s = 0; s < 4u; s++) {
        const uint32_t d = lane + s * 32u;
        if (d < head_dim) oh[d] = ov[s] * inv;
    }
}

extern "C" int sf37_cuda_vision_attention(sf37_cuda_tensor *out,
                                           const sf37_cuda_tensor *q,
                                           const sf37_cuda_tensor *k,
                                           const sf37_cuda_tensor *v,
                                           uint32_t n_tok,
                                           uint32_t n_heads,
                                           uint32_t head_dim) {
    if (!out || !q || !k || !v || n_tok == 0 || n_heads == 0 ||
        head_dim == 0 || head_dim > 128u) {
        return 0;
    }
    const uint64_t elems = (uint64_t)n_tok * n_heads * head_dim;
    if (elems > UINT64_MAX / sizeof(float) ||
        out->bytes < elems * sizeof(float) ||
        q->bytes < elems * sizeof(float) ||
        k->bytes < elems * sizeof(float) ||
        v->bytes < elems * sizeof(float)) {
        return 0;
    }
    const uint64_t warps = (uint64_t)n_tok * n_heads;
    vision_attention_warp_kernel<<<(unsigned)((warps + 7u) / 8u), 256>>>(
            (float *)out->ptr,
            (const float *)q->ptr,
            (const float *)k->ptr,
            (const float *)v->ptr,
            n_tok, n_heads, head_dim);
    return cuda_ok(cudaGetLastError(), "vision attention warp launch");
}

extern "C" int sf37_cuda_matvec_q3_asym_mapped(sf37_cuda_tensor *out,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                const sf37_cuda_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        in_dim % SF37_QK_K != 0) {
        return 0;
    }
    const uint64_t blocks = in_dim / SF37_QK_K;
    if (blocks != 0 && out_dim > UINT64_MAX / blocks / SF37_Q3_BLOCK_SIZE) return 0;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q3_BLOCK_SIZE;
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                             weight_bytes, "q3_asym");
    if (!w) return 0;
    if (cuda_qlow_q8k_use_for(out_dim) && in_dim <= (uint64_t)UINT32_MAX &&
        blocks <= (uint64_t)UINT32_MAX) {
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                blocks * sizeof(sf37_cuda_block_q8_k), "q3 asym mapped q8k activation");
        if (xq) {
            q8_k_quantize_sf37_kernel<<<dim3((unsigned)blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, (uint32_t)in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "q3 asym mapped q8k quantize launch")) return 0;
            matvec_q3_asym_q8k_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
                    (float *)out->ptr, w, xq, in_dim, out_dim, blocks);
            return cuda_ok(cudaGetLastError(), "q3 asym mapped q8k matvec launch");
        }
    }
    matvec_q3_asym_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, w, in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "q3 asym mapped matvec launch");
}

extern "C" int sf37_cuda_matvec_q2_asym_mapped(sf37_cuda_tensor *out,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint64_t in_dim,
                                                uint64_t out_dim,
                                                const sf37_cuda_tensor *x) {
    if (!out || !x || !model_map || in_dim == 0 || out_dim == 0 ||
        in_dim % SF37_QK_K != 0) {
        return 0;
    }
    const uint64_t blocks = in_dim / SF37_QK_K;
    if (blocks != 0 && out_dim > UINT64_MAX / blocks / SF37_Q2_BLOCK_SIZE) return 0;
    const uint64_t weight_bytes = out_dim * blocks * SF37_Q2_BLOCK_SIZE;
    if (!mapped_range_ok(model_size, weight_offset, weight_bytes) ||
        out->bytes < out_dim * sizeof(float) ||
        x->bytes < in_dim * sizeof(float)) {
        return 0;
    }
    const uint8_t *w = (const uint8_t *)cuda_model_range_ptr(model_map, weight_offset,
                                                             weight_bytes, "q2_asym");
    if (!w) return 0;
    if (cuda_qlow_q8k_use_for(out_dim) && in_dim <= (uint64_t)UINT32_MAX &&
        blocks <= (uint64_t)UINT32_MAX) {
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                blocks * sizeof(sf37_cuda_block_q8_k), "q2 asym mapped q8k activation");
        if (xq) {
            q8_k_quantize_sf37_kernel<<<dim3((unsigned)blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, (uint32_t)in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "q2 asym mapped q8k quantize launch")) return 0;
            matvec_q2_asym_q8k_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
                    (float *)out->ptr, w, xq, in_dim, out_dim, blocks);
            return cuda_ok(cudaGetLastError(), "q2 asym mapped q8k matvec launch");
        }
    }
    matvec_q2_asym_kernel<<<(unsigned)((out_dim + 7u) / 8u), 256>>>(
            (float *)out->ptr, w, in_dim, out_dim, (const float *)x->ptr);
    return cuda_ok(cudaGetLastError(), "q2 asym mapped matvec launch");
}

extern "C" int sf37_cuda_router_select_mapped(sf37_cuda_tensor *selected,
                                               sf37_cuda_tensor *weights,
                                               sf37_cuda_tensor *probs,
                                               const sf37_cuda_tensor *logits,
                                               const void *model_map,
                                               uint64_t model_size,
                                               uint64_t bias_offset,
                                               uint32_t n_experts,
                                               uint32_t topk,
                                               float scale) {
    if (!selected || !weights || !logits || !model_map ||
        n_experts == 0 || topk == 0 || topk > 16u) {
        return 0;
    }
    const uint64_t bias_bytes = (uint64_t)n_experts * sizeof(float);
    if (!mapped_range_ok(model_size, bias_offset, bias_bytes) ||
        selected->bytes < (uint64_t)topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)topk * sizeof(float) ||
        logits->bytes < (uint64_t)n_experts * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_experts * sizeof(float))) {
        return 0;
    }
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset,
                                                            bias_bytes, "router_bias");
    if (!bias) return 0;
    if (n_experts == 288u && topk == 8u &&
        !cuda_env_present("SF37_CUDA_NO_WARP_ROUTER_SELECT", NULL)) {
        router_select_sf37_warp_topk_kernel<<<1, 32>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                probs ? (float *)probs->ptr : NULL,
                (const float *)logits->ptr,
                bias,
                scale);
        return cuda_ok(cudaGetLastError(), "router_select mapped warp topk launch");
    }
    router_select_sf37_kernel<<<1, 1>>>(
            (int32_t *)selected->ptr,
            (float *)weights->ptr,
            probs ? (float *)probs->ptr : NULL,
            (const float *)logits->ptr,
            bias,
            n_experts, topk, scale);
    return cuda_ok(cudaGetLastError(), "router_select mapped launch");
}

__global__ static void router_select_sf37_warp_topk_batch_kernel(int32_t *selected,
                                                                 float *weights,
                                                                 float *probs,
                                                                 const float *logits,
                                                                 const float *bias,
                                                                 float scale,
                                                                 uint32_t n_tok) {
    const uint32_t lane = threadIdx.x;
    const uint32_t row_in_block = threadIdx.y;
    const uint32_t t = blockIdx.x * blockDim.y + row_in_block;
    if (t >= n_tok || lane >= 32u) return;

    const float *log = logits + (uint64_t)t * 288u;
    float *prob = probs ? probs + (uint64_t)t * 288u : NULL;
    int32_t *sel = selected + (uint64_t)t * 8u;
    float *w = weights + (uint64_t)t * 8u;

    uint32_t local_idx[9];
    float local_prob[9];
    float local_score[9];
    uint32_t local_n = 0;
#pragma unroll
    for (uint32_t j = 0; j < 9u; j++) {
        const uint32_t e = lane + j * 32u;
        if (e < 288u) {
            const float p = sf37_sigmoid_dev(log[e]);
            local_idx[local_n] = e;
            local_prob[local_n] = p;
            local_score[local_n] = p + bias[e];
            if (prob) prob[e] = p;
            local_n++;
        }
    }

    uint32_t sel_idx[8];
    float sel_prob[8];
    const unsigned mask = 0xffffffffu;
#pragma unroll
    for (uint32_t k = 0; k < 8u; k++) {
        float best_score = -INFINITY;
        float best_prob = 0.0f;
        uint32_t best_idx = UINT32_MAX;
#pragma unroll
        for (uint32_t j = 0; j < 9u; j++) {
            if (j >= local_n) continue;
            const uint32_t e = local_idx[j];
            bool used = false;
#pragma unroll
            for (uint32_t p = 0; p < 8u; p++) {
                if (p >= k) continue;
                if (sel_idx[p] == e) used = true;
            }
            if (!used && router_score_better_sf37(local_score[j], e,
                                                  best_score, best_idx)) {
                best_score = local_score[j];
                best_prob = local_prob[j];
                best_idx = e;
            }
        }

#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float os = __shfl_down_sync(mask, best_score, off);
            const float op = __shfl_down_sync(mask, best_prob, off);
            const uint32_t oi = __shfl_down_sync(mask, best_idx, off);
            if (router_score_better_sf37(os, oi, best_score, best_idx)) {
                best_score = os;
                best_prob = op;
                best_idx = oi;
            }
        }
        best_idx = __shfl_sync(mask, best_idx, 0);
        best_prob = __shfl_sync(mask, best_prob, 0);
        sel_idx[k] = best_idx;
        sel_prob[k] = best_prob;
        if (lane == 0) sel[k] = (int32_t)best_idx;
    }

    if (lane == 0) {
        float sum = 0.0f;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) sum += sel_prob[k];
        if (sum < 1.0e-20f) sum = 1.0e-20f;
#pragma unroll
        for (uint32_t k = 0; k < 8u; k++) w[k] = sel_prob[k] / sum * scale;
    }
}

extern "C" int sf37_cuda_router_select_batch_mapped(sf37_cuda_tensor *selected,
                                                     sf37_cuda_tensor *weights,
                                                     sf37_cuda_tensor *probs,
                                                     const sf37_cuda_tensor *logits,
                                                     const void *model_map,
                                                     uint64_t model_size,
                                                     uint64_t bias_offset,
                                                     uint32_t n_experts,
                                                     uint32_t topk,
                                                     float scale,
                                                     uint32_t n_tok) {
    if (!selected || !weights || !logits || !model_map ||
        n_experts == 0 || topk == 0 || topk > 16u || n_tok == 0) {
        return 0;
    }
    const uint64_t bias_bytes = (uint64_t)n_experts * sizeof(float);
    if (n_tok > UINT64_MAX / n_experts / sizeof(float) ||
        n_tok > UINT64_MAX / topk / sizeof(float) ||
        !mapped_range_ok(model_size, bias_offset, bias_bytes) ||
        logits->bytes < (uint64_t)n_tok * n_experts * sizeof(float) ||
        selected->bytes < (uint64_t)n_tok * topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)n_tok * topk * sizeof(float) ||
        (probs && probs->bytes < (uint64_t)n_tok * n_experts * sizeof(float))) {
        return 0;
    }
    const float *bias = (const float *)cuda_model_range_ptr(model_map, bias_offset,
                                                            bias_bytes,
                                                            "router_bias_batch");
    if (!bias) return 0;
    if (n_experts == 288u && topk == 8u &&
        !cuda_env_present("SF37_CUDA_NO_BATCH_ROUTER", NULL) &&
        !cuda_env_present("SF37_CUDA_NO_WARP_ROUTER_SELECT", NULL)) {
        dim3 block(32u, 4u, 1u);
        router_select_sf37_warp_topk_batch_kernel<<<(n_tok + 3u) / 4u, block>>>(
                (int32_t *)selected->ptr,
                (float *)weights->ptr,
                probs ? (float *)probs->ptr : NULL,
                (const float *)logits->ptr,
                bias,
                scale,
                n_tok);
        return cuda_ok(cudaGetLastError(), "router_select mapped batch warp topk launch");
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        sf37_cuda_tensor sel_view = {
            (char *)selected->ptr + (uint64_t)t * topk * sizeof(int32_t),
            (uint64_t)topk * sizeof(int32_t)
        };
        sf37_cuda_tensor w_view = {
            (char *)weights->ptr + (uint64_t)t * topk * sizeof(float),
            (uint64_t)topk * sizeof(float)
        };
        sf37_cuda_tensor prob_view = {
            probs ? (char *)probs->ptr + (uint64_t)t * n_experts * sizeof(float) : NULL,
            (uint64_t)n_experts * sizeof(float)
        };
        sf37_cuda_tensor log_view = {
            (char *)logits->ptr + (uint64_t)t * n_experts * sizeof(float),
            (uint64_t)n_experts * sizeof(float)
        };
        if (!sf37_cuda_router_select_mapped(&sel_view,
                                            &w_view,
                                            probs ? &prob_view : NULL,
                                            &log_view,
                                            model_map,
                                            model_size,
                                            bias_offset,
                                            n_experts,
                                            topk,
                                            scale)) {
            return 0;
        }
    }
    return 1;
}

extern "C" int sf37_cuda_routed_moe_one_mapped(sf37_cuda_tensor *out,
                                                sf37_cuda_tensor *gate,
                                                sf37_cuda_tensor *up,
                                                sf37_cuda_tensor *mid,
                                                sf37_cuda_tensor *down,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t gate_offset,
                                                uint64_t up_offset,
                                                uint64_t down_offset,
                                                const sf37_cuda_tensor *selected,
                                                const sf37_cuda_tensor *weights,
                                                uint32_t n_total_expert,
                                                uint32_t topk,
                                                uint32_t in_dim,
                                                uint32_t expert_mid_dim,
                                                uint32_t out_dim,
                                                float clamp,
                                                const sf37_cuda_tensor *x) {
    if (!out || !gate || !up || !mid || !down || !model_map ||
        !selected || !weights || !x ||
        n_total_expert == 0 || topk == 0 ||
        in_dim == 0 || expert_mid_dim == 0 || out_dim == 0 ||
        in_dim % SF37_QK_K != 0 || expert_mid_dim % SF37_QK_K != 0) {
        return 0;
    }
    const uint64_t gate_row_bytes = (uint64_t)(in_dim / SF37_QK_K) * SF37_Q3_BLOCK_SIZE;
    const uint64_t down_row_bytes = (uint64_t)(expert_mid_dim / SF37_QK_K) * SF37_Q2_BLOCK_SIZE;
    if (expert_mid_dim != 0 &&
        n_total_expert > UINT64_MAX / expert_mid_dim / gate_row_bytes) return 0;
    if (out_dim != 0 &&
        n_total_expert > UINT64_MAX / out_dim / down_row_bytes) return 0;
    const uint64_t gate_bytes = (uint64_t)n_total_expert * expert_mid_dim * gate_row_bytes;
    const uint64_t down_bytes = (uint64_t)n_total_expert * out_dim * down_row_bytes;
    const uint64_t pair_mid_bytes = (uint64_t)topk * expert_mid_dim * sizeof(float);
    const uint64_t pair_down_bytes = (uint64_t)topk * out_dim * sizeof(float);
	    if (!mapped_range_ok(model_size, gate_offset, gate_bytes) ||
	        !mapped_range_ok(model_size, up_offset, gate_bytes) ||
	        !mapped_range_ok(model_size, down_offset, down_bytes) ||
	        out->bytes < (uint64_t)out_dim * sizeof(float) ||
        gate->bytes < pair_mid_bytes ||
        up->bytes < pair_mid_bytes ||
        mid->bytes < pair_mid_bytes ||
        down->bytes < pair_down_bytes ||
        selected->bytes < (uint64_t)topk * sizeof(int32_t) ||
        weights->bytes < (uint64_t)topk * sizeof(float) ||
	        x->bytes < (uint64_t)in_dim * sizeof(float)) {
	        return 0;
	    }
	
	    if (cuda_moe_selected_expert_cache_enabled() &&
	        cuda_moe_direct_down_sum_enabled() &&
	        cuda_qlow_q8k_use_for(expert_mid_dim) &&
	        cuda_qlow_q8k_use_for(out_dim) &&
	        topk <= 64u &&
	        in_dim <= (uint32_t)UINT32_MAX &&
	        expert_mid_dim <= (uint32_t)UINT32_MAX) {
	        int32_t h_selected[64];
	        int32_t h_remap[64];
	        cudaError_t err = cudaMemcpy(h_selected, selected->ptr,
	                                     (size_t)topk * sizeof(int32_t),
	                                     cudaMemcpyDeviceToHost);
	        if (err == cudaSuccess) {
	            int valid = 1;
	            for (uint32_t i = 0; i < topk; i++) {
	                h_remap[i] = (int32_t)i;
	                if (h_selected[i] < 0 ||
	                    (uint32_t)h_selected[i] >= n_total_expert) {
	                    valid = 0;
	                }
	            }
	            const uint32_t xq_blocks = in_dim / SF37_QK_K;
	            const uint32_t midq_blocks = expert_mid_dim / SF37_QK_K;
	            const uint64_t gate_slice_bytes = (uint64_t)expert_mid_dim * gate_row_bytes;
	            const uint64_t down_slice_bytes = (uint64_t)out_dim * down_row_bytes;
	            uint64_t scratch_bytes = 0;
	            uint64_t gate_sel_off = 0;
	            uint64_t up_sel_off = 0;
	            uint64_t down_sel_off = 0;
	            uint64_t remap_off = 0;
	            uint64_t xq_off = 0;
	            uint64_t midq_off = 0;
	            int scratch_ok = valid;
	#define SF37_SEL_RESERVE(name, count, type) do { \
	                if (!scratch_ok) break; \
	                const uint64_t _cnt = (uint64_t)(count); \
	                if (_cnt > UINT64_MAX / (uint64_t)sizeof(type)) { scratch_ok = 0; break; } \
	                scratch_bytes = cuda_align_u64(scratch_bytes, 256u); \
	                name##_off = scratch_bytes; \
	                const uint64_t _bytes = _cnt * (uint64_t)sizeof(type); \
	                if (_bytes > UINT64_MAX - scratch_bytes) { scratch_ok = 0; break; } \
	                scratch_bytes += _bytes; \
	            } while (0)
	            SF37_SEL_RESERVE(gate_sel, (uint64_t)topk * gate_slice_bytes, uint8_t);
	            SF37_SEL_RESERVE(up_sel, (uint64_t)topk * gate_slice_bytes, uint8_t);
	            SF37_SEL_RESERVE(down_sel, (uint64_t)topk * down_slice_bytes, uint8_t);
	            SF37_SEL_RESERVE(remap, topk, int32_t);
	            SF37_SEL_RESERVE(xq, xq_blocks, sf37_cuda_block_q8_k);
	            SF37_SEL_RESERVE(midq, (uint64_t)topk * midq_blocks, sf37_cuda_block_q8_k);
	#undef SF37_SEL_RESERVE
	            char *scratch = scratch_ok ?
	                (char *)cuda_tmp_alloc(scratch_bytes, "routed_moe selected experts") :
	                NULL;
	            if (scratch) {
	                uint8_t *gate_sel = (uint8_t *)(scratch + gate_sel_off);
	                uint8_t *up_sel = (uint8_t *)(scratch + up_sel_off);
	                uint8_t *down_sel = (uint8_t *)(scratch + down_sel_off);
	                int32_t *remap = (int32_t *)(scratch + remap_off);
	                sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)(scratch + xq_off);
	                sf37_cuda_block_q8_k *midq = (sf37_cuda_block_q8_k *)(scratch + midq_off);
	                int copy_ok = 1;
	                for (uint32_t slot = 0; slot < topk; slot++) {
	                    const uint32_t expert = (uint32_t)h_selected[slot];
	                    const uint64_t gate_src_off = gate_offset + (uint64_t)expert * gate_slice_bytes;
	                    const uint64_t up_src_off = up_offset + (uint64_t)expert * gate_slice_bytes;
	                    const uint64_t down_src_off = down_offset + (uint64_t)expert * down_slice_bytes;
	                    const uint8_t *gate_src = (const uint8_t *)cuda_model_range_ptr(
	                            model_map, gate_src_off, gate_slice_bytes, "moe_gate.selected");
	                    const uint8_t *up_src = (const uint8_t *)cuda_model_range_ptr(
	                            model_map, up_src_off, gate_slice_bytes, "moe_up.selected");
	                    const uint8_t *down_src = (const uint8_t *)cuda_model_range_ptr(
	                            model_map, down_src_off, down_slice_bytes, "moe_down.selected");
	                    if (!gate_src || !up_src || !down_src) {
	                        copy_ok = 0;
	                        break;
	                    }
	                    err = cudaMemcpy(gate_sel + (uint64_t)slot * gate_slice_bytes,
	                                     gate_src, (size_t)gate_slice_bytes,
	                                     cudaMemcpyDefault);
	                    if (err != cudaSuccess) { copy_ok = 0; break; }
	                    err = cudaMemcpy(up_sel + (uint64_t)slot * gate_slice_bytes,
	                                     up_src, (size_t)gate_slice_bytes,
	                                     cudaMemcpyDefault);
	                    if (err != cudaSuccess) { copy_ok = 0; break; }
	                    err = cudaMemcpy(down_sel + (uint64_t)slot * down_slice_bytes,
	                                     down_src, (size_t)down_slice_bytes,
	                                     cudaMemcpyDefault);
	                    if (err != cudaSuccess) { copy_ok = 0; break; }
	                }
	                if (copy_ok) {
	                    err = cudaMemcpy(remap, h_remap, (size_t)topk * sizeof(int32_t),
	                                     cudaMemcpyHostToDevice);
	                    if (err != cudaSuccess) copy_ok = 0;
	                }
	                if (copy_ok) {
	                    cudaEvent_t prof_ev[5] = {NULL, NULL, NULL, NULL, NULL};
	                    const int profile_moe = cuda_moe_profile_start(prof_ev);
	                    const int write_gate_up = cuda_moe_write_gate_up();
	                    q8_k_quantize_sf37_kernel<<<dim3(xq_blocks, 1u, 1u), 256>>>(
	                            xq, (const float *)x->ptr, in_dim, 1u);
	                    if (!cuda_ok(cudaGetLastError(), "routed_moe selected q8k input quantize launch")) {
	                        if (profile_moe) cuda_moe_profile_abort(prof_ev);
	                        return 0;
	                    }
	                    if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);
	                    dim3 q8_mgrid((expert_mid_dim + 31u) / 32u, topk, 1u);
	                    moe_gate_up_mid_q3_asym_q8k_kernel<<<q8_mgrid, 256>>>(
	                            (float *)gate->ptr,
	                            (float *)up->ptr,
	                            (float *)mid->ptr,
	                            gate_sel,
	                            up_sel,
	                            xq,
	                            remap,
	                            (const float *)weights->ptr,
	                            topk, xq_blocks, expert_mid_dim, topk,
	                            clamp, write_gate_up);
	                    if (!cuda_ok(cudaGetLastError(), "routed_moe selected q8k gate/up launch")) {
	                        if (profile_moe) cuda_moe_profile_abort(prof_ev);
	                        return 0;
	                    }
	                    if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);
	                    q8_k_quantize_sf37_kernel<<<dim3(midq_blocks, topk, 1u), 256>>>(
	                            midq, (const float *)mid->ptr, expert_mid_dim, topk);
	                    if (!cuda_ok(cudaGetLastError(), "routed_moe selected q8k mid quantize launch")) {
	                        if (profile_moe) cuda_moe_profile_abort(prof_ev);
	                        return 0;
	                    }
	                    if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);
	                    moe_down_q2_asym_sum_q8k_kernel<<<(out_dim + 31u) / 32u, 256>>>(
	                            (float *)out->ptr,
	                            down_sel,
	                            midq,
	                            remap,
	                            topk, midq_blocks, out_dim, topk);
	                    if (!cuda_ok(cudaGetLastError(), "routed_moe selected q8k direct down-sum launch")) {
	                        if (profile_moe) cuda_moe_profile_abort(prof_ev);
	                        return 0;
	                    }
	                    if (profile_moe) {
	                        (void)cudaEventRecord(prof_ev[4], 0);
	                        cuda_moe_profile_finish(prof_ev, 1, topk,
	                                                in_dim, expert_mid_dim, out_dim);
	                    }
	                    return 1;
	                }
	                (void)cudaGetLastError();
	            }
	        } else {
	            (void)cudaGetLastError();
	        }
	    }
	
	    const uint8_t *gate_w = (const uint8_t *)cuda_model_range_ptr(model_map, gate_offset,
	                                                                  gate_bytes, "moe_gate");
    const uint8_t *up_w = (const uint8_t *)cuda_model_range_ptr(model_map, up_offset,
                                                                gate_bytes, "moe_up");
    const uint8_t *down_w = (const uint8_t *)cuda_model_range_ptr(model_map, down_offset,
                                                                  down_bytes, "moe_down");
    if (!gate_w || !up_w || !down_w) return 0;

    cudaEvent_t prof_ev[5] = {NULL, NULL, NULL, NULL, NULL};
    const int profile_moe = cuda_moe_profile_start(prof_ev);
    const int write_gate_up = cuda_moe_write_gate_up();
    const int direct_down_sum = cuda_moe_direct_down_sum_enabled();

    if (direct_down_sum &&
        cuda_qlow_q8k_use_for(expert_mid_dim) &&
        cuda_qlow_q8k_use_for(out_dim) &&
        in_dim <= (uint32_t)UINT32_MAX &&
        expert_mid_dim <= (uint32_t)UINT32_MAX) {
        const uint32_t xq_blocks = in_dim / SF37_QK_K;
        const uint32_t midq_blocks = expert_mid_dim / SF37_QK_K;
        const uint64_t xq_count = (uint64_t)xq_blocks;
        const uint64_t midq_count = (uint64_t)topk * midq_blocks;
        const uint64_t xq_bytes = xq_count * sizeof(sf37_cuda_block_q8_k);
        const uint64_t midq_bytes = midq_count * sizeof(sf37_cuda_block_q8_k);
        sf37_cuda_block_q8_k *xq = (sf37_cuda_block_q8_k *)cuda_tmp_alloc(
                xq_bytes + midq_bytes, "routed_moe mapped q8k activations");
        if (xq) {
            sf37_cuda_block_q8_k *midq = (sf37_cuda_block_q8_k *)((char *)xq + xq_bytes);
            q8_k_quantize_sf37_kernel<<<dim3(xq_blocks, 1u, 1u), 256>>>(
                    xq, (const float *)x->ptr, in_dim, 1u);
            if (!cuda_ok(cudaGetLastError(), "routed_moe mapped q8k input quantize launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);

            dim3 q8_mgrid((expert_mid_dim + 31u) / 32u, topk, 1u);
            moe_gate_up_mid_q3_asym_q8k_kernel<<<q8_mgrid, 256>>>(
                    (float *)gate->ptr,
                    (float *)up->ptr,
                    (float *)mid->ptr,
                    gate_w,
                    up_w,
                    xq,
                    (const int32_t *)selected->ptr,
                    (const float *)weights->ptr,
                    topk, xq_blocks, expert_mid_dim, n_total_expert,
                    clamp, write_gate_up);
            if (!cuda_ok(cudaGetLastError(), "routed_moe mapped q8k gate/up launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);

            q8_k_quantize_sf37_kernel<<<dim3(midq_blocks, topk, 1u), 256>>>(
                    midq, (const float *)mid->ptr, expert_mid_dim, topk);
            if (!cuda_ok(cudaGetLastError(), "routed_moe mapped q8k mid quantize launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);

            moe_down_q2_asym_sum_q8k_kernel<<<(out_dim + 31u) / 32u, 256>>>(
                    (float *)out->ptr,
                    down_w,
                    midq,
                    (const int32_t *)selected->ptr,
                    topk, midq_blocks, out_dim, n_total_expert);
            if (!cuda_ok(cudaGetLastError(), "routed_moe mapped q8k direct down-sum launch")) {
                if (profile_moe) cuda_moe_profile_abort(prof_ev);
                return 0;
            }
            if (profile_moe) {
                (void)cudaEventRecord(prof_ev[4], 0);
                cuda_moe_profile_finish(prof_ev, 1, topk, in_dim, expert_mid_dim, out_dim);
            }
            return 1;
        }
    }

    if (profile_moe) (void)cudaEventRecord(prof_ev[1], 0);
    dim3 mgrid(expert_mid_dim, topk, 1);
    moe_gate_up_mid_q3_asym_kernel<<<mgrid, 256>>>(
            (float *)gate->ptr,
            (float *)up->ptr,
            (float *)mid->ptr,
            gate_w,
            up_w,
            (const float *)x->ptr,
            (const int32_t *)selected->ptr,
            (const float *)weights->ptr,
            topk, in_dim, expert_mid_dim, n_total_expert, clamp,
            write_gate_up);
    if (!cuda_ok(cudaGetLastError(), "routed_moe mapped gate/up launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) (void)cudaEventRecord(prof_ev[2], 0);

    if (direct_down_sum) {
        moe_down_q2_asym_sum_kernel<<<out_dim, 256>>>(
                (float *)out->ptr,
                down_w,
                (const float *)mid->ptr,
                (const int32_t *)selected->ptr,
                topk, expert_mid_dim, out_dim, n_total_expert);
        if (!cuda_ok(cudaGetLastError(), "routed_moe mapped direct down-sum launch")) {
            if (profile_moe) cuda_moe_profile_abort(prof_ev);
            return 0;
        }
        if (profile_moe) {
            (void)cudaEventRecord(prof_ev[4], 0);
            cuda_moe_profile_finish(prof_ev, 0, topk, in_dim, expert_mid_dim, out_dim);
        }
        return 1;
    }

    dim3 dgrid(out_dim, topk, 1);
    moe_down_q2_asym_kernel<<<dgrid, 256>>>(
            (float *)down->ptr,
            down_w,
            (const float *)mid->ptr,
            (const int32_t *)selected->ptr,
            topk, expert_mid_dim, out_dim, n_total_expert);
    if (!cuda_ok(cudaGetLastError(), "routed_moe mapped down launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) (void)cudaEventRecord(prof_ev[3], 0);

    moe_sum_slots_kernel<<<(out_dim + 255u) / 256u, 256>>>(
            (float *)out->ptr,
            (const float *)down->ptr,
            out_dim, topk);
    if (!cuda_ok(cudaGetLastError(), "routed_moe mapped sum launch")) {
        if (profile_moe) cuda_moe_profile_abort(prof_ev);
        return 0;
    }
    if (profile_moe) {
        (void)cudaEventRecord(prof_ev[4], 0);
        cuda_moe_profile_finish(prof_ev, 0, topk, in_dim, expert_mid_dim, out_dim);
    }
    return 1;
}
