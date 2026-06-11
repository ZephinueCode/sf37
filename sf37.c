#define _POSIX_C_SOURCE 200809L

#include "sf37.h"
#include "sf37_ops.h"
#include "sf37_quant.h"
#ifdef SF37_USE_CUDA
#include "sf37_cuda.h"
#endif

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

#define SF37_GGUF_MAGIC "GGUF"
#define SF37_GGUF_VERSION 3
#define SF37_MAX_DIMS 8
#define SF37_QK8 32
#define SF37_QK_K 256

#define SF37_MAIN_LAYERS 45
#define SF37_MTP_LAYERS 3
#define SF37_TOTAL_TEXT_LAYERS 48
#define SF37_VISION_LAYERS 47
#define SF37_EMBD 4096
#define SF37_VOCAB 128896
#define SF37_CTX 262144
#define SF37_FULL_Q_HEADS 64
#define SF37_SLIDING_Q_HEADS 96
#define SF37_KV_HEADS 8
#define SF37_HEAD_DIM 128
#define SF37_DENSE_FF 11264
#define SF37_EXPERTS 288
#define SF37_EXPERT_USED 8
#define SF37_EXPERT_FF 1280
#define SF37_VISION_WIDTH 1536
#define SF37_VISION_HEADS 16
#define SF37_VISION_HEAD_DIM 96
#define SF37_VISION_MLP 8960
#define SF37_VISION_PROJECTOR_IN (SF37_VISION_WIDTH * 4)
#define SF37_VISION_IMAGE 728
#define SF37_VISION_PATCH 14
#define SF37_VISION_GRID 52
#define SF37_VISION_PATCH_IMAGE 504
#define SF37_VISION_FEATURES 169
#define SF37_VISION_PATCH_FEATURES 81
#define SF37_VISION_ROPE_THETA 10000.0
#define SF37_VISION_LN_EPS 1.0e-5f
#define SF37_SLIDING_WINDOW 512
#define SF37_RMS_EPS 1.0e-5f
#define SF37_PI 3.14159265358979323846
#define SF37_SESSION_SNAPSHOT_MAGIC 0x37334653u
#define SF37_SESSION_SNAPSHOT_VERSION 2u
#define SF37_SESSION_SNAPSHOT_U32_FIELDS 11u
#define SF37_SESSION_PAYLOAD_MAGIC 0x37375053u
#define SF37_SESSION_PAYLOAD_VERSION 3u
#define SF37_SESSION_PAYLOAD_U32_FIELDS 15u
#define SF37_SESSION_PAYLOAD_KV_I8_GROUPED 1u
#define SF37_SESSION_PAYLOAD_KV_GROUP 64u
#define SF37_SESSION_IO_CHUNK (1024u * 1024u)
#define SF37_CUDA_SLIDING_CACHE_MAX 8192u

typedef enum {
    GGUF_VALUE_UINT8   = 0,
    GGUF_VALUE_INT8    = 1,
    GGUF_VALUE_UINT16  = 2,
    GGUF_VALUE_INT16   = 3,
    GGUF_VALUE_UINT32  = 4,
    GGUF_VALUE_INT32   = 5,
    GGUF_VALUE_FLOAT32 = 6,
    GGUF_VALUE_BOOL    = 7,
    GGUF_VALUE_STRING  = 8,
    GGUF_VALUE_ARRAY   = 9,
    GGUF_VALUE_UINT64  = 10,
    GGUF_VALUE_INT64   = 11,
    GGUF_VALUE_FLOAT64 = 12,
} gguf_value_type;

typedef enum {
    SF37_TENSOR_F32  = 0,
    SF37_TENSOR_F16  = 1,
    SF37_TENSOR_Q8_0 = 8,
    SF37_TENSOR_Q2_K = 10,
    SF37_TENSOR_Q3_K = 11,
    SF37_TENSOR_BF16 = 30,
} sf37_tensor_type;

typedef struct {
    const char *name;
    uint32_t block;
    uint32_t bytes;
} sf37_type_info;

typedef struct {
    const uint8_t *data;
    uint64_t size;
    uint64_t pos;
    char error[160];
} sf37_cursor;

typedef struct {
    const char *ptr;
    uint64_t len;
} sf37_str;

typedef struct {
    char *key;
    uint32_t type;
    uint64_t value_pos;
} sf37_kv;

typedef struct {
    char *name;
    uint32_t ndim;
    uint64_t dim[SF37_MAX_DIMS];
    uint32_t type;
    uint64_t rel_offset;
    uint64_t abs_offset;
    uint64_t nbytes;
} sf37_tensor;

typedef struct {
    int fd;
    uint8_t *map;
    uint64_t size;
    uint32_t version;
    uint64_t n_kv;
    uint64_t n_tensors;
    uint32_t alignment;
    uint64_t data_offset;
    sf37_kv *kv;
    sf37_tensor *tensors;
} sf37_gguf;

typedef struct {
    sf37_tensor *q_proj;
    sf37_tensor *k_proj;
    sf37_tensor *v_proj;
    sf37_tensor *o_proj;
    sf37_tensor *g_proj;
    sf37_tensor *q_norm;
    sf37_tensor *k_norm;
    sf37_tensor *input_norm;
    sf37_tensor *post_norm;
    sf37_tensor *mlp_gate;
    sf37_tensor *mlp_up;
    sf37_tensor *mlp_down;
    sf37_tensor *router_gate;
    sf37_tensor *router_bias;
    sf37_tensor *moe_gate;
    sf37_tensor *moe_up;
    sf37_tensor *moe_down;
    sf37_tensor *share_gate;
    sf37_tensor *share_up;
    sf37_tensor *share_down;
} sf37_layer_weights;

typedef struct {
    sf37_tensor *in_proj_weight;
    sf37_tensor *in_proj_bias;
    sf37_tensor *out_proj_weight;
    sf37_tensor *out_proj_bias;
    sf37_tensor *ln1_weight;
    sf37_tensor *ln1_bias;
    sf37_tensor *ln2_weight;
    sf37_tensor *ln2_bias;
    sf37_tensor *ls1_gamma;
    sf37_tensor *ls2_gamma;
    sf37_tensor *mlp_fc_weight;
    sf37_tensor *mlp_fc_bias;
    sf37_tensor *mlp_proj_weight;
    sf37_tensor *mlp_proj_bias;
} sf37_vision_block;

typedef struct {
    sf37_tensor *conv1_weight;
    sf37_tensor *ln_pre_weight;
    sf37_tensor *ln_pre_bias;
    sf37_tensor *positional_embedding;
    sf37_vision_block block[SF37_VISION_LAYERS];
    sf37_tensor *down1_weight;
    sf37_tensor *down1_bias;
    sf37_tensor *down2_weight;
    sf37_tensor *down2_bias;
    sf37_tensor *projector;
} sf37_vision_weights;

typedef struct {
    float *k;
    float *v;
    uint32_t n;
    uint32_t cap;
} sf37_layer_kv_cache;

typedef struct {
    sf37_layer_kv_cache layer[SF37_MAIN_LAYERS];
    uint32_t cap;
} sf37_kv_cache;

typedef struct {
    sf37_str key;
    int value;
    bool used;
} sf37_str_i32_entry;

typedef struct {
    sf37_str_i32_entry *entry;
    uint64_t cap;
    uint64_t used;
} sf37_str_i32_table;

typedef struct {
    char *text;
    uint64_t len;
    int id;
} sf37_added_token;

typedef struct {
    char **token;
    uint64_t *token_len;
    int n_vocab;
    int bos_id;
    int eos_id;
    int im_start_id;
    int im_end_id;
    int think_start_id;
    int think_end_id;
    int im_patch_id;
    sf37_added_token *added;
    int n_added;
    int cap_added;
    char **merge_keys;
    int n_merge_keys;
    int cap_merge_keys;
    sf37_str_i32_table token_to_id;
    sf37_str_i32_table merge_rank;
    bool ready;
} sf37_vocab;

struct sf37_engine {
    sf37_gguf gguf;
    char *model_path;
    char *tokenizer_path;
    sf37_backend backend;
    sf37_vocab vocab;
    sf37_tensor *embed_tokens;
    sf37_tensor *lm_head;
    sf37_tensor *final_norm;
    sf37_layer_weights layer[SF37_MAIN_LAYERS];
    sf37_vision_weights vision;
    bool cuda_ready;
    uint64_t type_count[42];
    uint64_t type_bytes[42];
    uint32_t bound_text_tensors;
    uint32_t bound_vision_tensors;
    uint32_t inactive_mtp_tensors;
    double timing_model_map_sec;
    double timing_preload_sec;
};

static const sf37_type_info sf37_type_table[] = {
    [SF37_TENSOR_F32]  = {"f32", 1, 4},
    [SF37_TENSOR_F16]  = {"f16", 1, 2},
    [SF37_TENSOR_Q8_0] = {"q8_0", SF37_QK8, 34},
    [SF37_TENSOR_Q2_K] = {"q2_K", SF37_QK_K, 84},
    [SF37_TENSOR_Q3_K] = {"q3_K", SF37_QK_K, 110},
    [SF37_TENSOR_BF16] = {"bf16", 1, 2},
};

static void *xmalloc(size_t size) {
    void *p = malloc(size ? size : 1);
    if (!p) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xcalloc(size_t n, size_t size) {
    void *p = calloc(n, size);
    if (!p) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    return p;
}

static void *xrealloc(void *ptr, size_t size) {
    void *p = realloc(ptr, size);
    if (!p && size) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    return p;
}

static char *xstrndup(const char *s, uint64_t len) {
    if (len > SIZE_MAX - 1) {
        fprintf(stderr, "sf37: string too large\n");
        exit(1);
    }
    char *p = malloc((size_t)len + 1);
    if (!p) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    memcpy(p, s, (size_t)len);
    p[len] = '\0';
    return p;
}

static char *xstrdup(const char *s) {
    return xstrndup(s, strlen(s));
}

static bool path_exists(const char *path) {
    struct stat st;
    return path && stat(path, &st) == 0;
}

static char *path_dirname_dup(const char *path) {
    if (!path || !path[0]) return xstrdup(".");
    const char *slash = strrchr(path, '/');
    if (!slash) return xstrdup(".");
    if (slash == path) return xstrndup(path, 1);
    return xstrndup(path, (uint64_t)(slash - path));
}

static const char *type_name(uint32_t type) {
    if (type < sizeof(sf37_type_table) / sizeof(sf37_type_table[0]) &&
        sf37_type_table[type].name) return sf37_type_table[type].name;
    return "unknown";
}

static bool type_supported(uint32_t type) {
    return type < sizeof(sf37_type_table) / sizeof(sf37_type_table[0]) &&
           sf37_type_table[type].name != NULL;
}

static uint64_t checked_mul_u64(uint64_t a, uint64_t b, const char *what) {
    if (a != 0 && b > UINT64_MAX / a) {
        fprintf(stderr, "sf37: overflow computing %s\n", what);
        exit(1);
    }
    return a * b;
}

static uint64_t align_u64(uint64_t x, uint64_t n) {
    if (n == 0) return x;
    uint64_t r = x % n;
    return r ? x + (n - r) : x;
}

static void cursor_fail(sf37_cursor *c, const char *msg) {
    if (c->error[0] == '\0') {
        snprintf(c->error, sizeof(c->error), "%s at byte %" PRIu64, msg, c->pos);
    }
}

static bool cursor_need(sf37_cursor *c, uint64_t n) {
    if (n > c->size || c->pos > c->size - n) {
        cursor_fail(c, "truncated GGUF");
        return false;
    }
    return true;
}

static bool cursor_u32(sf37_cursor *c, uint32_t *out) {
    if (!cursor_need(c, 4)) return false;
    uint32_t v;
    memcpy(&v, c->data + c->pos, 4);
    c->pos += 4;
    *out = v;
    return true;
}

static bool cursor_u64(sf37_cursor *c, uint64_t *out) {
    if (!cursor_need(c, 8)) return false;
    uint64_t v;
    memcpy(&v, c->data + c->pos, 8);
    c->pos += 8;
    *out = v;
    return true;
}

static bool cursor_skip(sf37_cursor *c, uint64_t n) {
    if (!cursor_need(c, n)) return false;
    c->pos += n;
    return true;
}

static bool cursor_string(sf37_cursor *c, sf37_str *out) {
    uint64_t len = 0;
    if (!cursor_u64(c, &len)) return false;
    if (!cursor_need(c, len)) return false;
    out->ptr = (const char *)(c->data + c->pos);
    out->len = len;
    c->pos += len;
    return true;
}

static uint64_t gguf_value_size(uint32_t type) {
    switch (type) {
    case GGUF_VALUE_UINT8:
    case GGUF_VALUE_INT8:
    case GGUF_VALUE_BOOL:
        return 1;
    case GGUF_VALUE_UINT16:
    case GGUF_VALUE_INT16:
        return 2;
    case GGUF_VALUE_UINT32:
    case GGUF_VALUE_INT32:
    case GGUF_VALUE_FLOAT32:
        return 4;
    case GGUF_VALUE_UINT64:
    case GGUF_VALUE_INT64:
    case GGUF_VALUE_FLOAT64:
        return 8;
    default:
        return 0;
    }
}

static bool skip_value(sf37_cursor *c, uint32_t type, int depth) {
    if (depth > 2) {
        cursor_fail(c, "nested GGUF array is too deep");
        return false;
    }
    if (type == GGUF_VALUE_STRING) {
        sf37_str s;
        return cursor_string(c, &s);
    }
    if (type == GGUF_VALUE_ARRAY) {
        uint32_t elem_type = 0;
        uint64_t n = 0;
        if (!cursor_u32(c, &elem_type) || !cursor_u64(c, &n)) return false;
        uint64_t elem_size = gguf_value_size(elem_type);
        if (elem_size) return cursor_skip(c, checked_mul_u64(n, elem_size, "metadata array"));
        for (uint64_t i = 0; i < n; i++) {
            if (!skip_value(c, elem_type, depth + 1)) return false;
        }
        return true;
    }
    uint64_t size = gguf_value_size(type);
    if (!size) {
        cursor_fail(c, "unknown GGUF metadata type");
        return false;
    }
    return cursor_skip(c, size);
}

static uint64_t tensor_nbytes(uint32_t type, const uint64_t *dim, uint32_t ndim) {
    if (!type_supported(type) || ndim == 0 || ndim > SF37_MAX_DIMS) return 0;
    uint64_t elems = 1;
    for (uint32_t i = 0; i < ndim; i++) elems = checked_mul_u64(elems, dim[i], "tensor elements");
    const sf37_type_info *ti = &sf37_type_table[type];
    if (ti->block == 1) return checked_mul_u64(elems, ti->bytes, "tensor bytes");
    if (dim[0] % ti->block != 0 || elems % ti->block != 0) return 0;
    return checked_mul_u64(elems / ti->block, ti->bytes, "quant tensor bytes");
}

static sf37_kv *find_kv(const sf37_gguf *g, const char *key) {
    for (uint64_t i = 0; i < g->n_kv; i++) {
        if (strcmp(g->kv[i].key, key) == 0) return &g->kv[i];
    }
    return NULL;
}

static bool kv_string(const sf37_gguf *g, const char *key, sf37_str *out) {
    sf37_kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_VALUE_STRING) return false;
    sf37_cursor c = { .data = g->map, .size = g->size, .pos = kv->value_pos };
    return cursor_string(&c, out);
}

static bool kv_u32(const sf37_gguf *g, const char *key, uint32_t *out) {
    sf37_kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_VALUE_UINT32) return false;
    sf37_cursor c = { .data = g->map, .size = g->size, .pos = kv->value_pos };
    return cursor_u32(&c, out);
}

static bool kv_bool(const sf37_gguf *g, const char *key, bool *out) {
    sf37_kv *kv = find_kv(g, key);
    if (!kv || kv->type != GGUF_VALUE_BOOL || kv->value_pos >= g->size) return false;
    *out = g->map[kv->value_pos] != 0;
    return true;
}

static bool str_eq(sf37_str s, const char *lit) {
    uint64_t n = strlen(lit);
    return s.len == n && memcmp(s.ptr, lit, (size_t)n) == 0;
}

static int gguf_open(sf37_gguf *g, const char *path, char *err, size_t errlen) {
    memset(g, 0, sizeof(*g));
    g->fd = -1;
    g->alignment = 32;

    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return -1;
    }
    struct stat st;
    if (fstat(fd, &st) == -1) {
        snprintf(err, errlen, "stat %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }
    if (st.st_size < 32) {
        snprintf(err, errlen, "model file is too small to be GGUF");
        close(fd);
        return -1;
    }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) {
        snprintf(err, errlen, "mmap %s: %s", path, strerror(errno));
        close(fd);
        return -1;
    }

    g->fd = fd;
    g->map = map;
    g->size = (uint64_t)st.st_size;

    sf37_cursor c = { .data = g->map, .size = g->size, .pos = 0 };
    if (!cursor_need(&c, 4)) goto parse_error;
    if (memcmp(c.data, SF37_GGUF_MAGIC, 4) != 0) {
        snprintf(err, errlen, "model is not a GGUF file");
        return -1;
    }
    c.pos = 4;
    if (!cursor_u32(&c, &g->version) ||
        !cursor_u64(&c, &g->n_tensors) ||
        !cursor_u64(&c, &g->n_kv)) goto parse_error;
    if (g->version != SF37_GGUF_VERSION) {
        snprintf(err, errlen, "unsupported GGUF version %u", g->version);
        return -1;
    }

    g->kv = xcalloc((size_t)g->n_kv, sizeof(g->kv[0]));
    for (uint64_t i = 0; i < g->n_kv; i++) {
        sf37_str key = {0};
        if (!cursor_string(&c, &key) || !cursor_u32(&c, &g->kv[i].type)) goto parse_error;
        g->kv[i].key = xstrndup(key.ptr, key.len);
        g->kv[i].value_pos = c.pos;
        if (strcmp(g->kv[i].key, "general.alignment") == 0 &&
            g->kv[i].type == GGUF_VALUE_UINT32) {
            sf37_cursor tmp = c;
            uint32_t alignment = 0;
            if (cursor_u32(&tmp, &alignment) && alignment) g->alignment = alignment;
        }
        if (!skip_value(&c, g->kv[i].type, 0)) goto parse_error;
    }

    g->tensors = xcalloc((size_t)g->n_tensors, sizeof(g->tensors[0]));
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        sf37_tensor *t = &g->tensors[i];
        sf37_str name = {0};
        if (!cursor_string(&c, &name) || !cursor_u32(&c, &t->ndim)) goto parse_error;
        if (t->ndim == 0 || t->ndim > SF37_MAX_DIMS) {
            snprintf(err, errlen, "tensor %.*s has unsupported ndim %u",
                     (int)name.len, name.ptr, t->ndim);
            return -1;
        }
        t->name = xstrndup(name.ptr, name.len);
        for (uint32_t d = 0; d < t->ndim; d++) {
            if (!cursor_u64(&c, &t->dim[d])) goto parse_error;
        }
        if (!cursor_u32(&c, &t->type) || !cursor_u64(&c, &t->rel_offset)) goto parse_error;
        t->nbytes = tensor_nbytes(t->type, t->dim, t->ndim);
        if (!t->nbytes) {
            snprintf(err, errlen, "tensor %s has unsupported type/layout %u", t->name, t->type);
            return -1;
        }
    }

    g->data_offset = align_u64(c.pos, g->alignment);
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        sf37_tensor *t = &g->tensors[i];
        t->abs_offset = g->data_offset + t->rel_offset;
        if (t->abs_offset > g->size || t->nbytes > g->size - t->abs_offset) {
            snprintf(err, errlen, "tensor %s points outside GGUF file", t->name);
            return -1;
        }
    }
    return 0;

parse_error:
    snprintf(err, errlen, "%s", c.error[0] ? c.error : "failed to parse GGUF");
    return -1;
}

static void gguf_close(sf37_gguf *g) {
    if (!g) return;
    if (g->kv) {
        for (uint64_t i = 0; i < g->n_kv; i++) free(g->kv[i].key);
        free(g->kv);
    }
    if (g->tensors) {
        for (uint64_t i = 0; i < g->n_tensors; i++) free(g->tensors[i].name);
        free(g->tensors);
    }
    if (g->map && g->map != MAP_FAILED) munmap(g->map, (size_t)g->size);
    if (g->fd != -1) close(g->fd);
    memset(g, 0, sizeof(*g));
    g->fd = -1;
}

/* =========================================================================
 * tokenizer.json BPE runtime.
 * =========================================================================
 *
 * SF37 GGUF files currently keep model tensors and SF37 layout metadata only.
 * The tokenizer is loaded from --tokenizer/tokenizer.json, or from the GGUF
 * directory when --tokenizer is not specified. The implementation mirrors the
 * DS4 byte-level BPE path, with Step-3.7's added-token chat markers.
 */

typedef enum {
    SF37_JSON_OBJECT,
    SF37_JSON_ARRAY,
    SF37_JSON_STRING,
    SF37_JSON_PRIMITIVE,
} sf37_json_type;

typedef struct {
    sf37_json_type type;
    int start;
    int end;
    int parent;
    int size;
} sf37_json_tok;

typedef struct {
    sf37_json_tok *v;
    int len;
    int cap;
    const char *js;
    int js_len;
} sf37_json_doc;

static uint64_t sf37_hash_bytes(const void *ptr, uint64_t len) {
    const uint8_t *p = ptr;
    uint64_t h = 1469598103934665603ull;
    for (uint64_t i = 0; i < len; i++) {
        h ^= p[i];
        h *= 1099511628211ull;
    }
    return h;
}

static bool sf37_str_bytes_eq(sf37_str a, const char *ptr, uint64_t len) {
    return a.len == len && memcmp(a.ptr, ptr, (size_t)len) == 0;
}

static bool sf37_str_eq2(sf37_str a, sf37_str b) {
    return a.len == b.len && memcmp(a.ptr, b.ptr, (size_t)a.len) == 0;
}

static uint64_t sf37_next_pow2(uint64_t n) {
    uint64_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static void table_init(sf37_str_i32_table *t, uint64_t expected) {
    t->cap = sf37_next_pow2(expected * 2u + 16u);
    t->used = 0;
    t->entry = xcalloc((size_t)t->cap, sizeof(t->entry[0]));
}

static void table_free(sf37_str_i32_table *t) {
    free(t->entry);
    memset(t, 0, sizeof(*t));
}

static void table_put(sf37_str_i32_table *t, sf37_str key, int value) {
    if (t->cap == 0) table_init(t, 16);
    uint64_t mask = t->cap - 1u;
    uint64_t i = sf37_hash_bytes(key.ptr, key.len) & mask;
    while (t->entry[i].used) {
        if (sf37_str_eq2(t->entry[i].key, key)) {
            t->entry[i].value = value;
            return;
        }
        i = (i + 1u) & mask;
    }
    t->entry[i].used = true;
    t->entry[i].key = key;
    t->entry[i].value = value;
    t->used++;
}

static bool table_get(const sf37_str_i32_table *t, const char *ptr, uint64_t len, int *value) {
    if (!t || t->cap == 0) return false;
    uint64_t mask = t->cap - 1u;
    uint64_t i = sf37_hash_bytes(ptr, len) & mask;
    while (t->entry[i].used) {
        if (sf37_str_bytes_eq(t->entry[i].key, ptr, len)) {
            if (value) *value = t->entry[i].value;
            return true;
        }
        i = (i + 1u) & mask;
    }
    return false;
}

void sf37_tokens_push(sf37_tokens *tv, int token) {
    if (!tv) return;
    if (tv->len == tv->cap) {
        tv->cap = tv->cap ? tv->cap * 2 : 64;
        tv->v = xrealloc(tv->v, (size_t)tv->cap * sizeof(tv->v[0]));
    }
    tv->v[tv->len++] = token;
}

void sf37_tokens_free(sf37_tokens *tv) {
    if (!tv) return;
    free(tv->v);
    memset(tv, 0, sizeof(*tv));
}

void sf37_tokens_copy(sf37_tokens *dst, const sf37_tokens *src) {
    if (!dst || !src) return;
    dst->len = 0;
    for (int i = 0; i < src->len; i++) sf37_tokens_push(dst, src->v[i]);
}

bool sf37_tokens_starts_with(const sf37_tokens *tokens, const sf37_tokens *prefix) {
    if (!tokens || !prefix || prefix->len > tokens->len) return false;
    for (int i = 0; i < prefix->len; i++) {
        if (tokens->v[i] != prefix->v[i]) return false;
    }
    return true;
}

static int json_add(sf37_json_doc *d, sf37_json_type type, int start, int end, int parent) {
    if (d->len == d->cap) {
        d->cap = d->cap ? d->cap * 2 : 4096;
        d->v = xrealloc(d->v, (size_t)d->cap * sizeof(d->v[0]));
    }
    const int id = d->len++;
    d->v[id] = (sf37_json_tok){ .type = type, .start = start, .end = end, .parent = parent, .size = 0 };
    if (parent >= 0) d->v[parent].size++;
    return id;
}

static int json_parse_text(sf37_json_doc *d, const char *js, size_t len, char *err, size_t errlen) {
    memset(d, 0, sizeof(*d));
    d->js = js;
    d->js_len = (int)len;
    int parent = -1;
    for (int i = 0; i < (int)len; i++) {
        unsigned char c = (unsigned char)js[i];
        if (isspace(c) || c == ':' || c == ',') continue;
        if (c == '{' || c == '[') {
            parent = json_add(d, c == '{' ? SF37_JSON_OBJECT : SF37_JSON_ARRAY, i, -1, parent);
            continue;
        }
        if (c == '}' || c == ']') {
            if (parent < 0) {
                snprintf(err, errlen, "bad JSON: unmatched close");
                return -1;
            }
            d->v[parent].end = i + 1;
            parent = d->v[parent].parent;
            continue;
        }
        if (c == '"') {
            int start = i + 1;
            bool esc = false;
            i++;
            for (; i < (int)len; i++) {
                if (esc) {
                    esc = false;
                } else if (js[i] == '\\') {
                    esc = true;
                } else if (js[i] == '"') {
                    break;
                }
            }
            if (i >= (int)len) {
                snprintf(err, errlen, "bad JSON: unterminated string");
                return -1;
            }
            json_add(d, SF37_JSON_STRING, start, i, parent);
            continue;
        }
        int start = i;
        while (i < (int)len && !isspace((unsigned char)js[i]) &&
               js[i] != ',' && js[i] != ']' && js[i] != '}') {
            i++;
        }
        json_add(d, SF37_JSON_PRIMITIVE, start, i, parent);
        i--;
    }
    if (parent != -1) {
        snprintf(err, errlen, "bad JSON: unterminated object/array");
        return -1;
    }
    if (d->len == 0 || d->v[0].type != SF37_JSON_OBJECT) {
        snprintf(err, errlen, "bad JSON: root must be an object");
        return -1;
    }
    return 0;
}

static void json_free(sf37_json_doc *d) {
    free(d->v);
    memset(d, 0, sizeof(*d));
}

static bool json_tok_eq(const sf37_json_doc *d, int tok, const char *s) {
    const sf37_json_tok *t = &d->v[tok];
    const int n = t->end - t->start;
    return t->type == SF37_JSON_STRING &&
           (int)strlen(s) == n &&
           memcmp(d->js + t->start, s, (size_t)n) == 0;
}

static bool json_is_descendant(const sf37_json_doc *d, int tok, int parent) {
    for (int p = d->v[tok].parent; p >= 0; p = d->v[p].parent) {
        if (p == parent) return true;
    }
    return false;
}

static int json_skip(const sf37_json_doc *d, int tok) {
    int i = tok + 1;
    while (i < d->len && json_is_descendant(d, i, tok)) i++;
    return i;
}

static int json_obj_get(const sf37_json_doc *d, int obj, const char *key) {
    if (obj < 0 || d->v[obj].type != SF37_JSON_OBJECT) return -1;
    for (int i = obj + 1; i < d->len && d->v[i].parent == obj;) {
        int k = i;
        int v = i + 1;
        if (v >= d->len || d->v[v].parent != obj) return -1;
        if (json_tok_eq(d, k, key)) return v;
        i = json_skip(d, v);
    }
    return -1;
}

static int64_t json_i64(const sf37_json_doc *d, int tok) {
    char tmp[64];
    const int n = d->v[tok].end - d->v[tok].start;
    if (tok < 0 || n <= 0 || n >= (int)sizeof(tmp)) return 0;
    memcpy(tmp, d->js + d->v[tok].start, (size_t)n);
    tmp[n] = '\0';
    return strtoll(tmp, NULL, 10);
}

static int hex_val(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static void utf8_put(char **p, uint32_t cp) {
    if (cp <= 0x7f) {
        *(*p)++ = (char)cp;
    } else if (cp <= 0x7ff) {
        *(*p)++ = (char)(0xc0 | (cp >> 6));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else if (cp <= 0xffff) {
        *(*p)++ = (char)(0xe0 | (cp >> 12));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    } else {
        *(*p)++ = (char)(0xf0 | (cp >> 18));
        *(*p)++ = (char)(0x80 | ((cp >> 12) & 0x3f));
        *(*p)++ = (char)(0x80 | ((cp >> 6) & 0x3f));
        *(*p)++ = (char)(0x80 | (cp & 0x3f));
    }
}

static bool json_read_u16_escape(const char *s, int n, int *i, uint32_t *cp) {
    if (*i + 4 > n) return false;
    uint32_t v = 0;
    for (int k = 0; k < 4; k++) {
        int h = hex_val(s[*i + k]);
        if (h < 0) return false;
        v = (v << 4) | (uint32_t)h;
    }
    *i += 4;
    *cp = v;
    return true;
}

static char *json_strdup_decoded(const sf37_json_doc *d, int tok, uint64_t *len_out) {
    const sf37_json_tok *t = &d->v[tok];
    const char *s = d->js + t->start;
    const int n = t->end - t->start;
    char *out = xmalloc((size_t)n * 4u + 1u);
    char *p = out;
    for (int i = 0; i < n; i++) {
        unsigned char c = (unsigned char)s[i];
        if (c != '\\') {
            *p++ = (char)c;
            continue;
        }
        if (++i >= n) break;
        switch (s[i]) {
        case '"': *p++ = '"'; break;
        case '\\': *p++ = '\\'; break;
        case '/': *p++ = '/'; break;
        case 'b': *p++ = '\b'; break;
        case 'f': *p++ = '\f'; break;
        case 'n': *p++ = '\n'; break;
        case 'r': *p++ = '\r'; break;
        case 't': *p++ = '\t'; break;
        case 'u': {
            i++;
            uint32_t cp = 0;
            if (!json_read_u16_escape(s, n, &i, &cp)) {
                i = n;
                break;
            }
            i--;
            if (cp >= 0xd800 && cp <= 0xdbff &&
                i + 6 < n && s[i + 1] == '\\' && s[i + 2] == 'u') {
                int j = i + 3;
                uint32_t lo = 0;
                if (json_read_u16_escape(s, n, &j, &lo) &&
                    lo >= 0xdc00 && lo <= 0xdfff) {
                    cp = 0x10000u + ((cp - 0xd800u) << 10) + (lo - 0xdc00u);
                    i = j - 1;
                }
            }
            utf8_put(&p, cp);
            break;
        }
        default:
            *p++ = s[i];
            break;
        }
    }
    *p = '\0';
    if (len_out) *len_out = (uint64_t)(p - out);
    return out;
}

static char *read_text_file(const char *path, size_t *len_out, char *err, size_t errlen) {
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        snprintf(err, errlen, "open %s: %s", path, strerror(errno));
        return NULL;
    }
    if (fseek(fp, 0, SEEK_END) != 0) {
        snprintf(err, errlen, "seek %s failed", path);
        fclose(fp);
        return NULL;
    }
    long n = ftell(fp);
    if (n < 0) {
        snprintf(err, errlen, "tell %s failed", path);
        fclose(fp);
        return NULL;
    }
    if (fseek(fp, 0, SEEK_SET) != 0) {
        snprintf(err, errlen, "rewind %s failed", path);
        fclose(fp);
        return NULL;
    }
    char *buf = xmalloc((size_t)n + 1u);
    if (n > 0 && fread(buf, 1, (size_t)n, fp) != (size_t)n) {
        snprintf(err, errlen, "read %s failed", path);
        free(buf);
        fclose(fp);
        return NULL;
    }
    fclose(fp);
    buf[n] = '\0';
    if (len_out) *len_out = (size_t)n;
    return buf;
}

static void vocab_merge_key_push(sf37_vocab *vocab, char *key) {
    if (vocab->n_merge_keys == vocab->cap_merge_keys) {
        vocab->cap_merge_keys = vocab->cap_merge_keys ? vocab->cap_merge_keys * 2 : 1024;
        vocab->merge_keys = xrealloc(vocab->merge_keys,
                                     (size_t)vocab->cap_merge_keys * sizeof(vocab->merge_keys[0]));
    }
    vocab->merge_keys[vocab->n_merge_keys++] = key;
}

static void vocab_added_push(sf37_vocab *vocab, int id, char *text, uint64_t len) {
    if (id < 0) {
        free(text);
        return;
    }
    if (vocab->n_added == vocab->cap_added) {
        vocab->cap_added = vocab->cap_added ? vocab->cap_added * 2 : 128;
        vocab->added = xrealloc(vocab->added, (size_t)vocab->cap_added * sizeof(vocab->added[0]));
    }
    vocab->added[vocab->n_added++] = (sf37_added_token){ .text = text, .len = len, .id = id };
}

static int vocab_lookup_soft(const sf37_vocab *vocab, const char *text) {
    int token = -1;
    table_get(&vocab->token_to_id, text, strlen(text), &token);
    return token;
}

static int vocab_lookup_required(const sf37_vocab *vocab, const char *text) {
    int token = vocab_lookup_soft(vocab, text);
    if (token < 0) {
        fprintf(stderr, "sf37: required tokenizer token is missing: %s\n", text);
        exit(1);
    }
    return token;
}

static void vocab_free(sf37_vocab *vocab) {
    if (!vocab) return;
    if (vocab->token) {
        for (int i = 0; i < vocab->n_vocab; i++) free(vocab->token[i]);
        free(vocab->token);
    }
    free(vocab->token_len);
    for (int i = 0; i < vocab->n_added; i++) free(vocab->added[i].text);
    free(vocab->added);
    for (int i = 0; i < vocab->n_merge_keys; i++) free(vocab->merge_keys[i]);
    free(vocab->merge_keys);
    table_free(&vocab->token_to_id);
    table_free(&vocab->merge_rank);
    memset(vocab, 0, sizeof(*vocab));
}

static int vocab_load_from_tokenizer_json(sf37_vocab *vocab, const char *path, char *err, size_t errlen) {
    memset(vocab, 0, sizeof(*vocab));
    size_t len = 0;
    char *text = read_text_file(path, &len, err, errlen);
    if (!text) return -1;

    sf37_json_doc d;
    if (json_parse_text(&d, text, len, err, errlen) != 0) {
        free(text);
        return -1;
    }

    int model = json_obj_get(&d, 0, "model");
    int vocab_tok = json_obj_get(&d, model, "vocab");
    int merges_tok = json_obj_get(&d, model, "merges");
    if (model < 0 || vocab_tok < 0 || merges_tok < 0 ||
        d.v[vocab_tok].type != SF37_JSON_OBJECT ||
        d.v[merges_tok].type != SF37_JSON_ARRAY) {
        snprintf(err, errlen, "tokenizer.json missing model.vocab/model.merges");
        json_free(&d);
        free(text);
        return -1;
    }

    int max_id = -1;
    for (int i = vocab_tok + 1; i < d.len && d.v[i].parent == vocab_tok;) {
        int v = i + 1;
        int id = (int)json_i64(&d, v);
        if (id > max_id) max_id = id;
        i = json_skip(&d, v);
    }
    int added_tok = json_obj_get(&d, 0, "added_tokens");
    if (added_tok >= 0 && d.v[added_tok].type == SF37_JSON_ARRAY) {
        for (int i = added_tok + 1; i < d.len && d.v[i].parent == added_tok; i = json_skip(&d, i)) {
            if (d.v[i].type != SF37_JSON_OBJECT) continue;
            int id_tok = json_obj_get(&d, i, "id");
            if (id_tok >= 0) {
                int id = (int)json_i64(&d, id_tok);
                if (id > max_id) max_id = id;
            }
        }
    }
    if (max_id < 0 || max_id > 10000000) {
        snprintf(err, errlen, "tokenizer vocab id range is invalid");
        json_free(&d);
        free(text);
        return -1;
    }

    vocab->n_vocab = max_id + 1;
    vocab->token = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token[0]));
    vocab->token_len = xcalloc((size_t)vocab->n_vocab, sizeof(vocab->token_len[0]));
    table_init(&vocab->token_to_id, (uint64_t)vocab->n_vocab);

    for (int i = vocab_tok + 1; i < d.len && d.v[i].parent == vocab_tok;) {
        int k = i;
        int v = i + 1;
        uint64_t slen = 0;
        char *s = json_strdup_decoded(&d, k, &slen);
        int id = (int)json_i64(&d, v);
        if (id >= 0 && id < vocab->n_vocab) {
            free(vocab->token[id]);
            vocab->token[id] = s;
            vocab->token_len[id] = slen;
            table_put(&vocab->token_to_id, (sf37_str){s, slen}, id);
        } else {
            free(s);
        }
        i = json_skip(&d, v);
    }

    if (added_tok >= 0 && d.v[added_tok].type == SF37_JSON_ARRAY) {
        for (int i = added_tok + 1; i < d.len && d.v[i].parent == added_tok; i = json_skip(&d, i)) {
            if (d.v[i].type != SF37_JSON_OBJECT) continue;
            int id_tok = json_obj_get(&d, i, "id");
            int content_tok = json_obj_get(&d, i, "content");
            if (id_tok < 0 || content_tok < 0 || d.v[content_tok].type != SF37_JSON_STRING) continue;
            uint64_t slen = 0;
            char *s = json_strdup_decoded(&d, content_tok, &slen);
            int id = (int)json_i64(&d, id_tok);
            if (id >= 0 && id < vocab->n_vocab) {
                if (!vocab->token[id]) {
                    vocab->token[id] = xstrndup(s, slen);
                    vocab->token_len[id] = slen;
                    table_put(&vocab->token_to_id, (sf37_str){vocab->token[id], slen}, id);
                }
                vocab_added_push(vocab, id, s, slen);
            } else {
                free(s);
            }
        }
    }

    uint64_t merge_count = 0;
    for (int i = merges_tok + 1; i < d.len && d.v[i].parent == merges_tok; i = json_skip(&d, i)) {
        merge_count++;
    }
    table_init(&vocab->merge_rank, merge_count);
    int rank = 0;
    for (int i = merges_tok + 1; i < d.len && d.v[i].parent == merges_tok; i = json_skip(&d, i), rank++) {
        char *key = NULL;
        uint64_t klen = 0;
        if (d.v[i].type == SF37_JSON_STRING) {
            key = json_strdup_decoded(&d, i, &klen);
        } else if (d.v[i].type == SF37_JSON_ARRAY) {
            int a = -1, b = -1;
            for (int j = i + 1; j < d.len && d.v[j].parent == i; j = json_skip(&d, j)) {
                if (a < 0) a = j;
                else if (b < 0) { b = j; break; }
            }
            if (a >= 0 && b >= 0) {
                uint64_t alen = 0, blen = 0;
                char *as = json_strdup_decoded(&d, a, &alen);
                char *bs = json_strdup_decoded(&d, b, &blen);
                klen = alen + 1u + blen;
                key = xmalloc((size_t)klen + 1u);
                memcpy(key, as, (size_t)alen);
                key[alen] = ' ';
                memcpy(key + alen + 1u, bs, (size_t)blen);
                key[klen] = '\0';
                free(as);
                free(bs);
            }
        }
        if (key) {
            table_put(&vocab->merge_rank, (sf37_str){key, klen}, rank);
            vocab_merge_key_push(vocab, key);
        }
    }

    vocab->bos_id = vocab_lookup_required(vocab, "<｜begin▁of▁sentence｜>");
    vocab->eos_id = vocab_lookup_required(vocab, "<|im_end|>");
    vocab->im_start_id = vocab_lookup_required(vocab, "<|im_start|>");
    vocab->im_end_id = vocab_lookup_required(vocab, "<|im_end|>");
    vocab->think_start_id = vocab_lookup_soft(vocab, "<think>");
    vocab->think_end_id = vocab_lookup_soft(vocab, "</think>");
    vocab->im_patch_id = vocab_lookup_soft(vocab, "<im_patch>");
    vocab->ready = true;

    json_free(&d);
    free(text);
    return 0;
}

static uint32_t gpt2_byte_to_codepoint(uint8_t b) {
    if ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174)) return b;
    uint32_t n = 0;
    for (uint32_t x = 0; x < 256; x++) {
        if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) continue;
        if (x == b) return 256u + n;
        n++;
    }
    return b;
}

static int gpt2_codepoint_to_byte(uint32_t cp) {
    if ((cp >= 33 && cp <= 126) || (cp >= 161 && cp <= 172) ||
        (cp >= 174 && cp <= 255)) {
        return (int)cp;
    }
    if (cp >= 256) {
        uint32_t n = 0;
        for (uint32_t x = 0; x < 256; x++) {
            if ((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174)) continue;
            if (n == cp - 256u) return (int)x;
            n++;
        }
    }
    return -1;
}

static int utf8_len_from_first_byte(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xe0) == 0xc0) return 2;
    if ((c & 0xf0) == 0xe0) return 3;
    if ((c & 0xf8) == 0xf0) return 4;
    return 1;
}

static uint32_t utf8_peek_one(const char *s, uint64_t len, uint64_t pos, uint64_t *next) {
    const uint8_t c0 = (uint8_t)s[pos];
    int n = utf8_len_from_first_byte(c0);
    if (pos + (uint64_t)n > len) n = 1;
    *next = pos + (uint64_t)n;
    if (n == 1) return c0;
    if (n == 2) {
        return ((uint32_t)(c0 & 0x1f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f));
    }
    if (n == 3) {
        return ((uint32_t)(c0 & 0x0f) << 12) |
               ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 6) |
               ((uint32_t)((uint8_t)s[pos + 2] & 0x3f));
    }
    return ((uint32_t)(c0 & 0x07) << 18) |
           ((uint32_t)((uint8_t)s[pos + 1] & 0x3f) << 12) |
           ((uint32_t)((uint8_t)s[pos + 2] & 0x3f) << 6) |
           ((uint32_t)((uint8_t)s[pos + 3] & 0x3f));
}

static uint64_t next_utf8_char(const char *s, uint64_t len, uint64_t pos) {
    int n = utf8_len_from_first_byte((uint8_t)s[pos]);
    if (pos + (uint64_t)n > len) n = 1;
    return pos + (uint64_t)n;
}

static char *byte_encode(sf37_str in, uint64_t *out_len) {
    char *out = xmalloc((size_t)in.len * 4u + 1u);
    char *p = out;
    for (uint64_t i = 0; i < in.len; i++) {
        utf8_put(&p, gpt2_byte_to_codepoint((uint8_t)in.ptr[i]));
    }
    *p = '\0';
    *out_len = (uint64_t)(p - out);
    return out;
}

typedef struct {
    char *ptr;
    uint64_t len;
} owned_str;

static owned_str owned_copy(const char *ptr, uint64_t len) {
    owned_str s;
    s.ptr = xmalloc((size_t)len);
    memcpy(s.ptr, ptr, (size_t)len);
    s.len = len;
    return s;
}

static int bpe_rank(const sf37_vocab *vocab, const owned_str *a, const owned_str *b) {
    uint64_t len = a->len + 1u + b->len;
    char stack[512];
    char *buf = len <= sizeof(stack) ? stack : xmalloc((size_t)len);
    memcpy(buf, a->ptr, (size_t)a->len);
    buf[a->len] = ' ';
    memcpy(buf + a->len + 1u, b->ptr, (size_t)b->len);
    int rank = -1;
    table_get(&vocab->merge_rank, buf, len, &rank);
    if (buf != stack) free(buf);
    return rank;
}

static void bpe_emit_piece(const sf37_vocab *vocab, sf37_str raw_piece, sf37_tokens *out) {
    uint64_t encoded_len = 0;
    char *encoded = byte_encode(raw_piece, &encoded_len);
    int n_sym = 0;
    int cap_sym = 32;
    owned_str *sym = xcalloc((size_t)cap_sym, sizeof(sym[0]));
    for (uint64_t off = 0; off < encoded_len;) {
        int n = utf8_len_from_first_byte((uint8_t)encoded[off]);
        if (off + (uint64_t)n > encoded_len) n = 1;
        if (n_sym == cap_sym) {
            cap_sym *= 2;
            sym = xrealloc(sym, (size_t)cap_sym * sizeof(sym[0]));
        }
        sym[n_sym++] = owned_copy(encoded + off, (uint64_t)n);
        off += (uint64_t)n;
    }
    for (;;) {
        int best_i = -1;
        int best_rank = INT_MAX;
        for (int i = 0; i + 1 < n_sym; i++) {
            int rank = bpe_rank(vocab, &sym[i], &sym[i + 1]);
            if (rank >= 0 && rank < best_rank) {
                best_rank = rank;
                best_i = i;
            }
        }
        if (best_i < 0) break;
        owned_str merged;
        merged.len = sym[best_i].len + sym[best_i + 1].len;
        merged.ptr = xmalloc((size_t)merged.len);
        memcpy(merged.ptr, sym[best_i].ptr, (size_t)sym[best_i].len);
        memcpy(merged.ptr + sym[best_i].len, sym[best_i + 1].ptr, (size_t)sym[best_i + 1].len);
        free(sym[best_i].ptr);
        free(sym[best_i + 1].ptr);
        sym[best_i] = merged;
        for (int j = best_i + 1; j + 1 < n_sym; j++) sym[j] = sym[j + 1];
        n_sym--;
    }
    for (int i = 0; i < n_sym; i++) {
        int token = -1;
        if (table_get(&vocab->token_to_id, sym[i].ptr, sym[i].len, &token)) {
            sf37_tokens_push(out, token);
        } else {
            for (uint64_t j = 0; j < sym[i].len; j++) {
                if (table_get(&vocab->token_to_id, sym[i].ptr + j, 1, &token)) {
                    sf37_tokens_push(out, token);
                }
            }
        }
        free(sym[i].ptr);
    }
    free(sym);
    free(encoded);
}

static bool ascii_alpha(uint8_t c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z');
}

static bool ascii_digit(uint8_t c) {
    return c >= '0' && c <= '9';
}

static bool ascii_space(uint8_t c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
           c == '\v' || c == '\f';
}

static bool ascii_newline(uint8_t c) {
    return c == '\n' || c == '\r';
}

static bool joyai_ascii_punct_symbol(uint8_t c) {
    return (c >= '!' && c <= '/') ||
           (c >= ':' && c <= '@') ||
           (c >= '[' && c <= '`') ||
           (c >= '{' && c <= '~');
}

static bool utf8_is_cjk_hira_kata(uint32_t cp) {
    return (cp >= 0x4e00 && cp <= 0x9fa5) ||
           (cp >= 0x3040 && cp <= 0x309f) ||
           (cp >= 0x30a0 && cp <= 0x30ff);
}

static bool joyai_letter_like_at(const char *s, uint64_t len, uint64_t pos) {
    (void)len;
    uint8_t c = (uint8_t)s[pos];
    if (c < 128) return ascii_alpha(c);
    return true;
}

static uint64_t joyai_consume_letters(const char *s, uint64_t len, uint64_t pos) {
    while (pos < len && joyai_letter_like_at(s, len, pos)) {
        pos = next_utf8_char(s, len, pos);
    }
    return pos;
}

static bool joyai_cjk_at(const char *s, uint64_t len, uint64_t pos) {
    if ((uint8_t)s[pos] < 128) return false;
    uint64_t next = pos;
    uint32_t cp = utf8_peek_one(s, len, pos, &next);
    return utf8_is_cjk_hira_kata(cp);
}

static void bpe_tokenize_text(const sf37_vocab *vocab, const char *text, sf37_tokens *out) {
    if (!text) text = "";
    const uint64_t len = strlen(text);
    uint64_t pos = 0;
    while (pos < len) {
        uint64_t start = pos;
        uint8_t c = (uint8_t)text[pos];
        if (ascii_digit(c)) {
            int ndigits = 0;
            while (pos < len && ascii_digit((uint8_t)text[pos]) && ndigits < 3) {
                pos++;
                ndigits++;
            }
        } else if (joyai_cjk_at(text, len, pos)) {
            do {
                pos = next_utf8_char(text, len, pos);
            } while (pos < len && joyai_cjk_at(text, len, pos));
        } else if (joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   ascii_alpha((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && ascii_alpha((uint8_t)text[pos])) pos++;
        } else if (joyai_letter_like_at(text, len, pos)) {
            pos = joyai_consume_letters(text, len, pos);
        } else if (!ascii_newline(c) &&
                   !joyai_ascii_punct_symbol(c) &&
                   pos + 1 < len &&
                   joyai_letter_like_at(text, len, pos + 1)) {
            pos++;
            pos = joyai_consume_letters(text, len, pos);
        } else if (c == ' ' &&
                   pos + 1 < len &&
                   joyai_ascii_punct_symbol((uint8_t)text[pos + 1])) {
            pos++;
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (joyai_ascii_punct_symbol(c)) {
            while (pos < len && joyai_ascii_punct_symbol((uint8_t)text[pos])) pos++;
            while (pos < len && ascii_newline((uint8_t)text[pos])) pos++;
        } else if (ascii_space(c)) {
            uint64_t p = pos;
            uint64_t last_newline_end = 0;
            while (p < len && ascii_space((uint8_t)text[p])) {
                uint8_t sc = (uint8_t)text[p++];
                if (ascii_newline(sc)) last_newline_end = p;
            }
            if (last_newline_end) {
                pos = last_newline_end;
            } else if (p < len && p > pos + 1 &&
                       (joyai_letter_like_at(text, len, p) ||
                        joyai_ascii_punct_symbol((uint8_t)text[p]))) {
                pos = p - 1;
            } else {
                pos = p;
            }
        } else {
            pos = next_utf8_char(text, len, pos);
        }
        if (pos == start) pos = next_utf8_char(text, len, pos);
        bpe_emit_piece(vocab, (sf37_str){ text + start, pos - start }, out);
    }
}

static bool added_token_at(const sf37_vocab *vocab, const char *p, int *token, size_t *len) {
    int best = -1;
    size_t best_len = 0;
    for (int i = 0; i < vocab->n_added; i++) {
        const sf37_added_token *a = &vocab->added[i];
        if (a->len > best_len && a->len > 0 && strncmp(p, a->text, (size_t)a->len) == 0) {
            best = a->id;
            best_len = (size_t)a->len;
        }
    }
    if (best >= 0) {
        *token = best;
        *len = best_len;
        return true;
    }
    return false;
}

static void tokenize_span(const sf37_vocab *vocab, const char *p, size_t n, sf37_tokens *out) {
    if (!n) return;
    char *tmp = xmalloc(n + 1u);
    memcpy(tmp, p, n);
    tmp[n] = '\0';
    bpe_tokenize_text(vocab, tmp, out);
    free(tmp);
}

static void tokenize_rendered_chat_vocab(const sf37_vocab *vocab, const char *text, sf37_tokens *out) {
    if (!text) text = "";
    const char *span = text;
    const char *p = text;
    while (*p) {
        int token = -1;
        size_t len = 0;
        if (added_token_at(vocab, p, &token, &len)) {
            tokenize_span(vocab, span, (size_t)(p - span), out);
            sf37_tokens_push(out, token);
            p += len;
            span = p;
            continue;
        }
        p++;
    }
    tokenize_span(vocab, span, (size_t)(p - span), out);
}

int sf37_tokenizer_ready(sf37_engine *e) {
    return e && e->vocab.ready;
}

void sf37_tokenize_text(sf37_engine *e, const char *text, sf37_tokens *out) {
    if (!e || !e->vocab.ready || !out) return;
    sf37_tokens_push(out, e->vocab.bos_id);
    bpe_tokenize_text(&e->vocab, text ? text : "", out);
}

void sf37_tokenize_rendered_chat(sf37_engine *e, const char *text, sf37_tokens *out) {
    if (!e || !e->vocab.ready || !out) return;
    tokenize_rendered_chat_vocab(&e->vocab, text ? text : "", out);
}

void sf37_chat_begin(sf37_engine *e, sf37_tokens *tokens) {
    if (!e || !e->vocab.ready || !tokens) return;
    sf37_tokens_push(tokens, e->vocab.bos_id);
}

void sf37_chat_append_message(sf37_engine *e, sf37_tokens *tokens,
                              const char *role, const char *content) {
    if (!e || !e->vocab.ready || !tokens) return;
    const char *r = role && role[0] ? role : "user";
    const char *c = content ? content : "";
    const size_t n = strlen("<|im_start|>\n<|im_end|>\n") + strlen(r) + strlen(c) + 1u;
    char *rendered = xmalloc(n);
    snprintf(rendered, n, "<|im_start|>%s\n%s<|im_end|>\n", r, c);
    tokenize_rendered_chat_vocab(&e->vocab, rendered, tokens);
    free(rendered);
}

void sf37_chat_append_assistant_prefix(sf37_engine *e, sf37_tokens *tokens,
                                       sf37_think_mode think_mode) {
    if (!e || !e->vocab.ready || !tokens) return;
    const char *rendered = think_mode == SF37_THINK_ENABLED ?
        "<|im_start|>assistant\n<think>\n" :
        "<|im_start|>assistant\n";
    tokenize_rendered_chat_vocab(&e->vocab, rendered, tokens);
}

void sf37_encode_chat_prompt(sf37_engine *e, const char *system, const char *prompt,
                             sf37_think_mode think_mode, sf37_tokens *out) {
    if (!e || !e->vocab.ready || !out) return;
    sf37_chat_begin(e, out);
    if (system && system[0]) sf37_chat_append_message(e, out, "system", system);
    sf37_chat_append_message(e, out, "user", prompt ? prompt : "");
    sf37_chat_append_assistant_prefix(e, out, think_mode);
}

static bool token_is_added(const sf37_vocab *vocab, int token) {
    for (int i = 0; i < vocab->n_added; i++) {
        if (vocab->added[i].id == token) return true;
    }
    return false;
}

char *sf37_token_text(sf37_engine *e, int token, size_t *len) {
    if (len) *len = 0;
    if (!e || !e->vocab.ready || token < 0 || token >= e->vocab.n_vocab ||
        !e->vocab.token[token]) {
        return xstrdup("");
    }
    const char *src = e->vocab.token[token];
    uint64_t src_len = e->vocab.token_len[token];
    if (token_is_added(&e->vocab, token)) {
        if (len) *len = (size_t)src_len;
        return xstrndup(src, src_len);
    }
    char *out = xmalloc((size_t)src_len + 1u);
    char *p = out;
    for (uint64_t off = 0; off < src_len;) {
        uint64_t next = off;
        uint32_t cp = utf8_peek_one(src, src_len, off, &next);
        int b = gpt2_codepoint_to_byte(cp);
        if (b >= 0) {
            *p++ = (char)b;
        } else {
            uint64_t n = next - off;
            memcpy(p, src + off, (size_t)n);
            p += n;
        }
        off = next;
    }
    *p = '\0';
    if (len) *len = (size_t)(p - out);
    return out;
}

int sf37_token_bos(sf37_engine *e) {
    return e && e->vocab.ready ? e->vocab.bos_id : -1;
}

int sf37_token_eos(sf37_engine *e) {
    return e && e->vocab.ready ? e->vocab.eos_id : -1;
}

int sf37_token_im_start(sf37_engine *e) {
    return e && e->vocab.ready ? e->vocab.im_start_id : -1;
}

int sf37_token_im_end(sf37_engine *e) {
    return e && e->vocab.ready ? e->vocab.im_end_id : -1;
}

int sf37_token_im_patch(sf37_engine *e) {
    return e && e->vocab.ready ? e->vocab.im_patch_id : -1;
}

static sf37_tensor *find_tensor(const sf37_gguf *g, const char *name) {
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        if (strcmp(g->tensors[i].name, name) == 0) return &g->tensors[i];
    }
    return NULL;
}

static const void *tensor_data(const sf37_engine *e, const sf37_tensor *t) {
    return e->gguf.map + t->abs_offset;
}

static bool has_prefix(const char *s, const char *prefix) {
    return strncmp(s, prefix, strlen(prefix)) == 0;
}

static bool has_mtp_prefix(const char *s) {
    return has_prefix(s, "model.layers.45.") ||
           has_prefix(s, "model.layers.46.") ||
           has_prefix(s, "model.layers.47.");
}

static void tensor_shape(char *buf, size_t len, const sf37_tensor *t) {
    size_t off = 0;
    off += (size_t)snprintf(buf + off, off < len ? len - off : 0, "[");
    for (uint32_t i = 0; i < t->ndim; i++) {
        off += (size_t)snprintf(buf + off, off < len ? len - off : 0,
                                "%s%" PRIu64, i ? "," : "", t->dim[i]);
    }
    snprintf(buf + (off < len ? off : len - 1), off < len ? len - off : 1, "]");
}

static bool tensor_dims_eq(const sf37_tensor *t, uint32_t ndim,
                           uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3) {
    const uint64_t d[4] = {d0, d1, d2, d3};
    if (!t || t->ndim != ndim) return false;
    for (uint32_t i = 0; i < ndim; i++) {
        if (t->dim[i] != d[i]) return false;
    }
    return true;
}

static void bind_error(char *err, size_t errlen, const char *name, const sf37_tensor *t,
                       const char *why) {
    if (!t) {
        snprintf(err, errlen, "required tensor is missing: %s", name);
    } else {
        char shape[128];
        tensor_shape(shape, sizeof(shape), t);
        snprintf(err, errlen, "tensor %s has invalid layout: type=%s shape=%s (%s)",
                 name, type_name(t->type), shape, why);
    }
}

static sf37_tensor *require_tensor(sf37_engine *e, const char *name,
                                   uint32_t type, uint32_t ndim,
                                   uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3,
                                   char *err, size_t errlen) {
    sf37_tensor *t = find_tensor(&e->gguf, name);
    if (!t) {
        bind_error(err, errlen, name, t, "missing");
        return NULL;
    }
    if (type != UINT32_MAX && t->type != type) {
        bind_error(err, errlen, name, t, "wrong type");
        return NULL;
    }
    if (!tensor_dims_eq(t, ndim, d0, d1, d2, d3)) {
        bind_error(err, errlen, name, t, "wrong dimensions");
        return NULL;
    }
    return t;
}

static sf37_tensor *require_any_tensor(sf37_engine *e, const char *name,
                                       uint32_t ndim,
                                       uint64_t d0, uint64_t d1, uint64_t d2, uint64_t d3,
                                       char *err, size_t errlen) {
    return require_tensor(e, name, UINT32_MAX, ndim, d0, d1, d2, d3, err, errlen);
}

static int validate_metadata(sf37_engine *e, char *err, size_t errlen) {
    const sf37_gguf *g = &e->gguf;
    sf37_str s = {0};
    if (!kv_string(g, "general.architecture", &s) || !str_eq(s, "sf37")) {
        snprintf(err, errlen, "general.architecture must be sf37");
        return -1;
    }
    if (!kv_string(g, "sf37.q3_layout", &s) ||
        !str_eq(s, "q3_asym_g64_f16scale_u8zp_payload3_pad110")) {
        snprintf(err, errlen, "sf37.q3_layout is missing or incompatible");
        return -1;
    }
    if (!kv_string(g, "sf37.q2_layout", &s) ||
        !str_eq(s, "q2_asym_g64_f16scale_u8zp_payload2_pad84")) {
        snprintf(err, errlen, "sf37.q2_layout is missing or incompatible");
        return -1;
    }

    uint32_t u = 0;
    if (!kv_u32(g, "sf37.block_count", &u) || u != SF37_MAIN_LAYERS) {
        snprintf(err, errlen, "sf37.block_count must be %u", SF37_MAIN_LAYERS);
        return -1;
    }
    if (!kv_u32(g, "sf37.mtp_layer_count", &u) || u != SF37_MTP_LAYERS) {
        snprintf(err, errlen, "sf37.mtp_layer_count must be %u", SF37_MTP_LAYERS);
        return -1;
    }
    if (!kv_u32(g, "sf37.embedding_length", &u) || u != SF37_EMBD) {
        snprintf(err, errlen, "sf37.embedding_length must be %u", SF37_EMBD);
        return -1;
    }
    if (!kv_u32(g, "sf37.expert_count", &u) || u != SF37_EXPERTS) {
        snprintf(err, errlen, "sf37.expert_count must be %u", SF37_EXPERTS);
        return -1;
    }
    bool vision = false;
    if (!kv_bool(g, "sf37.vision_included", &vision) || !vision) {
        snprintf(err, errlen, "sf37.vision_included must be true");
        return -1;
    }
    return 0;
}

static uint32_t layer_q_heads(uint32_t il) {
    return (il % 4 == 0) ? SF37_FULL_Q_HEADS : SF37_SLIDING_Q_HEADS;
}

static int bind_text_layer(sf37_engine *e, uint32_t il, char *err, size_t errlen) {
    sf37_layer_weights *l = &e->layer[il];
    char name[192];
    uint32_t q_heads = layer_q_heads(il);
    uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;

#define REQ_FIELD(field, fmt, type, ndim, d0, d1, d2, d3) do { \
        snprintf(name, sizeof(name), fmt, il); \
        l->field = require_tensor(e, name, type, ndim, d0, d1, d2, d3, err, errlen); \
        if (!l->field) return -1; \
        e->bound_text_tensors++; \
    } while (0)
#define REQ_ANY_FIELD(field, fmt, ndim, d0, d1, d2, d3) do { \
        snprintf(name, sizeof(name), fmt, il); \
        l->field = require_any_tensor(e, name, ndim, d0, d1, d2, d3, err, errlen); \
        if (!l->field) return -1; \
        e->bound_text_tensors++; \
    } while (0)

    REQ_FIELD(input_norm, "model.layers.%u.input_layernorm.weight", SF37_TENSOR_BF16, 1, SF37_EMBD, 0, 0, 0);
    REQ_FIELD(post_norm, "model.layers.%u.post_attention_layernorm.weight", SF37_TENSOR_BF16, 1, SF37_EMBD, 0, 0, 0);
    REQ_FIELD(q_proj, "model.layers.%u.self_attn.q_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, q_dim, 0, 0);
    REQ_FIELD(k_proj, "model.layers.%u.self_attn.k_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, kv_dim, 0, 0);
    REQ_FIELD(v_proj, "model.layers.%u.self_attn.v_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, kv_dim, 0, 0);
    REQ_FIELD(o_proj, "model.layers.%u.self_attn.o_proj.weight", SF37_TENSOR_Q8_0, 2, q_dim, SF37_EMBD, 0, 0);
    REQ_FIELD(g_proj, "model.layers.%u.self_attn.g_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, q_heads, 0, 0);
    REQ_FIELD(q_norm, "model.layers.%u.self_attn.q_norm.weight", SF37_TENSOR_BF16, 1, SF37_HEAD_DIM, 0, 0, 0);
    REQ_FIELD(k_norm, "model.layers.%u.self_attn.k_norm.weight", SF37_TENSOR_BF16, 1, SF37_HEAD_DIM, 0, 0, 0);

    if (il < 3) {
        REQ_FIELD(mlp_gate, "model.layers.%u.mlp.gate_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, SF37_DENSE_FF, 0, 0);
        REQ_FIELD(mlp_up, "model.layers.%u.mlp.up_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, SF37_DENSE_FF, 0, 0);
        REQ_FIELD(mlp_down, "model.layers.%u.mlp.down_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_DENSE_FF, SF37_EMBD, 0, 0);
    } else {
        REQ_ANY_FIELD(router_gate, "model.layers.%u.moe.gate.weight", 2, SF37_EMBD, SF37_EXPERTS, 0, 0);
        REQ_FIELD(router_bias, "model.layers.%u.moe.router_bias", SF37_TENSOR_F32, 1, SF37_EXPERTS, 0, 0, 0);
        REQ_FIELD(moe_gate, "model.layers.%u.moe.gate_proj.weight", SF37_TENSOR_Q3_K, 3, SF37_EMBD, SF37_EXPERT_FF, SF37_EXPERTS, 0);
        REQ_FIELD(moe_up, "model.layers.%u.moe.up_proj.weight", SF37_TENSOR_Q3_K, 3, SF37_EMBD, SF37_EXPERT_FF, SF37_EXPERTS, 0);
        REQ_FIELD(moe_down, "model.layers.%u.moe.down_proj.weight", SF37_TENSOR_Q2_K, 3, SF37_EXPERT_FF, SF37_EMBD, SF37_EXPERTS, 0);
        REQ_FIELD(share_gate, "model.layers.%u.share_expert.gate_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, SF37_EXPERT_FF, 0, 0);
        REQ_FIELD(share_up, "model.layers.%u.share_expert.up_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EMBD, SF37_EXPERT_FF, 0, 0);
        REQ_FIELD(share_down, "model.layers.%u.share_expert.down_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_EXPERT_FF, SF37_EMBD, 0, 0);
    }

#undef REQ_FIELD
#undef REQ_ANY_FIELD
    return 0;
}

static int bind_vision_block(sf37_engine *e, uint32_t il, char *err, size_t errlen) {
    sf37_vision_block *b = &e->vision.block[il];
    char name[224];

#define VREQ(field, fmt, type, ndim, d0, d1, d2, d3) do { \
        snprintf(name, sizeof(name), fmt, il); \
        b->field = require_tensor(e, name, type, ndim, d0, d1, d2, d3, err, errlen); \
        if (!b->field) return -1; \
        e->bound_vision_tensors++; \
    } while (0)
#define VREQ_ANY(field, fmt, ndim, d0, d1, d2, d3) do { \
        snprintf(name, sizeof(name), fmt, il); \
        b->field = require_any_tensor(e, name, ndim, d0, d1, d2, d3, err, errlen); \
        if (!b->field) return -1; \
        e->bound_vision_tensors++; \
    } while (0)

    VREQ(in_proj_weight, "vision_model.transformer.resblocks.%u.attn.in_proj_weight", SF37_TENSOR_Q8_0, 2, SF37_VISION_WIDTH, SF37_VISION_WIDTH * 3, 0, 0);
    VREQ_ANY(in_proj_bias, "vision_model.transformer.resblocks.%u.attn.in_proj_bias", 1, SF37_VISION_WIDTH * 3, 0, 0, 0);
    VREQ(out_proj_weight, "vision_model.transformer.resblocks.%u.attn.out_proj.weight", SF37_TENSOR_Q8_0, 2, SF37_VISION_WIDTH, SF37_VISION_WIDTH, 0, 0);
    VREQ_ANY(out_proj_bias, "vision_model.transformer.resblocks.%u.attn.out_proj.bias", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ln1_weight, "vision_model.transformer.resblocks.%u.ln_1.weight", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ln1_bias, "vision_model.transformer.resblocks.%u.ln_1.bias", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ln2_weight, "vision_model.transformer.resblocks.%u.ln_2.weight", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ln2_bias, "vision_model.transformer.resblocks.%u.ln_2.bias", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ls1_gamma, "vision_model.transformer.resblocks.%u.ls_1.gamma", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ_ANY(ls2_gamma, "vision_model.transformer.resblocks.%u.ls_2.gamma", 1, SF37_VISION_WIDTH, 0, 0, 0);
    VREQ(mlp_fc_weight, "vision_model.transformer.resblocks.%u.mlp.c_fc.weight", SF37_TENSOR_Q8_0, 2, SF37_VISION_WIDTH, 8960, 0, 0);
    VREQ_ANY(mlp_fc_bias, "vision_model.transformer.resblocks.%u.mlp.c_fc.bias", 1, 8960, 0, 0, 0);
    VREQ(mlp_proj_weight, "vision_model.transformer.resblocks.%u.mlp.c_proj.weight", SF37_TENSOR_Q8_0, 2, 8960, SF37_VISION_WIDTH, 0, 0);
    VREQ_ANY(mlp_proj_bias, "vision_model.transformer.resblocks.%u.mlp.c_proj.bias", 1, SF37_VISION_WIDTH, 0, 0, 0);

#undef VREQ
#undef VREQ_ANY
    return 0;
}

static int bind_vision(sf37_engine *e, char *err, size_t errlen) {
    sf37_vision_weights *v = &e->vision;

#define WREQ(field, name, type, ndim, d0, d1, d2, d3) do { \
        v->field = require_tensor(e, name, type, ndim, d0, d1, d2, d3, err, errlen); \
        if (!v->field) return -1; \
        e->bound_vision_tensors++; \
    } while (0)
#define WREQ_ANY(field, name, ndim, d0, d1, d2, d3) do { \
        v->field = require_any_tensor(e, name, ndim, d0, d1, d2, d3, err, errlen); \
        if (!v->field) return -1; \
        e->bound_vision_tensors++; \
    } while (0)

    WREQ(conv1_weight, "vision_model.conv1.weight", SF37_TENSOR_BF16, 4, SF37_VISION_PATCH, SF37_VISION_PATCH, 3, SF37_VISION_WIDTH);
    WREQ_ANY(ln_pre_weight, "vision_model.ln_pre.weight", 1, SF37_VISION_WIDTH, 0, 0, 0);
    WREQ_ANY(ln_pre_bias, "vision_model.ln_pre.bias", 1, SF37_VISION_WIDTH, 0, 0, 0);
    WREQ(positional_embedding, "vision_model.positional_embedding", SF37_TENSOR_Q8_0, 2, SF37_VISION_WIDTH, SF37_VISION_GRID * SF37_VISION_GRID, 0, 0);

    for (uint32_t il = 0; il < SF37_VISION_LAYERS; il++) {
        if (bind_vision_block(e, il, err, errlen) != 0) return -1;
    }

    WREQ(down1_weight, "vision_model.vit_downsampler1.weight", SF37_TENSOR_BF16, 4, 3, 3, SF37_VISION_WIDTH, SF37_VISION_WIDTH * 2);
    WREQ_ANY(down1_bias, "vision_model.vit_downsampler1.bias", 1, SF37_VISION_WIDTH * 2, 0, 0, 0);
    WREQ(down2_weight, "vision_model.vit_downsampler2.weight", SF37_TENSOR_BF16, 4, 3, 3, SF37_VISION_WIDTH * 2, SF37_VISION_WIDTH * 4);
    WREQ_ANY(down2_bias, "vision_model.vit_downsampler2.bias", 1, SF37_VISION_WIDTH * 4, 0, 0, 0);
    WREQ(projector, "vit_large_projector.weight", SF37_TENSOR_BF16, 2, SF37_VISION_WIDTH * 4, SF37_EMBD, 0, 0);

#undef WREQ
#undef WREQ_ANY
    return 0;
}

static int bind_model(sf37_engine *e, char *err, size_t errlen) {
    if (validate_metadata(e, err, errlen) != 0) return -1;

    for (uint64_t i = 0; i < e->gguf.n_tensors; i++) {
        sf37_tensor *t = &e->gguf.tensors[i];
        if (t->type < 42) {
            e->type_count[t->type]++;
            e->type_bytes[t->type] += t->nbytes;
        }
        if (has_mtp_prefix(t->name)) e->inactive_mtp_tensors++;
    }

    e->embed_tokens = require_tensor(e, "model.embed_tokens.weight", SF37_TENSOR_BF16,
                                     2, SF37_EMBD, SF37_VOCAB, 0, 0, err, errlen);
    if (!e->embed_tokens) return -1;
    e->lm_head = require_tensor(e, "lm_head.weight", SF37_TENSOR_BF16,
                                2, SF37_EMBD, SF37_VOCAB, 0, 0, err, errlen);
    if (!e->lm_head) return -1;
    e->final_norm = require_tensor(e, "model.norm.weight", SF37_TENSOR_BF16,
                                   1, SF37_EMBD, 0, 0, 0, err, errlen);
    if (!e->final_norm) return -1;
    e->bound_text_tensors += 3;

    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (bind_text_layer(e, il, err, errlen) != 0) return -1;
    }
    if (bind_vision(e, err, errlen) != 0) return -1;
    return 0;
}

static float *xmalloc_f32(uint64_t n) {
    if (n > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "sf37: allocation too large\n");
        exit(1);
    }
    float *p = malloc((size_t)n * sizeof(p[0]));
    if (!p) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    return p;
}

static void tensor_bf16_to_f32(const sf37_engine *e, const sf37_tensor *t,
                               float *out, uint64_t n) {
    const uint16_t *p = tensor_data(e, t);
    for (uint64_t i = 0; i < n; i++) out[i] = sf37_bf16_to_f32(p[i]);
}

static void embed_token_bf16(const sf37_engine *e, int token, float *out) {
    if (token < 0 || (uint64_t)token >= e->embed_tokens->dim[1]) {
        fprintf(stderr, "sf37: token id is outside embedding table\n");
        exit(1);
    }
    const uint16_t *base = tensor_data(e, e->embed_tokens);
    const uint64_t stride = e->embed_tokens->dim[0];
    const uint16_t *row = base + (uint64_t)token * stride;
    for (uint64_t i = 0; i < stride; i++) out[i] = sf37_bf16_to_f32(row[i]);
}

static void head_rms_norm_weight1(float *x, uint32_t n_head, uint32_t head_dim,
                                  const float *weight, float eps) {
    float tmp[SF37_HEAD_DIM];
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        sf37_rms_norm_weight1(tmp, head, weight, head_dim, eps);
        memcpy(head, tmp, (size_t)head_dim * sizeof(tmp[0]));
    }
}

static void print_vec_stats(FILE *fp, const char *name, const float *x, uint64_t n) {
    float minv = INFINITY;
    float maxv = -INFINITY;
    double ss = 0.0;
    double sum = 0.0;
    for (uint64_t i = 0; i < n; i++) {
        const float v = x[i];
        if (v < minv) minv = v;
        if (v > maxv) maxv = v;
        ss += (double)v * (double)v;
        sum += (double)v;
    }
    fprintf(fp, "%s: min=%g max=%g mean=%g rms=%g\n",
            name, minv, maxv, sum / (double)n, sqrt(ss / (double)n));
}

static const uint8_t *expert_bytes(const sf37_engine *e, const sf37_tensor *t,
                                   uint32_t expert, uint64_t block_bytes) {
    if (!t || t->ndim != 3 || expert >= t->dim[2]) {
        fprintf(stderr, "sf37: invalid expert tensor access\n");
        exit(1);
    }
    const uint64_t in_dim = t->dim[0];
    const uint64_t out_dim = t->dim[1];
    const uint64_t blocks = in_dim / SF37_QK_K;
    const uint64_t row_bytes = blocks * block_bytes;
    return (const uint8_t *)tensor_data(e, t) + (uint64_t)expert * out_dim * row_bytes;
}

static void matvec_dense_tensor(float *out, const sf37_engine *e,
                                const sf37_tensor *t, const float *x) {
    if (!t || t->ndim != 2) {
        fprintf(stderr, "sf37: dense matvec expects a rank-2 tensor\n");
        exit(1);
    }
    if (t->type == SF37_TENSOR_Q8_0) {
        sf37_q8_0_matvec(out, tensor_data(e, t), t->dim[0], t->dim[1], x);
    } else if (t->type == SF37_TENSOR_BF16) {
        sf37_bf16_matvec(out, tensor_data(e, t), t->dim[0], t->dim[1], x);
    } else if (t->type == SF37_TENSOR_F32) {
        const float *w = tensor_data(e, t);
        for (uint64_t r = 0; r < t->dim[1]; r++) {
            double acc = 0.0;
            for (uint64_t i = 0; i < t->dim[0]; i++) acc += (double)w[r * t->dim[0] + i] * x[i];
            out[r] = (float)acc;
        }
    } else {
        fprintf(stderr, "sf37: unsupported dense matvec tensor type %u\n", t->type);
        exit(1);
    }
}

static bool layer_is_full(uint32_t il) {
    return (il % 4u) == 0u;
}

static uint32_t layer_rotary_dim(uint32_t il) {
    return layer_is_full(il) ? 64u : 128u;
}

static double layer_rope_theta(uint32_t il) {
    return layer_is_full(il) ? 5000000.0 : 10000.0;
}

static double sf37_inv_freq(uint32_t pair, uint32_t dim, double theta, bool llama3) {
    double inv = 1.0 / pow(theta, (double)(pair * 2u) / (double)dim);
    if (!llama3) return inv;

    const double factor = 2.0;
    const double low_freq_factor = 1.0;
    const double high_freq_factor = 32.0;
    const double old_context_len = 131072.0;
    const double low_freq_wavelen = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;
    const double wavelen = 2.0 * SF37_PI / inv;

    if (wavelen > low_freq_wavelen) return inv / factor;
    if (wavelen < high_freq_wavelen) return inv;

    const double smooth = (old_context_len / wavelen - low_freq_factor) /
                          (high_freq_factor - low_freq_factor);
    return (1.0 - smooth) * inv / factor + smooth * inv;
}

static void apply_rope_one(float *x, uint32_t n_head, uint32_t head_dim,
                           uint32_t rotary_dim, double theta, bool llama3,
                           uint32_t pos) {
    const uint32_t half = rotary_dim / 2u;
    for (uint32_t h = 0; h < n_head; h++) {
        float *head = x + (uint64_t)h * head_dim;
        for (uint32_t i = 0; i < half; i++) {
            const double inv = sf37_inv_freq(i, rotary_dim, theta, llama3);
            const float c = cosf((float)((double)pos * inv));
            const float s = sinf((float)((double)pos * inv));
            const float a = head[i];
            const float b = head[i + half];
            head[i] = a * c - b * s;
            head[i + half] = b * c + a * s;
        }
    }
}

static void kv_cache_init(sf37_kv_cache *cache, uint32_t cap) {
    memset(cache, 0, sizeof(*cache));
    if (cap == 0) cap = 1;
    cache->cap = cap;
    const uint64_t row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        cache->layer[il].cap = cap;
        cache->layer[il].k = xcalloc((size_t)cap * row, sizeof(float));
        cache->layer[il].v = xcalloc((size_t)cap * row, sizeof(float));
    }
}

static void kv_cache_free(sf37_kv_cache *cache) {
    if (!cache) return;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        free(cache->layer[il].k);
        free(cache->layer[il].v);
    }
    memset(cache, 0, sizeof(*cache));
}

static void kv_cache_push(sf37_layer_kv_cache *layer, const float *k, const float *v) {
    const uint64_t row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    if (layer->n >= layer->cap) {
        memmove(layer->k, layer->k + row, (size_t)(layer->cap - 1u) * row * sizeof(float));
        memmove(layer->v, layer->v + row, (size_t)(layer->cap - 1u) * row * sizeof(float));
        layer->n = layer->cap - 1u;
    }
    memcpy(layer->k + (uint64_t)layer->n * row, k, (size_t)row * sizeof(float));
    memcpy(layer->v + (uint64_t)layer->n * row, v, (size_t)row * sizeof(float));
    layer->n++;
}

static void attention_decode(float *out_heads, const float *q,
                             const sf37_layer_kv_cache *cache,
                             uint32_t q_heads, bool sliding) {
    const uint32_t repeat = q_heads / SF37_KV_HEADS;
    const uint64_t kv_row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    uint32_t start = 0;
    if (sliding && cache->n > SF37_SLIDING_WINDOW) start = cache->n - SF37_SLIDING_WINDOW;
    for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kvh = h / repeat;
        const float *qh = q + (uint64_t)h * SF37_HEAD_DIM;
        float *oh = out_heads + (uint64_t)h * SF37_HEAD_DIM;
        float max_score = -INFINITY;
        for (uint32_t r = start; r < cache->n; r++) {
            const float *kh = cache->k + (uint64_t)r * kv_row + (uint64_t)kvh * SF37_HEAD_DIM;
            float score = 0.0f;
            for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) score += qh[d] * kh[d];
            score *= 0.08838834764831845f;
            if (score > max_score) max_score = score;
        }

        float denom = 0.0f;
        memset(oh, 0, (size_t)SF37_HEAD_DIM * sizeof(oh[0]));
        for (uint32_t r = start; r < cache->n; r++) {
            const float *kh = cache->k + (uint64_t)r * kv_row + (uint64_t)kvh * SF37_HEAD_DIM;
            const float *vh = cache->v + (uint64_t)r * kv_row + (uint64_t)kvh * SF37_HEAD_DIM;
            float score = 0.0f;
            for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) score += qh[d] * kh[d];
            score *= 0.08838834764831845f;
            const float w = expf(score - max_score);
            denom += w;
            for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) oh[d] += w * vh[d];
        }
        const float inv = denom > 0.0f ? 1.0f / denom : 0.0f;
        for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) oh[d] *= inv;
    }
}

static float layer_swiglu_limit(uint32_t il, bool shared) {
    if (il == 43u || il == 44u) return shared ? 16.0f : 7.0f;
    return 0.0f;
}

static void layer_dense_ffn(float *out, const sf37_engine *e,
                            const sf37_layer_weights *l, const float *x,
                            uint32_t il) {
    const float clamp = layer_swiglu_limit(il, true);
    float *gate = xmalloc_f32(SF37_DENSE_FF);
    float *up = xmalloc_f32(SF37_DENSE_FF);
    float *mid = xmalloc_f32(SF37_DENSE_FF);
    sf37_q8_0_matvec_pair(gate, up,
                          tensor_data(e, l->mlp_gate),
                          tensor_data(e, l->mlp_up),
                          SF37_EMBD, SF37_DENSE_FF, x);
    sf37_swiglu(mid, gate, up, SF37_DENSE_FF, clamp);
    sf37_q8_0_matvec(out, tensor_data(e, l->mlp_down),
                     SF37_DENSE_FF, SF37_EMBD, mid);
    free(mid);
    free(up);
    free(gate);
}

static void layer_moe_ffn(float *out, const sf37_engine *e,
                          const sf37_layer_weights *l, const float *x,
                          uint32_t il) {
    float *router_logits = xmalloc_f32(SF37_EXPERTS);
    float *router_prob = xmalloc_f32(SF37_EXPERTS);
    float *selection = xmalloc_f32(SF37_EXPERTS);
    float *shared_gate = xmalloc_f32(SF37_EXPERT_FF);
    float *shared_up = xmalloc_f32(SF37_EXPERT_FF);
    float *shared_mid = xmalloc_f32(SF37_EXPERT_FF);
    float *shared_out = xmalloc_f32(SF37_EMBD);
    float *expert_gate = xmalloc_f32(SF37_EXPERT_FF);
    float *expert_up = xmalloc_f32(SF37_EXPERT_FF);
    float *expert_mid = xmalloc_f32(SF37_EXPERT_FF);
    float *expert_down = xmalloc_f32(SF37_EMBD);
    int selected[SF37_EXPERT_USED];
    float weights[SF37_EXPERT_USED];

    matvec_dense_tensor(router_logits, e, l->router_gate, x);
    const float *bias = tensor_data(e, l->router_bias);
    for (uint32_t i = 0; i < SF37_EXPERTS; i++) {
        router_prob[i] = sf37_sigmoid(router_logits[i]);
        selection[i] = router_prob[i] + bias[i];
    }
    sf37_topk_desc(selection, SF37_EXPERTS, SF37_EXPERT_USED, selected);
    float sum = 0.0f;
    for (uint32_t i = 0; i < SF37_EXPERT_USED; i++) {
        weights[i] = router_prob[selected[i]];
        sum += weights[i];
    }
    if (sum < 1.0e-20f) sum = 1.0e-20f;
    for (uint32_t i = 0; i < SF37_EXPERT_USED; i++) weights[i] = weights[i] / sum * 3.0f;

    sf37_q8_0_matvec_pair(shared_gate, shared_up,
                          tensor_data(e, l->share_gate),
                          tensor_data(e, l->share_up),
                          SF37_EMBD, SF37_EXPERT_FF, x);
    sf37_swiglu(shared_mid, shared_gate, shared_up, SF37_EXPERT_FF,
                layer_swiglu_limit(il, true));
    sf37_q8_0_matvec(shared_out, tensor_data(e, l->share_down),
                     SF37_EXPERT_FF, SF37_EMBD, shared_mid);
    memcpy(out, shared_out, (size_t)SF37_EMBD * sizeof(out[0]));

    for (uint32_t i = 0; i < SF37_EXPERT_USED; i++) {
        const uint32_t ex = (uint32_t)selected[i];
        const uint8_t *gate_w = expert_bytes(e, l->moe_gate, ex, SF37_Q3_BLOCK_SIZE);
        const uint8_t *up_w = expert_bytes(e, l->moe_up, ex, SF37_Q3_BLOCK_SIZE);
        const uint8_t *down_w = expert_bytes(e, l->moe_down, ex, SF37_Q2_BLOCK_SIZE);
        sf37_q3_asym_matvec(expert_gate, gate_w, SF37_EMBD, SF37_EXPERT_FF, x);
        sf37_q3_asym_matvec(expert_up, up_w, SF37_EMBD, SF37_EXPERT_FF, x);
        sf37_swiglu(expert_mid, expert_gate, expert_up, SF37_EXPERT_FF,
                    layer_swiglu_limit(il, false));
        for (uint32_t j = 0; j < SF37_EXPERT_FF; j++) expert_mid[j] *= weights[i];
        sf37_q2_asym_matvec(expert_down, down_w, SF37_EXPERT_FF, SF37_EMBD, expert_mid);
        for (uint32_t j = 0; j < SF37_EMBD; j++) out[j] += expert_down[j];
    }

    free(expert_down);
    free(expert_mid);
    free(expert_up);
    free(expert_gate);
    free(shared_out);
    free(shared_mid);
    free(shared_up);
    free(shared_gate);
    free(selection);
    free(router_prob);
    free(router_logits);
}

static void run_layer_decode(sf37_engine *e, sf37_kv_cache *cache,
                             uint32_t il, uint32_t pos, float *hidden) {
    sf37_layer_weights *l = &e->layer[il];
    const bool full = layer_is_full(il);
    const uint32_t q_heads = layer_q_heads(il);
    const uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    const uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;

    float *attn_norm = xmalloc_f32(SF37_EMBD);
    float *q = xmalloc_f32(q_dim);
    float *k = xmalloc_f32(kv_dim);
    float *v = xmalloc_f32(kv_dim);
    float *head_gate = xmalloc_f32(q_heads);
    float *attn_heads = xmalloc_f32(q_dim);
    float *attn_out = xmalloc_f32(SF37_EMBD);
    float *ffn_norm = xmalloc_f32(SF37_EMBD);
    float *ffn_out = xmalloc_f32(SF37_EMBD);
    float *w_embd = xmalloc_f32(SF37_EMBD);
    float *w_head = xmalloc_f32(SF37_HEAD_DIM);

    tensor_bf16_to_f32(e, l->input_norm, w_embd, SF37_EMBD);
    sf37_rms_norm_weight1(attn_norm, hidden, w_embd, SF37_EMBD, SF37_RMS_EPS);
    sf37_q8_0_matvec(q, tensor_data(e, l->q_proj), SF37_EMBD, q_dim, attn_norm);
    sf37_q8_0_matvec(k, tensor_data(e, l->k_proj), SF37_EMBD, kv_dim, attn_norm);
    sf37_q8_0_matvec(v, tensor_data(e, l->v_proj), SF37_EMBD, kv_dim, attn_norm);
    sf37_q8_0_matvec(head_gate, tensor_data(e, l->g_proj), SF37_EMBD, q_heads, attn_norm);

    tensor_bf16_to_f32(e, l->q_norm, w_head, SF37_HEAD_DIM);
    head_rms_norm_weight1(q, q_heads, SF37_HEAD_DIM, w_head, SF37_RMS_EPS);
    tensor_bf16_to_f32(e, l->k_norm, w_head, SF37_HEAD_DIM);
    head_rms_norm_weight1(k, SF37_KV_HEADS, SF37_HEAD_DIM, w_head, SF37_RMS_EPS);

    const uint32_t rotary_dim = layer_rotary_dim(il);
    const double theta = layer_rope_theta(il);
    apply_rope_one(q, q_heads, SF37_HEAD_DIM, rotary_dim, theta, full, pos);
    apply_rope_one(k, SF37_KV_HEADS, SF37_HEAD_DIM, rotary_dim, theta, full, pos);
    kv_cache_push(&cache->layer[il], k, v);

    attention_decode(attn_heads, q, &cache->layer[il], q_heads, !full);
    for (uint32_t h = 0; h < q_heads; h++) {
        const float g = sf37_sigmoid(head_gate[h]);
        float *head = attn_heads + (uint64_t)h * SF37_HEAD_DIM;
        for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) head[d] *= g;
    }
    sf37_q8_0_matvec(attn_out, tensor_data(e, l->o_proj), q_dim, SF37_EMBD, attn_heads);
    for (uint32_t i = 0; i < SF37_EMBD; i++) hidden[i] += attn_out[i];

    tensor_bf16_to_f32(e, l->post_norm, w_embd, SF37_EMBD);
    sf37_rms_norm_weight1(ffn_norm, hidden, w_embd, SF37_EMBD, SF37_RMS_EPS);
    if (il < 3u) {
        layer_dense_ffn(ffn_out, e, l, ffn_norm, il);
    } else {
        layer_moe_ffn(ffn_out, e, l, ffn_norm, il);
    }
    for (uint32_t i = 0; i < SF37_EMBD; i++) hidden[i] += ffn_out[i];

    free(w_head);
    free(w_embd);
    free(ffn_out);
    free(ffn_norm);
    free(attn_out);
    free(attn_heads);
    free(head_gate);
    free(v);
    free(k);
    free(q);
    free(attn_norm);
}

static void output_logits_bf16(float *logits, const sf37_engine *e,
                               const float *hidden, FILE *fp) {
    float *norm = xmalloc_f32(SF37_EMBD);
    float *w = xmalloc_f32(SF37_EMBD);

    tensor_bf16_to_f32(e, e->final_norm, w, SF37_EMBD);
    sf37_rms_norm_weight1(norm, hidden, w, SF37_EMBD, SF37_RMS_EPS);
    if (fp) print_vec_stats(fp, "final_norm", norm, SF37_EMBD);
    sf37_bf16_matvec(logits, tensor_data(e, e->lm_head),
                     SF37_EMBD, SF37_VOCAB, norm);

    free(w);
    free(norm);
}

static void print_top_logits(FILE *fp, const float *logits, int topk) {
    if (topk <= 0) return;
    if (topk > SF37_VOCAB) topk = SF37_VOCAB;

    int *idx = malloc((size_t)topk * sizeof(idx[0]));
    if (!idx) {
        fprintf(stderr, "sf37: out of memory\n");
        exit(1);
    }
    sf37_topk_desc(logits, SF37_VOCAB, topk, idx);

    fprintf(fp, "top logits:\n");
    for (int i = 0; i < topk; i++) {
        fprintf(fp, "  %6d  %12.5f\n", idx[i], logits[idx[i]]);
    }
    free(idx);
}

int sf37_engine_smoke_decode(sf37_engine *e, int token, int layers, int topk, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (layers < 0 || layers > SF37_MAIN_LAYERS) {
        sf37_log(stderr, SF37_LOG_ERROR, "--smoke-layers must be in [0,%u]", SF37_MAIN_LAYERS);
        return -1;
    }
    if (topk < 0 || topk > SF37_VOCAB) {
        sf37_log(stderr, SF37_LOG_ERROR, "--smoke-topk must be in [0,%u]", SF37_VOCAB);
        return -1;
    }
    float *hidden = xmalloc_f32(SF37_EMBD);
    sf37_kv_cache cache;
    kv_cache_init(&cache, 8);
    embed_token_bf16(e, token, hidden);

    fprintf(fp, "sf37 smoke token=%d decode-reference layers=%d/%u\n",
            token, layers, SF37_MAIN_LAYERS);
    print_vec_stats(fp, "embed", hidden, SF37_EMBD);
    for (int il = 0; il < layers; il++) {
        char label[64];
        run_layer_decode(e, &cache, (uint32_t)il, 0, hidden);
        snprintf(label, sizeof(label), "after_layer%d%s", il, il >= 3 ? "_moe" : "");
        print_vec_stats(fp, label, hidden, SF37_EMBD);
    }

    if (topk > 0) {
        if (layers != SF37_MAIN_LAYERS) {
            fprintf(fp, "logits note: partial transformer stack (%d/%u layers); use --smoke-layers %u for a full CPU reference.\n",
                    layers, SF37_MAIN_LAYERS, SF37_MAIN_LAYERS);
        }
        float *logits = xmalloc_f32(SF37_VOCAB);
        output_logits_bf16(logits, e, hidden, fp);
        print_vec_stats(fp, "logits", logits, SF37_VOCAB);
        print_top_logits(fp, logits, topk);
        free(logits);
    }

    kv_cache_free(&cache);
    free(hidden);
    return 0;
}

int sf37_engine_smoke_token(sf37_engine *e, int token, FILE *fp) {
    return sf37_engine_smoke_decode(e, token, 4, 0, fp);
}

int sf37_engine_vocab_size(sf37_engine *e) {
    (void)e;
    return SF37_VOCAB;
}

static double sf37_now_sec(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
    }
    return (double)clock() / (double)CLOCKS_PER_SEC;
}

#ifdef SF37_USE_CUDA
typedef struct {
    uint64_t off;
    uint64_t end;
} sf37_model_map_span;

typedef struct {
    sf37_model_map_span *v;
    uint32_t len;
    uint32_t cap;
    uint64_t max_tensor_bytes;
} sf37_model_map_span_vec;

typedef struct {
    uint64_t off;
    uint64_t end;
} sf37_accelerator_tensor_span;

static int sf37_env_present(const char *sf37_name, const char *ds4_name) {
    return (sf37_name && getenv(sf37_name) != NULL) ||
           (ds4_name && getenv(ds4_name) != NULL);
}

static const char *sf37_env_value(const char *sf37_name, const char *ds4_name) {
    const char *v = sf37_name ? getenv(sf37_name) : NULL;
    if (v && v[0]) return v;
    v = ds4_name ? getenv(ds4_name) : NULL;
    if (v && v[0]) return v;
    return NULL;
}

static void model_map_span_include_tensor(const sf37_tensor *t,
                                          uint64_t *lo,
                                          uint64_t *hi,
                                          uint64_t *max_tensor_bytes) {
    if (!t || t->nbytes == 0) return;
    const uint64_t end = t->abs_offset + t->nbytes;
    if (*lo == UINT64_MAX || t->abs_offset < *lo) *lo = t->abs_offset;
    if (end > *hi) *hi = end;
    if (t->nbytes > *max_tensor_bytes) *max_tensor_bytes = t->nbytes;
}

static void model_map_span_vec_append(sf37_model_map_span_vec *spans,
                                      uint64_t lo,
                                      uint64_t hi) {
    if (!spans || lo == UINT64_MAX || hi <= lo) return;
    if (spans->len == spans->cap) {
        uint32_t new_cap = spans->cap ? spans->cap * 2u : 16u;
        spans->v = xrealloc(spans->v, (size_t)new_cap * sizeof(spans->v[0]));
        spans->cap = new_cap;
    }
    spans->v[spans->len++] = (sf37_model_map_span){lo, hi};
}

static void model_map_span_vec_include_one(sf37_model_map_span_vec *spans,
                                           const sf37_tensor *t) {
    if (!t || t->nbytes == 0) return;
    uint64_t lo = UINT64_MAX, hi = 0;
    model_map_span_include_tensor(t, &lo, &hi, &spans->max_tensor_bytes);
    model_map_span_vec_append(spans, lo, hi);
}

static void model_map_span_vec_include_layer(sf37_model_map_span_vec *spans,
                                             const sf37_layer_weights *l) {
#define SF37_INCLUDE_TENSOR(t_) model_map_span_vec_include_one(spans, (t_))
    SF37_INCLUDE_TENSOR(l->q_proj);
    SF37_INCLUDE_TENSOR(l->k_proj);
    SF37_INCLUDE_TENSOR(l->v_proj);
    SF37_INCLUDE_TENSOR(l->o_proj);
    SF37_INCLUDE_TENSOR(l->g_proj);
    SF37_INCLUDE_TENSOR(l->q_norm);
    SF37_INCLUDE_TENSOR(l->k_norm);
    SF37_INCLUDE_TENSOR(l->input_norm);
    SF37_INCLUDE_TENSOR(l->post_norm);
    SF37_INCLUDE_TENSOR(l->mlp_gate);
    SF37_INCLUDE_TENSOR(l->mlp_up);
    SF37_INCLUDE_TENSOR(l->mlp_down);
    SF37_INCLUDE_TENSOR(l->router_gate);
    SF37_INCLUDE_TENSOR(l->router_bias);
    SF37_INCLUDE_TENSOR(l->moe_gate);
    SF37_INCLUDE_TENSOR(l->moe_up);
    SF37_INCLUDE_TENSOR(l->moe_down);
    SF37_INCLUDE_TENSOR(l->share_gate);
    SF37_INCLUDE_TENSOR(l->share_up);
    SF37_INCLUDE_TENSOR(l->share_down);
#undef SF37_INCLUDE_TENSOR
}

static void model_map_span_vec_include_output(sf37_model_map_span_vec *spans,
                                              const sf37_engine *e) {
    model_map_span_vec_include_one(spans, e->final_norm);
    model_map_span_vec_include_one(spans, e->lm_head);
}

static void model_map_span_vec_include_vision(sf37_model_map_span_vec *spans,
                                              const sf37_vision_weights *v) {
    model_map_span_vec_include_one(spans, v->conv1_weight);
    model_map_span_vec_include_one(spans, v->ln_pre_weight);
    model_map_span_vec_include_one(spans, v->ln_pre_bias);
    model_map_span_vec_include_one(spans, v->positional_embedding);
    for (uint32_t il = 0; il < SF37_VISION_LAYERS; il++) {
        const sf37_vision_block *b = &v->block[il];
        model_map_span_vec_include_one(spans, b->in_proj_weight);
        model_map_span_vec_include_one(spans, b->in_proj_bias);
        model_map_span_vec_include_one(spans, b->out_proj_weight);
        model_map_span_vec_include_one(spans, b->out_proj_bias);
        model_map_span_vec_include_one(spans, b->ln1_weight);
        model_map_span_vec_include_one(spans, b->ln1_bias);
        model_map_span_vec_include_one(spans, b->ln2_weight);
        model_map_span_vec_include_one(spans, b->ln2_bias);
        model_map_span_vec_include_one(spans, b->ls1_gamma);
        model_map_span_vec_include_one(spans, b->ls2_gamma);
        model_map_span_vec_include_one(spans, b->mlp_fc_weight);
        model_map_span_vec_include_one(spans, b->mlp_fc_bias);
        model_map_span_vec_include_one(spans, b->mlp_proj_weight);
        model_map_span_vec_include_one(spans, b->mlp_proj_bias);
    }
    model_map_span_vec_include_one(spans, v->down1_weight);
    model_map_span_vec_include_one(spans, v->down1_bias);
    model_map_span_vec_include_one(spans, v->down2_weight);
    model_map_span_vec_include_one(spans, v->down2_bias);
    model_map_span_vec_include_one(spans, v->projector);
}

static int model_map_span_cmp(const void *a, const void *b) {
    const sf37_model_map_span *sa = a;
    const sf37_model_map_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static bool sf37_weights_model_map_spans(const sf37_engine *e,
                                         bool include_output,
                                         bool include_vision,
                                         sf37_model_map_span_vec *spans) {
    if (!e || !spans) return false;
    memset(spans, 0, sizeof(*spans));
    model_map_span_vec_include_one(spans, e->embed_tokens);
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        model_map_span_vec_include_layer(spans, &e->layer[il]);
    }
    if (include_output) model_map_span_vec_include_output(spans, e);
    if (include_vision) model_map_span_vec_include_vision(spans, &e->vision);
    if (spans->len == 0 || spans->max_tensor_bytes == 0) return false;

    qsort(spans->v, spans->len, sizeof(spans->v[0]), model_map_span_cmp);
    uint32_t out = 0;
    for (uint32_t i = 0; i < spans->len; i++) {
        if (out == 0 || spans->v[i].off > spans->v[out - 1u].end) {
            spans->v[out++] = spans->v[i];
        } else if (spans->v[i].end > spans->v[out - 1u].end) {
            spans->v[out - 1u].end = spans->v[i].end;
        }
    }
    spans->len = out;
    return spans->len != 0;
}

static int accelerator_tensor_span_cmp(const void *a, const void *b) {
    const sf37_accelerator_tensor_span *sa = a;
    const sf37_accelerator_tensor_span *sb = b;
    if (sa->off < sb->off) return -1;
    if (sa->off > sb->off) return 1;
    if (sa->end < sb->end) return -1;
    if (sa->end > sb->end) return 1;
    return 0;
}

static uint64_t accelerator_cuda_preload_span_bytes(void) {
    uint64_t mb = 1024;
    const char *env = sf37_env_value("SF37_CUDA_WEIGHT_PRELOAD_SPAN_MB",
                                     "DS4_CUDA_WEIGHT_PRELOAD_SPAN_MB");
    if (env) {
        char *end = NULL;
        unsigned long long v = strtoull(env, &end, 10);
        if (end != env && v > 0) mb = (uint64_t)v;
    }
    if (mb < 64) mb = 64;
    if (mb > 4096) mb = 4096;
    return mb * 1048576ull;
}

static uint64_t accelerator_cuda_startup_preload_limit_bytes(bool *present_out) {
    const char *env = sf37_env_value("SF37_CUDA_STARTUP_PRELOAD_GB",
                                     "DS4_CUDA_STARTUP_PRELOAD_GB");
    if (present_out) *present_out = env != NULL;
    if (!env) return UINT64_MAX;
    char *end = NULL;
    unsigned long long v = strtoull(env, &end, 10);
    if (end == env) return UINT64_MAX;
    if (v > UINT64_MAX / 1073741824ull) return UINT64_MAX;
    return (uint64_t)v * 1073741824ull;
}

static bool accelerator_cuda_startup_preload_explicit(void);

static uint64_t accelerator_cuda_default_limited_preload_bytes(uint64_t span_bytes,
                                                               bool *limited_out) {
    if (limited_out) *limited_out = false;
    if (accelerator_cuda_startup_preload_explicit()) return UINT64_MAX;
    uint64_t free_b = 0, total_b = 0;
    if (!sf37_cuda_memory_info(&free_b, &total_b)) return UINT64_MAX;
    const uint64_t gib = 1073741824ull;
    if (total_b <= 128ull * gib && span_bytes >= 64ull * gib) {
        if (limited_out) *limited_out = true;
        (void)free_b;
        return 8ull * gib;
    }
    return UINT64_MAX;
}

static bool accelerator_cuda_startup_preload_explicit(void) {
    return sf37_env_present("SF37_CUDA_STARTUP_PRELOAD_GB", "DS4_CUDA_STARTUP_PRELOAD_GB") ||
           sf37_env_present("SF37_CUDA_WEIGHT_PRELOAD", "DS4_CUDA_WEIGHT_PRELOAD");
}

static uint64_t accelerator_span_total_bytes(const uint64_t *span_sizes, uint32_t span_count,
                                             const sf37_gguf *g) {
    if (span_count == 0) return g && g->size > g->data_offset ? g->size - g->data_offset : 0;
    uint64_t total = 0;
    for (uint32_t i = 0; i < span_count; i++) total += span_sizes[i];
    return total;
}

static bool accelerator_span_filter_contains(uint64_t off,
                                             uint64_t bytes,
                                             const uint64_t *span_offsets,
                                             const uint64_t *span_sizes,
                                             uint32_t span_count) {
    if (span_count == 0) return true;
    if (bytes == 0) return true;
    const uint64_t end = off + bytes;
    if (end < off) return false;
    for (uint32_t i = 0; i < span_count; i++) {
        const uint64_t span_end = span_offsets[i] + span_sizes[i];
        if (span_end < span_offsets[i]) return false;
        if (off >= span_offsets[i] && end <= span_end) return true;
    }
    return false;
}

static bool accelerator_prepare_model_tensor_spans(const sf37_gguf *g,
                                                   const uint64_t *span_offsets,
                                                   const uint64_t *span_sizes,
                                                   uint32_t span_count,
                                                   uint64_t *prepared_out) {
    if (!g || !g->map || g->size == 0) return false;
    if (g->n_tensors == 0) {
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    sf37_accelerator_tensor_span *spans =
        xcalloc((size_t)g->n_tensors, sizeof(spans[0]));
    uint64_t nspan = 0;
    for (uint32_t i = 0; i < span_count; i++) {
        if (span_offsets[i] > g->size ||
            span_sizes[i] == 0 ||
            span_sizes[i] > g->size - span_offsets[i]) {
            free(spans);
            return false;
        }
    }
    for (uint64_t i = 0; i < g->n_tensors; i++) {
        const sf37_tensor *t = &g->tensors[i];
        if (t->nbytes == 0) continue;
        if (t->abs_offset > g->size || t->nbytes > g->size - t->abs_offset) {
            free(spans);
            return false;
        }
        if (!accelerator_span_filter_contains(t->abs_offset, t->nbytes,
                                              span_offsets, span_sizes,
                                              span_count)) {
            continue;
        }
        spans[nspan++] = (sf37_accelerator_tensor_span){
            .off = t->abs_offset,
            .end = t->abs_offset + t->nbytes,
        };
    }
    if (nspan == 0) {
        free(spans);
        if (prepared_out) *prepared_out = 0;
        return true;
    }

    qsort(spans, (size_t)nspan, sizeof(spans[0]), accelerator_tensor_span_cmp);

    const uint64_t max_span = accelerator_cuda_preload_span_bytes();
    const int tty = isatty(STDERR_FILENO);
    const uint64_t progress_step = (tty ? 2ull : 16ull) * 1073741824ull;
    uint64_t next_progress = progress_step;
    double last_progress = sf37_now_sec();
    uint64_t prepared = 0;
    uint64_t merged = 0;
    bool startup_limit_present = false;
    uint64_t startup_limit =
        accelerator_cuda_startup_preload_limit_bytes(&startup_limit_present);
    bool default_limited = false;
    if (!startup_limit_present) {
        const uint64_t default_limit =
            accelerator_cuda_default_limited_preload_bytes(
                    accelerator_span_total_bytes(span_sizes, span_count, g),
                    &default_limited);
        if (default_limited) {
            startup_limit = default_limit;
            startup_limit_present = true;
            fprintf(stderr,
                    "sf37: CUDA startup residency default: limited preload %.2f GiB + layer-aware on-demand\n",
                    (double)startup_limit / 1073741824.0);
        }
    }

    fprintf(stderr, "%ssf37: CUDA preparing model tensor mappings%s",
            tty ? "\r\033[K" : "",
            tty ? ": 0.00 GiB" : "\n");
    fflush(stderr);

    for (uint64_t i = 0; i < nspan;) {
        uint64_t off = spans[i].off;
        uint64_t end = spans[i].end;
        i++;
        while (i < nspan &&
               spans[i].off <= end + 65536u &&
               spans[i].end - off <= max_span) {
            if (spans[i].end > end) end = spans[i].end;
            i++;
        }
        char label[96];
        snprintf(label, sizeof(label), "tensor-span:%" PRIu64, merged);
        if (startup_limit_present &&
            (prepared >= startup_limit || end - off > startup_limit - prepared)) {
            if (tty) fputc('\n', stderr);
            fprintf(stderr,
                    "sf37: CUDA startup preload budget reached after %.2f GiB; "
                    "uncached tensors will use mapped/on-demand access\n",
                    (double)prepared / 1073741824.0);
            break;
        }
        if (!sf37_cuda_cache_model_range(g->map, g->size, off, end - off, label)) {
            if (tty) fputc('\n', stderr);
            if (sf37_env_present("SF37_CUDA_STRICT_WEIGHT_CACHE",
                                 "DS4_CUDA_STRICT_WEIGHT_CACHE")) {
                fprintf(stderr,
                        "sf37: accelerator failed to prepare model tensor span %" PRIu64
                        " at offset %" PRIu64 "\n",
                        merged, off);
                free(spans);
                return false;
            }
            fprintf(stderr,
                    "sf37: CUDA optional startup preload stopped at tensor span %" PRIu64
                    " after %.2f GiB; uncached tensors will use mapped/on-demand access\n",
                    merged, (double)prepared / 1073741824.0);
            break;
        }
        prepared += end - off;
        merged++;

        const double now = sf37_now_sec();
        if (prepared >= next_progress || now - last_progress >= (tty ? 2.0 : 10.0)) {
            if (tty) {
                fprintf(stderr, "\r\033[Ksf37: CUDA preparing model tensor mappings: %.2f GiB",
                        (double)prepared / 1073741824.0);
            } else {
                fprintf(stderr, "sf37: CUDA prepared model tensor mappings %.2f GiB\n",
                        (double)prepared / 1073741824.0);
            }
            fflush(stderr);
            last_progress = now;
            while (next_progress <= prepared) next_progress += progress_step;
        }
    }

    if (tty) fputc('\n', stderr);
    free(spans);
    if (prepared_out) *prepared_out = prepared;
    return true;
}

static bool accelerator_cache_model_tensors(sf37_engine *e,
                                            const uint64_t *span_offsets,
                                            const uint64_t *span_sizes,
                                            uint32_t span_count) {
    if (!e || e->backend != SF37_BACKEND_CUDA) return true;
    if (!e->gguf.map || e->gguf.size == 0) return false;
    const uint64_t span_total = accelerator_span_total_bytes(span_sizes, span_count, &e->gguf);
    const char *q8_cache = sf37_env_present("SF37_CUDA_NO_Q8_F16_CACHE", "DS4_CUDA_NO_Q8_F16_CACHE") &&
                           sf37_env_present("SF37_CUDA_NO_Q8_F32_CACHE", "DS4_CUDA_NO_Q8_F32_CACHE")
                           ? "disabled" : "enabled";
    const bool direct_model = sf37_env_present("SF37_CUDA_DIRECT_MODEL", "DS4_CUDA_DIRECT_MODEL");
    const bool no_startup_preload =
        sf37_env_present("SF37_CUDA_NO_STARTUP_PRELOAD", "DS4_CUDA_NO_STARTUP_PRELOAD");
    const char *strategy = "limited preload + layer-aware on-demand";
    if (direct_model) {
        strategy = "direct mapped";
    } else if (no_startup_preload) {
        strategy = "layer-aware on-demand";
    } else if (accelerator_cuda_startup_preload_explicit()) {
        bool present = false;
        uint64_t lim = accelerator_cuda_startup_preload_limit_bytes(&present);
        if (!present || lim >= span_total) strategy = "full preload";
        else strategy = "limited preload";
    }
    fprintf(stderr,
            "sf37: CUDA residency strategy: %s (tensor_span=%.2f GiB, q8_expanded_cache=%s)\n",
            strategy,
            (double)span_total / 1073741824.0,
            q8_cache);

    if (direct_model) return true;
    if (no_startup_preload) {
        fprintf(stderr,
                "sf37: CUDA startup model preparation skipped by SF37_CUDA_NO_STARTUP_PRELOAD\n");
        return true;
    }

    const double t0 = sf37_now_sec();
    uint64_t prepared = 0;
    if (!accelerator_prepare_model_tensor_spans(&e->gguf, span_offsets, span_sizes,
                                                span_count, &prepared)) {
        return false;
    }
    const double t1 = sf37_now_sec();
    fprintf(stderr,
            "sf37: CUDA startup model preparation covered %.2f GiB of tensor spans in %.3fs\n",
            (double)prepared / 1073741824.0, t1 - t0);
    return true;
}

static bool sf37_cuda_prepare_model(sf37_engine *e) {
    if (!e || e->backend != SF37_BACKEND_CUDA) return true;
    (void)sf37_cuda_set_model_fd(e->gguf.fd);

    const bool include_vision =
        sf37_env_present("SF37_CUDA_PRELOAD_VISION", "DS4_CUDA_PRELOAD_VISION");
    sf37_model_map_span_vec spans;
    if (!sf37_weights_model_map_spans(e, true, include_vision, &spans)) {
        if (!sf37_cuda_set_model_map_range(e->gguf.map, e->gguf.size,
                                           e->gguf.data_offset,
                                           e->gguf.size - e->gguf.data_offset,
                                           0)) {
            return false;
        }
        return accelerator_cache_model_tensors(e, NULL, NULL, 0);
    }

    uint64_t *offsets = xcalloc(spans.len, sizeof(offsets[0]));
    uint64_t *sizes = xcalloc(spans.len, sizeof(sizes[0]));
    uint64_t span_bytes = 0;
    for (uint32_t i = 0; i < spans.len; i++) {
        offsets[i] = spans.v[i].off;
        sizes[i] = spans.v[i].end - spans.v[i].off;
        span_bytes += sizes[i];
    }

    fprintf(stderr,
            "sf37: restricting CUDA model map to text/output%s (%u spans, %.2f GiB tensor span)\n",
            include_vision ? "+vision" : "",
            spans.len,
            (double)span_bytes / 1073741824.0);

    const int map_ok = sf37_cuda_set_model_map_spans(e->gguf.map,
                                                     e->gguf.size,
                                                     offsets,
                                                     sizes,
                                                     spans.len,
                                                     spans.max_tensor_bytes);
    bool ok = map_ok != 0;
    if (ok) ok = accelerator_cache_model_tensors(e, offsets, sizes, spans.len);

    free(offsets);
    free(sizes);
    free(spans.v);
    return ok;
}

static sf37_cuda_tensor *cuda_tensor_from_host(const void *data, uint64_t bytes,
                                               const char *what) {
    sf37_cuda_tensor *t = sf37_cuda_tensor_alloc(bytes);
    if (!t) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA allocation failed for %s", what);
        return NULL;
    }
    if (!sf37_cuda_tensor_write(t, 0, data, bytes)) {
        sf37_cuda_tensor_free(t);
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA upload failed for %s", what);
        return NULL;
    }
    return t;
}

static void compare_vec(FILE *fp, const char *name, const float *a, const float *b, uint64_t n) {
    double ss = 0.0;
    float max_abs = 0.0f;
    uint64_t max_i = 0;
    for (uint64_t i = 0; i < n; i++) {
        const float d = fabsf(a[i] - b[i]);
        if (d > max_abs) {
            max_abs = d;
            max_i = i;
        }
        ss += (double)d * (double)d;
    }
    fprintf(fp, "%s: max_abs=%g at=%" PRIu64 " rms_abs=%g\n",
            name, max_abs, max_i, sqrt(ss / (double)n));
}

#define CUDA_CHECK(expr, label) do { \
    if (!(expr)) { \
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA smoke failed at %s", label); \
        rc = -1; \
        goto cleanup; \
    } \
} while (0)

static int cuda_matvec_dense_uploaded(sf37_cuda_tensor *out, const sf37_tensor *t,
                                      const sf37_cuda_tensor *weights,
                                      const sf37_cuda_tensor *x) {
    if (!out || !t || !weights || !x || t->ndim != 2) return 0;
    switch (t->type) {
    case SF37_TENSOR_Q8_0:
        return sf37_cuda_matvec_q8_0(out, weights, t->dim[0], t->dim[1], x);
    case SF37_TENSOR_BF16:
        return sf37_cuda_matvec_bf16(out, weights, t->dim[0], t->dim[1], x);
    case SF37_TENSOR_F32:
        return sf37_cuda_matvec_f32(out, weights, t->dim[0], t->dim[1], x);
    default:
        break;
    }
    return 0;
}

static int cuda_matvec_dense_mapped(sf37_cuda_tensor *out, const sf37_engine *e,
                                    const sf37_tensor *t,
                                    const sf37_cuda_tensor *x) {
    if (!out || !e || !t || !x || t->ndim != 2) return 0;
    switch (t->type) {
    case SF37_TENSOR_Q8_0:
        return sf37_cuda_matvec_q8_0_mapped(out, e->gguf.map, e->gguf.size,
                                            t->abs_offset, t->dim[0], t->dim[1], x);
    case SF37_TENSOR_BF16:
        return sf37_cuda_matvec_bf16_mapped(out, e->gguf.map, e->gguf.size,
                                            t->abs_offset, t->dim[0], t->dim[1], x);
    case SF37_TENSOR_F32:
        return sf37_cuda_matvec_f32_mapped(out, e->gguf.map, e->gguf.size,
                                           t->abs_offset, t->dim[0], t->dim[1], x);
    default:
        break;
    }
    return 0;
}

static int cuda_matmul_dense_mapped(sf37_cuda_tensor *out, const sf37_engine *e,
                                    const sf37_tensor *t,
                                    const sf37_cuda_tensor *x,
                                    uint32_t n_tok) {
    if (!out || !e || !t || !x || t->ndim != 2 || n_tok == 0) return 0;
    switch (t->type) {
    case SF37_TENSOR_Q8_0:
        return sf37_cuda_matmul_q8_0_mapped(out, e->gguf.map, e->gguf.size,
                                            t->abs_offset, t->dim[0], t->dim[1], x, n_tok);
    case SF37_TENSOR_BF16:
        return sf37_cuda_matmul_bf16_mapped(out, e->gguf.map, e->gguf.size,
                                            t->abs_offset, t->dim[0], t->dim[1], x, n_tok);
    case SF37_TENSOR_F32:
        return sf37_cuda_matmul_f32_mapped(out, e->gguf.map, e->gguf.size,
                                           t->abs_offset, t->dim[0], t->dim[1], x, n_tok);
    default:
        break;
    }
    return 0;
}

typedef struct {
    sf37_cuda_tensor *hidden;
    sf37_cuda_tensor *attn_norm;
    sf37_cuda_tensor *q;
    sf37_cuda_tensor *k;
    sf37_cuda_tensor *v;
    sf37_cuda_tensor *head_gate;
    sf37_cuda_tensor *attn_heads;
    sf37_cuda_tensor *attn_out;
    sf37_cuda_tensor *ffn_norm;
    sf37_cuda_tensor *gate;
    sf37_cuda_tensor *up;
    sf37_cuda_tensor *mid;
    sf37_cuda_tensor *ffn_out;
    sf37_cuda_tensor *router_logits;
    sf37_cuda_tensor *router_probs;
    sf37_cuda_tensor *router_selected;
    sf37_cuda_tensor *router_weights;
    sf37_cuda_tensor *routed_down;
    sf37_cuda_tensor *routed_out;
    sf37_cuda_tensor *output_norm;
    sf37_cuda_tensor *logits;
    sf37_cuda_tensor *k_cache[SF37_MAIN_LAYERS];
    sf37_cuda_tensor *v_cache[SF37_MAIN_LAYERS];
    uint32_t cache_cap[SF37_MAIN_LAYERS];
    uint32_t ctx_size;
    uint32_t prefill_cap;
    uint64_t kv_cache_bytes;
    bool managed_kv_cache;
} sf37_cuda_decode_state;

typedef struct {
    sf37_cuda_tensor *tokens;
    sf37_cuda_tensor *hidden;
    sf37_cuda_tensor *attn_norm;
    sf37_cuda_tensor *q;
    sf37_cuda_tensor *k;
    sf37_cuda_tensor *v;
    sf37_cuda_tensor *head_gate;
    sf37_cuda_tensor *attn_heads;
    sf37_cuda_tensor *attn_out;
    sf37_cuda_tensor *ffn_norm;
    sf37_cuda_tensor *gate;
    sf37_cuda_tensor *up;
    sf37_cuda_tensor *mid;
    sf37_cuda_tensor *ffn_out;
    sf37_cuda_tensor *router_logits;
    sf37_cuda_tensor *router_probs;
    sf37_cuda_tensor *router_selected;
    sf37_cuda_tensor *router_weights;
    sf37_cuda_tensor *routed_out;
    sf37_cuda_tensor *routed_gate;
    sf37_cuda_tensor *routed_up;
    sf37_cuda_tensor *routed_mid;
    sf37_cuda_tensor *routed_down;
    sf37_cuda_tensor *image_in;
    sf37_cuda_tensor *image_proj;
    uint32_t cap;
    uint32_t image_cap;
    uint32_t image_dim;
} sf37_cuda_prefill_state;

static void cuda_decode_state_free(sf37_cuda_decode_state *s) {
    if (!s) return;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        sf37_cuda_tensor_free(s->v_cache[il]);
        sf37_cuda_tensor_free(s->k_cache[il]);
    }
    sf37_cuda_tensor_free(s->routed_out);
    sf37_cuda_tensor_free(s->routed_down);
    sf37_cuda_tensor_free(s->router_weights);
    sf37_cuda_tensor_free(s->router_selected);
    sf37_cuda_tensor_free(s->router_probs);
    sf37_cuda_tensor_free(s->router_logits);
    sf37_cuda_tensor_free(s->logits);
    sf37_cuda_tensor_free(s->output_norm);
    sf37_cuda_tensor_free(s->ffn_out);
    sf37_cuda_tensor_free(s->mid);
    sf37_cuda_tensor_free(s->up);
    sf37_cuda_tensor_free(s->gate);
    sf37_cuda_tensor_free(s->ffn_norm);
    sf37_cuda_tensor_free(s->attn_out);
    sf37_cuda_tensor_free(s->attn_heads);
    sf37_cuda_tensor_free(s->head_gate);
    sf37_cuda_tensor_free(s->v);
    sf37_cuda_tensor_free(s->k);
    sf37_cuda_tensor_free(s->q);
    sf37_cuda_tensor_free(s->attn_norm);
    sf37_cuda_tensor_free(s->hidden);
    memset(s, 0, sizeof(*s));
}

static void cuda_prefill_state_free(sf37_cuda_prefill_state *s) {
    if (!s) return;
    sf37_cuda_tensor_free(s->image_proj);
    sf37_cuda_tensor_free(s->image_in);
    sf37_cuda_tensor_free(s->routed_down);
    sf37_cuda_tensor_free(s->routed_mid);
    sf37_cuda_tensor_free(s->routed_up);
    sf37_cuda_tensor_free(s->routed_gate);
    sf37_cuda_tensor_free(s->routed_out);
    sf37_cuda_tensor_free(s->router_weights);
    sf37_cuda_tensor_free(s->router_selected);
    sf37_cuda_tensor_free(s->router_probs);
    sf37_cuda_tensor_free(s->router_logits);
    sf37_cuda_tensor_free(s->ffn_out);
    sf37_cuda_tensor_free(s->mid);
    sf37_cuda_tensor_free(s->up);
    sf37_cuda_tensor_free(s->gate);
    sf37_cuda_tensor_free(s->ffn_norm);
    sf37_cuda_tensor_free(s->attn_out);
    sf37_cuda_tensor_free(s->attn_heads);
    sf37_cuda_tensor_free(s->head_gate);
    sf37_cuda_tensor_free(s->v);
    sf37_cuda_tensor_free(s->k);
    sf37_cuda_tensor_free(s->q);
    sf37_cuda_tensor_free(s->attn_norm);
    sf37_cuda_tensor_free(s->hidden);
    sf37_cuda_tensor_free(s->tokens);
    memset(s, 0, sizeof(*s));
}

static uint32_t cuda_prefill_default_chunk(uint32_t suffix);

static uint32_t sf37_align_up_u32(uint32_t v, uint32_t align) {
    if (align == 0) return v;
    const uint32_t rem = v % align;
    if (rem == 0) return v;
    if (v > UINT32_MAX - (align - rem)) return UINT32_MAX;
    return v + (align - rem);
}

static uint32_t cuda_sliding_cache_cap(uint32_t ctx_size, uint32_t prefill_cap) {
    if (ctx_size == 0) return 1;
    uint32_t raw_window = SF37_SLIDING_WINDOW;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;

    uint64_t wanted = (uint64_t)raw_window + (uint64_t)prefill_cap;
    if (wanted > ctx_size) wanted = ctx_size;
    if (wanted == 0) wanted = 1;
    wanted = sf37_align_up_u32((uint32_t)wanted, 256u);
    if (wanted > ctx_size) wanted = ctx_size;
    if (wanted > SF37_CUDA_SLIDING_CACHE_MAX) wanted = SF37_CUDA_SLIDING_CACHE_MAX;
    if (wanted < raw_window) wanted = raw_window;
    return (uint32_t)wanted;
}

static uint32_t cuda_sliding_cache_prefill_cap(uint32_t ctx_size, uint32_t cache_cap) {
    if (cache_cap == 0) return 0;
    uint32_t raw_window = SF37_SLIDING_WINDOW;
    if (raw_window > ctx_size) raw_window = ctx_size;
    if (raw_window == 0) raw_window = 1;
    if (ctx_size <= raw_window) return cache_cap;
    if (cache_cap <= raw_window) return 1;
    return cache_cap - raw_window;
}

static uint64_t cuda_decode_kv_bytes_for_cap(uint32_t cap) {
    const uint64_t kv_dim = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    return (uint64_t)cap * kv_dim * 2u * sizeof(float);
}

static uint64_t cuda_prefill_scratch_bytes(uint32_t cap) {
    if (cap == 0) return 0;
    const uint64_t cap64 = cap;
    const uint64_t embd = SF37_EMBD;
    const uint64_t max_q_dim = (uint64_t)SF37_SLIDING_Q_HEADS * SF37_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    uint64_t bytes = 0;
    bytes += cap64 * sizeof(int32_t);
    bytes += cap64 * embd * sizeof(float) * 2u;
    bytes += cap64 * max_q_dim * sizeof(float) * 2u;
    bytes += cap64 * kv_dim * sizeof(float) * 2u;
    bytes += cap64 * SF37_SLIDING_Q_HEADS * sizeof(float);
    bytes += cap64 * embd * sizeof(float) * 3u;
    bytes += cap64 * SF37_DENSE_FF * sizeof(float) * 3u;
    bytes += cap64 * SF37_EXPERTS * sizeof(float) * 2u;
    bytes += cap64 * SF37_EXPERT_USED * sizeof(int32_t);
    bytes += cap64 * SF37_EXPERT_USED * sizeof(float);
    bytes += (uint64_t)SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float) * 3u;
    bytes += (uint64_t)SF37_EXPERT_USED * SF37_EMBD * sizeof(float);
    return bytes;
}

static uint64_t cuda_decode_scratch_bytes(void) {
    const uint64_t embd_bytes = (uint64_t)SF37_EMBD * sizeof(float);
    const uint64_t max_q_dim = (uint64_t)SF37_SLIDING_Q_HEADS * SF37_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    uint64_t bytes = 0;
    bytes += embd_bytes * 7u;
    bytes += max_q_dim * sizeof(float) * 2u;
    bytes += kv_dim * sizeof(float) * 2u;
    bytes += (uint64_t)SF37_SLIDING_Q_HEADS * sizeof(float);
    bytes += (uint64_t)SF37_DENSE_FF * sizeof(float) * 3u;
    bytes += (uint64_t)SF37_EXPERTS * sizeof(float) * 2u;
    bytes += (uint64_t)SF37_EXPERT_USED * sizeof(int32_t);
    bytes += (uint64_t)SF37_EXPERT_USED * sizeof(float);
    bytes += (uint64_t)SF37_EXPERT_USED * SF37_EMBD * sizeof(float);
    bytes += (uint64_t)SF37_VOCAB * sizeof(float);
    return bytes;
}

static sf37_cuda_tensor *cuda_alloc_kv_cache_tensor(bool managed, uint64_t bytes) {
    return managed ? sf37_cuda_tensor_alloc_managed(bytes) : sf37_cuda_tensor_alloc(bytes);
}

static int cuda_prefill_state_init(sf37_cuda_prefill_state *s, uint32_t cap) {
    memset(s, 0, sizeof(*s));
    if (cap == 0) return 0;
    s->cap = cap;
    const uint64_t embd = SF37_EMBD;
    const uint64_t max_q_dim = (uint64_t)SF37_SLIDING_Q_HEADS * SF37_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint64_t cap64 = cap;
    s->tokens = sf37_cuda_tensor_alloc(cap64 * sizeof(int32_t));
    s->hidden = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->attn_norm = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->q = sf37_cuda_tensor_alloc(cap64 * max_q_dim * sizeof(float));
    s->k = sf37_cuda_tensor_alloc(cap64 * kv_dim * sizeof(float));
    s->v = sf37_cuda_tensor_alloc(cap64 * kv_dim * sizeof(float));
    s->head_gate = sf37_cuda_tensor_alloc(cap64 * SF37_SLIDING_Q_HEADS * sizeof(float));
    s->attn_heads = sf37_cuda_tensor_alloc(cap64 * max_q_dim * sizeof(float));
    s->attn_out = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->ffn_norm = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->gate = sf37_cuda_tensor_alloc(cap64 * SF37_DENSE_FF * sizeof(float));
    s->up = sf37_cuda_tensor_alloc(cap64 * SF37_DENSE_FF * sizeof(float));
    s->mid = sf37_cuda_tensor_alloc(cap64 * SF37_DENSE_FF * sizeof(float));
    s->ffn_out = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->router_logits = sf37_cuda_tensor_alloc(cap64 * SF37_EXPERTS * sizeof(float));
    s->router_probs = sf37_cuda_tensor_alloc(cap64 * SF37_EXPERTS * sizeof(float));
    s->router_selected = sf37_cuda_tensor_alloc(cap64 * SF37_EXPERT_USED * sizeof(int32_t));
    s->router_weights = sf37_cuda_tensor_alloc(cap64 * SF37_EXPERT_USED * sizeof(float));
    s->routed_out = sf37_cuda_tensor_alloc(cap64 * embd * sizeof(float));
    s->routed_gate = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float));
    s->routed_up = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float));
    s->routed_mid = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float));
    s->routed_down = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * SF37_EMBD * sizeof(float));
    if (!s->tokens || !s->hidden || !s->attn_norm || !s->q || !s->k || !s->v ||
        !s->head_gate || !s->attn_heads || !s->attn_out || !s->ffn_norm ||
        !s->gate || !s->up || !s->mid || !s->ffn_out || !s->router_logits ||
        !s->router_probs || !s->router_selected || !s->router_weights ||
        !s->routed_out || !s->routed_gate || !s->routed_up || !s->routed_mid ||
        !s->routed_down) {
        cuda_prefill_state_free(s);
        return 0;
    }
    return 1;
}

static int cuda_decode_state_init(sf37_cuda_decode_state *s, uint32_t ctx_size) {
    memset(s, 0, sizeof(*s));
    if (ctx_size == 0) ctx_size = 1;
    if (ctx_size > SF37_CTX) ctx_size = SF37_CTX;
    s->ctx_size = ctx_size;

    const uint64_t embd_bytes = (uint64_t)SF37_EMBD * sizeof(float);
    const uint64_t max_q_dim = (uint64_t)SF37_SLIDING_Q_HEADS * SF37_HEAD_DIM;
    const uint64_t kv_dim = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint64_t dense_ff_bytes = (uint64_t)SF37_DENSE_FF * sizeof(float);
    const uint64_t routed_down_bytes = (uint64_t)SF37_EXPERT_USED * SF37_EMBD * sizeof(float);
    const uint32_t requested_prefill_cap = cuda_prefill_default_chunk(ctx_size);
    const uint32_t sliding_cap = cuda_sliding_cache_cap(ctx_size, requested_prefill_cap);
    uint32_t full_layers = 0;
    uint32_t sliding_layers = 0;
    s->prefill_cap = cuda_sliding_cache_prefill_cap(ctx_size, sliding_cap);
    if (s->prefill_cap == 0) s->prefill_cap = 1;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (layer_is_full(il)) {
            s->cache_cap[il] = ctx_size;
            full_layers++;
        } else {
            s->cache_cap[il] = sliding_cap;
            sliding_layers++;
        }
        s->kv_cache_bytes += cuda_decode_kv_bytes_for_cap(s->cache_cap[il]);
    }
    const uint64_t context_bytes = s->kv_cache_bytes +
                                   cuda_decode_scratch_bytes() +
                                   cuda_prefill_scratch_bytes(s->prefill_cap);
    s->managed_kv_cache =
        sf37_cuda_should_use_managed_kv_cache(s->kv_cache_bytes, context_bytes) != 0;
    fprintf(stderr,
            "sf37: CUDA KV cache ctx=%u full_layers=%u full_cap=%u "
            "sliding_layers=%u sliding_cap=%u prefill_cap=%u kv=%.2f GiB%s\n",
            ctx_size,
            full_layers,
            ctx_size,
            sliding_layers,
            sliding_cap,
            s->prefill_cap,
            (double)s->kv_cache_bytes / 1073741824.0,
            s->managed_kv_cache ? " managed" : "");
    if (s->managed_kv_cache) {
        fprintf(stderr,
                "sf37: CUDA using managed KV cache "
                "(kv cache %.2f GiB, context buffers %.2f GiB)\n",
                (double)s->kv_cache_bytes / 1073741824.0,
                (double)context_bytes / 1073741824.0);
    }

    s->hidden = sf37_cuda_tensor_alloc(embd_bytes);
    s->attn_norm = sf37_cuda_tensor_alloc(embd_bytes);
    s->q = sf37_cuda_tensor_alloc(max_q_dim * sizeof(float));
    s->k = sf37_cuda_tensor_alloc(kv_dim * sizeof(float));
    s->v = sf37_cuda_tensor_alloc(kv_dim * sizeof(float));
    s->head_gate = sf37_cuda_tensor_alloc((uint64_t)SF37_SLIDING_Q_HEADS * sizeof(float));
    s->attn_heads = sf37_cuda_tensor_alloc(max_q_dim * sizeof(float));
    s->attn_out = sf37_cuda_tensor_alloc(embd_bytes);
    s->ffn_norm = sf37_cuda_tensor_alloc(embd_bytes);
    s->gate = sf37_cuda_tensor_alloc(dense_ff_bytes);
    s->up = sf37_cuda_tensor_alloc(dense_ff_bytes);
    s->mid = sf37_cuda_tensor_alloc(dense_ff_bytes);
    s->ffn_out = sf37_cuda_tensor_alloc(embd_bytes);
    s->router_logits = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERTS * sizeof(float));
    s->router_probs = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERTS * sizeof(float));
    s->router_selected = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * sizeof(int32_t));
    s->router_weights = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * sizeof(float));
    s->routed_down = sf37_cuda_tensor_alloc(routed_down_bytes);
    s->routed_out = sf37_cuda_tensor_alloc(embd_bytes);
    s->output_norm = sf37_cuda_tensor_alloc(embd_bytes);
    s->logits = sf37_cuda_tensor_alloc((uint64_t)SF37_VOCAB * sizeof(float));
    if (!s->hidden || !s->attn_norm || !s->q || !s->k || !s->v || !s->head_gate ||
        !s->attn_heads || !s->attn_out || !s->ffn_norm || !s->gate || !s->up ||
        !s->mid || !s->ffn_out || !s->router_logits || !s->router_probs ||
        !s->router_selected || !s->router_weights || !s->routed_down || !s->routed_out ||
        !s->output_norm || !s->logits) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA decode scratch allocation failed");
        cuda_decode_state_free(s);
        return 0;
    }

    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        const uint32_t cap = s->cache_cap[il];
        s->k_cache[il] = cuda_alloc_kv_cache_tensor(s->managed_kv_cache,
                                                    (uint64_t)cap * kv_dim * sizeof(float));
        s->v_cache[il] = cuda_alloc_kv_cache_tensor(s->managed_kv_cache,
                                                    (uint64_t)cap * kv_dim * sizeof(float));
        if (!s->k_cache[il] || !s->v_cache[il]) {
            sf37_log(stderr, SF37_LOG_ERROR, "CUDA KV cache allocation failed for layer %u", il);
            cuda_decode_state_free(s);
            return 0;
        }
    }
    return 1;
}

static void cuda_preload_layer_weights(sf37_engine *e, uint32_t il);
static void cuda_preload_output_weights(sf37_engine *e);

static int cuda_decode_layer(sf37_engine *e, sf37_cuda_decode_state *s,
                             uint32_t il, uint32_t pos) {
    sf37_layer_weights *l = &e->layer[il];
    const uint32_t q_heads = layer_q_heads(il);
    const uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    const uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t rotary_dim = layer_rotary_dim(il);
    const double theta = layer_rope_theta(il);
    const int full = layer_is_full(il) ? 1 : 0;
    const uint32_t cache_cap = s->cache_cap[il] ? s->cache_cap[il] : 1u;
    int ok = 1;
    (void)sf37_cuda_begin_layer(il);
    cuda_preload_layer_weights(e, il);

#define CUDA_LAYER_CHECK(expr, label) do { \
        if (!(expr)) { \
            sf37_log(stderr, SF37_LOG_ERROR, "CUDA decode layer %u failed at %s", il, label); \
            ok = 0; \
            goto cleanup; \
        } \
    } while (0)

    CUDA_LAYER_CHECK(sf37_cuda_rms_norm_weight1_bf16_mapped(s->attn_norm, s->hidden,
                                                            e->gguf.map, e->gguf.size,
                                                            l->input_norm->abs_offset,
                                                            SF37_EMBD, SF37_RMS_EPS),
                     "input rms");
    CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->q, e->gguf.map, e->gguf.size,
                                                  l->q_proj->abs_offset,
                                                  SF37_EMBD, q_dim, s->attn_norm), "q_proj");
    CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->k, e->gguf.map, e->gguf.size,
                                                  l->k_proj->abs_offset,
                                                  SF37_EMBD, kv_dim, s->attn_norm), "k_proj");
    CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->v, e->gguf.map, e->gguf.size,
                                                  l->v_proj->abs_offset,
                                                  SF37_EMBD, kv_dim, s->attn_norm), "v_proj");
    CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->head_gate, e->gguf.map, e->gguf.size,
                                                  l->g_proj->abs_offset,
                                                  SF37_EMBD, q_heads, s->attn_norm), "g_proj");
    CUDA_LAYER_CHECK(sf37_cuda_head_rms_norm_weight1_bf16_mapped(s->q,
                                                                 e->gguf.map, e->gguf.size,
                                                                 l->q_norm->abs_offset,
                                                                 q_heads, SF37_HEAD_DIM,
                                                                 SF37_RMS_EPS), "q head norm");
    CUDA_LAYER_CHECK(sf37_cuda_head_rms_norm_weight1_bf16_mapped(s->k,
                                                                 e->gguf.map, e->gguf.size,
                                                                 l->k_norm->abs_offset,
                                                                 SF37_KV_HEADS, SF37_HEAD_DIM,
                                                                 SF37_RMS_EPS), "k head norm");
    CUDA_LAYER_CHECK(sf37_cuda_rope_split_half(s->q, q_heads, SF37_HEAD_DIM,
                                               rotary_dim, theta, full, pos), "q rope");
    CUDA_LAYER_CHECK(sf37_cuda_rope_split_half(s->k, SF37_KV_HEADS, SF37_HEAD_DIM,
                                               rotary_dim, theta, full, pos), "k rope");
    CUDA_LAYER_CHECK(sf37_cuda_store_kv_cache_batch(s->k_cache[il], s->v_cache[il],
                                                    s->k, s->v, pos, 1u,
                                                    cache_cap, kv_dim),
                     "kv cache store");
    CUDA_LAYER_CHECK(sf37_cuda_attention_decode_heads_at(s->attn_heads, s->q,
                                                         s->k_cache[il], s->v_cache[il],
                                                         s->head_gate,
                                                         pos, cache_cap, q_heads,
                                                         SF37_KV_HEADS, SF37_HEAD_DIM,
                                                         full ? 0 : 1, SF37_SLIDING_WINDOW),
                     "attention decode");
    CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->attn_out, e->gguf.map, e->gguf.size,
                                                  l->o_proj->abs_offset,
                                                  q_dim, SF37_EMBD, s->attn_heads), "o_proj");
    CUDA_LAYER_CHECK(sf37_cuda_add_inplace_f32(s->hidden, s->attn_out, SF37_EMBD), "attention residual");

    CUDA_LAYER_CHECK(sf37_cuda_rms_norm_weight1_bf16_mapped(s->ffn_norm, s->hidden,
                                                            e->gguf.map, e->gguf.size,
                                                            l->post_norm->abs_offset,
                                                            SF37_EMBD, SF37_RMS_EPS),
                     "post rms");
    if (il < 3u) {
        CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_pair_mapped(s->gate, s->up,
                                                           e->gguf.map, e->gguf.size,
                                                           l->mlp_gate->abs_offset,
                                                           l->mlp_up->abs_offset,
                                                           SF37_EMBD, SF37_DENSE_FF,
                                                           s->ffn_norm),
                         "dense gate/up");
        CUDA_LAYER_CHECK(sf37_cuda_swiglu_f32(s->mid, s->gate, s->up, SF37_DENSE_FF,
                                              layer_swiglu_limit(il, true)), "dense swiglu");
        CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->ffn_out,
                                                      e->gguf.map, e->gguf.size,
                                                      l->mlp_down->abs_offset,
                                                      SF37_DENSE_FF, SF37_EMBD, s->mid),
                         "dense down");
    } else {
        CUDA_LAYER_CHECK(cuda_matvec_dense_mapped(s->router_logits, e, l->router_gate,
                                                  s->ffn_norm), "router gate");
        CUDA_LAYER_CHECK(sf37_cuda_router_select_mapped(s->router_selected, s->router_weights,
                                                        s->router_probs, s->router_logits,
                                                        e->gguf.map, e->gguf.size,
                                                        l->router_bias->abs_offset,
                                                        SF37_EXPERTS,
                                                        SF37_EXPERT_USED, 3.0f),
                         "router select");
        CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_pair_mapped(s->gate, s->up,
                                                           e->gguf.map, e->gguf.size,
                                                           l->share_gate->abs_offset,
                                                           l->share_up->abs_offset,
                                                           SF37_EMBD, SF37_EXPERT_FF,
                                                           s->ffn_norm),
                         "shared gate/up");
        CUDA_LAYER_CHECK(sf37_cuda_swiglu_f32(s->mid, s->gate, s->up, SF37_EXPERT_FF,
                                              layer_swiglu_limit(il, true)), "shared swiglu");
        CUDA_LAYER_CHECK(sf37_cuda_matvec_q8_0_mapped(s->ffn_out,
                                                      e->gguf.map, e->gguf.size,
                                                      l->share_down->abs_offset,
                                                      SF37_EXPERT_FF, SF37_EMBD, s->mid),
                         "shared down");
        CUDA_LAYER_CHECK(sf37_cuda_routed_moe_one_mapped(s->routed_out, s->gate, s->up,
                                                         s->mid, s->routed_down,
                                                         e->gguf.map, e->gguf.size,
                                                         l->moe_gate->abs_offset,
                                                         l->moe_up->abs_offset,
                                                         l->moe_down->abs_offset,
                                                         s->router_selected, s->router_weights,
                                                         SF37_EXPERTS, SF37_EXPERT_USED,
                                                         SF37_EMBD, SF37_EXPERT_FF, SF37_EMBD,
                                                         layer_swiglu_limit(il, false),
                                                         s->ffn_norm),
                         "routed moe");
        CUDA_LAYER_CHECK(sf37_cuda_add_inplace_f32(s->ffn_out, s->routed_out, SF37_EMBD),
                         "shared+routed add");
    }
    CUDA_LAYER_CHECK(sf37_cuda_add_inplace_f32(s->hidden, s->ffn_out, SF37_EMBD), "ffn residual");

cleanup:
#undef CUDA_LAYER_CHECK
    sf37_cuda_end_layer();
    return ok;
}

static int cuda_output_logits(sf37_engine *e, sf37_cuda_decode_state *s) {
    if (!e || !s || !s->hidden || !s->output_norm || !s->logits) return 0;
    (void)sf37_cuda_begin_layer(SF37_MAIN_LAYERS);
    cuda_preload_output_weights(e);
    if (!sf37_cuda_rms_norm_weight1_bf16_mapped(s->output_norm, s->hidden,
                                                e->gguf.map, e->gguf.size,
                                                e->final_norm->abs_offset,
                                                SF37_EMBD, SF37_RMS_EPS)) {
        sf37_cuda_end_layer();
        return 0;
    }
    int ok = sf37_cuda_matvec_bf16_mapped(s->logits,
                                          e->gguf.map, e->gguf.size,
                                          e->lm_head->abs_offset,
                                          SF37_EMBD, SF37_VOCAB,
                                          s->output_norm);
    sf37_cuda_end_layer();
    return ok;
}

static int cuda_output_argmax(sf37_engine *e, sf37_cuda_decode_state *s,
                              int excluded_id, int *out_token) {
    if (!e || !s || !s->hidden || !s->output_norm || !s->logits || !out_token) return 0;
    if (sf37_env_present("SF37_CUDA_NO_LM_HEAD_GPU_ARGMAX",
                         "DS4_CUDA_NO_LM_HEAD_GPU_ARGMAX")) return 0;
    (void)sf37_cuda_begin_layer(SF37_MAIN_LAYERS);
    cuda_preload_output_weights(e);
    if (!sf37_cuda_rms_norm_weight1_bf16_mapped(s->output_norm, s->hidden,
                                                e->gguf.map, e->gguf.size,
                                                e->final_norm->abs_offset,
                                                SF37_EMBD, SF37_RMS_EPS)) {
        sf37_cuda_end_layer();
        return 0;
    }
    int32_t tok = 0;
    int ok = 0;
    if (sf37_env_present("SF37_CUDA_LM_HEAD_FUSED_ARGMAX", NULL) &&
        !sf37_env_present("SF37_CUDA_NO_LM_HEAD_FUSED_ARGMAX",
                          "DS4_CUDA_NO_LM_HEAD_FUSED_ARGMAX")) {
        ok = sf37_cuda_matvec_bf16_argmax_mapped(&tok,
                                                 e->gguf.map, e->gguf.size,
                                                 e->lm_head->abs_offset,
                                                 SF37_EMBD, SF37_VOCAB,
                                                 s->output_norm,
                                                 (int32_t)excluded_id);
    }
    if (!ok) {
        ok = sf37_cuda_matvec_bf16_mapped(s->logits,
                                          e->gguf.map, e->gguf.size,
                                          e->lm_head->abs_offset,
                                          SF37_EMBD, SF37_VOCAB,
                                          s->output_norm) &&
             sf37_cuda_argmax_f32(&tok, s->logits, SF37_VOCAB,
                                  (int32_t)excluded_id);
    }
    sf37_cuda_end_layer();
    if (!ok) return 0;
    *out_token = (int)tok;
    return 1;
}

static uint32_t cuda_prefill_q8_cache_min_tokens(void) {
    uint32_t v = 8;
    const char *env = sf37_env_value("SF37_CUDA_PREFILL_Q8_CACHE_MIN_TOKENS",
                                     "DS4_PREFILL_BATCH");
    if (env) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed <= 262144ul) v = (uint32_t)parsed;
    }
    return v;
}

static bool cuda_prefill_q8_cache_enabled(uint32_t suffix_tokens) {
    if (sf37_env_present("SF37_CUDA_NO_PREFILL_Q8_CACHE",
                         "DS4_CUDA_NO_PREFILL_Q8_CACHE")) {
        return false;
    }
    return suffix_tokens >= cuda_prefill_q8_cache_min_tokens();
}

static void cuda_prefill_cache_q8_tensor(sf37_engine *e,
                                         const sf37_tensor *t,
                                         const char *label,
                                         uint32_t *attempts,
                                         uint64_t *bytes) {
    static int warned = 0;
    if (!e || !t || t->type != SF37_TENSOR_Q8_0 || t->ndim != 2) return;
    if (attempts) (*attempts)++;
    if (bytes) *bytes += t->nbytes;
    if (!sf37_cuda_cache_q8_f16_range(e->gguf.map,
                                      e->gguf.size,
                                      t->abs_offset,
                                      t->nbytes,
                                      t->dim[0],
                                      t->dim[1],
                                      label)) {
        if (sf37_env_present("SF37_CUDA_STRICT_PREFILL_Q8_CACHE", NULL)) {
            sf37_log(stderr, SF37_LOG_ERROR,
                     "CUDA prefill Q8 cache failed for %s", label ? label : "q8 tensor");
        } else if (!warned) {
            warned = 1;
            sf37_log(stderr, SF37_LOG_WARNING,
                     "CUDA prefill Q8 cache skipped after a cache failure; mapped Q8 kernels remain enabled");
        }
    }
}

static void cuda_prefill_cache_layer_q8(sf37_engine *e,
                                        uint32_t il,
                                        uint32_t *attempts,
                                        uint64_t *bytes) {
    sf37_layer_weights *l = &e->layer[il];
    char label[64];
#define CACHE_Q8_TENSOR(member) do { \
        snprintf(label, sizeof(label), "prefill.l%u.%s", il, #member); \
        cuda_prefill_cache_q8_tensor(e, l->member, label, attempts, bytes); \
    } while (0)
    CACHE_Q8_TENSOR(q_proj);
    CACHE_Q8_TENSOR(k_proj);
    CACHE_Q8_TENSOR(v_proj);
    CACHE_Q8_TENSOR(g_proj);
    CACHE_Q8_TENSOR(o_proj);
    if (il < 3u) {
        CACHE_Q8_TENSOR(mlp_gate);
        CACHE_Q8_TENSOR(mlp_up);
        CACHE_Q8_TENSOR(mlp_down);
    } else {
        CACHE_Q8_TENSOR(router_gate);
        CACHE_Q8_TENSOR(share_gate);
        CACHE_Q8_TENSOR(share_up);
        CACHE_Q8_TENSOR(share_down);
    }
#undef CACHE_Q8_TENSOR
}

static bool cuda_layer_aware_preload_enabled(void) {
    return !sf37_env_present("SF37_CUDA_NO_LAYER_AWARE_PRELOAD",
                             "DS4_CUDA_NO_LAYER_AWARE_PRELOAD");
}

static bool cuda_moe_selected_cache_enabled(void) {
    return !sf37_env_present("SF37_CUDA_NO_MOE_SELECTED_CACHE",
                             "DS4_CUDA_NO_MOE_SELECTED_CACHE");
}

static void cuda_preload_tensor_range(sf37_engine *e,
                                      const sf37_tensor *t,
                                      const char *label) {
    if (!e || !t || t->nbytes == 0) return;
    (void)sf37_cuda_cache_model_range(e->gguf.map,
                                      e->gguf.size,
                                      t->abs_offset,
                                      t->nbytes,
                                      label);
}

static void cuda_preload_layer_weights(sf37_engine *e, uint32_t il) {
    if (!e || il >= SF37_MAIN_LAYERS || !cuda_layer_aware_preload_enabled()) return;
    sf37_layer_weights *l = &e->layer[il];
    char label[80];
#define PRELOAD_TENSOR(member) do { \
        snprintf(label, sizeof(label), "layer%02u.%s", il, #member); \
        cuda_preload_tensor_range(e, l->member, label); \
    } while (0)
    PRELOAD_TENSOR(input_norm);
    PRELOAD_TENSOR(q_proj);
    PRELOAD_TENSOR(k_proj);
    PRELOAD_TENSOR(v_proj);
    PRELOAD_TENSOR(g_proj);
    PRELOAD_TENSOR(q_norm);
    PRELOAD_TENSOR(k_norm);
    PRELOAD_TENSOR(o_proj);
    PRELOAD_TENSOR(post_norm);
    if (il < 3u) {
        PRELOAD_TENSOR(mlp_gate);
        PRELOAD_TENSOR(mlp_up);
        PRELOAD_TENSOR(mlp_down);
    } else {
        PRELOAD_TENSOR(router_gate);
        PRELOAD_TENSOR(router_bias);
        PRELOAD_TENSOR(share_gate);
        PRELOAD_TENSOR(share_up);
        PRELOAD_TENSOR(share_down);
        if (!cuda_moe_selected_cache_enabled()) {
            PRELOAD_TENSOR(moe_gate);
            PRELOAD_TENSOR(moe_up);
            PRELOAD_TENSOR(moe_down);
        }
    }
#undef PRELOAD_TENSOR
}

static void cuda_preload_output_weights(sf37_engine *e) {
    if (!e || !cuda_layer_aware_preload_enabled()) return;
    cuda_preload_tensor_range(e, e->final_norm, "output.final_norm");
    cuda_preload_tensor_range(e, e->lm_head, "output.lm_head");
}

static void cuda_prefill_prepare_q8_cache(sf37_engine *e,
                                          sf37_session_progress_fn progress,
                                          void *progress_ud,
                                          int common,
                                          int target_len) {
    if (!e || target_len <= common) return;
    const uint32_t suffix = (uint32_t)(target_len - common);
    if (!cuda_prefill_q8_cache_enabled(suffix)) return;

    if (progress) progress(progress_ud, "prefill_cache", common, target_len);
    const double t0 = sf37_now_sec();
    uint32_t attempts = 0;
    uint64_t bytes = 0;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        cuda_prefill_cache_layer_q8(e, il, &attempts, &bytes);
    }
    const double t1 = sf37_now_sec();
    if (sf37_env_present("SF37_CUDA_PREFILL_Q8_CACHE_VERBOSE",
                         "DS4_CUDA_PREFILL_DETAIL")) {
        fprintf(stderr,
                "sf37: CUDA prefill Q8 cache checked %u tensors (%.2f GiB packed) for %u suffix tokens in %.3fs\n",
                attempts,
                (double)bytes / 1073741824.0,
                suffix,
                t1 - t0);
    }
}

int sf37_engine_cuda_smoke_decode(sf37_engine *e, int token, int layers, int topk, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA decode smoke requires --backend cuda in a CUDA build");
        return -1;
    }
    if (layers < 0 || layers > SF37_MAIN_LAYERS) {
        sf37_log(stderr, SF37_LOG_ERROR, "--smoke-layers must be in [0,%u]", SF37_MAIN_LAYERS);
        return -1;
    }
    if (topk < 0 || topk > SF37_VOCAB) {
        sf37_log(stderr, SF37_LOG_ERROR, "--smoke-topk must be in [0,%u]", SF37_VOCAB);
        return -1;
    }

    int rc = 0;
    float *seed = xmalloc_f32(SF37_EMBD);
    float *cpu = xmalloc_f32(SF37_EMBD);
    float *gpu = xmalloc_f32(SF37_EMBD);
    sf37_cuda_decode_state state;
    sf37_kv_cache cpu_cache;
    memset(&state, 0, sizeof(state));
    memset(&cpu_cache, 0, sizeof(cpu_cache));

    embed_token_bf16(e, token, seed);
    memcpy(cpu, seed, (size_t)SF37_EMBD * sizeof(cpu[0]));
    CUDA_CHECK(cuda_decode_state_init(&state, 1), "decode state allocation");
    CUDA_CHECK(sf37_cuda_tensor_write(state.hidden, 0, seed,
                                      (uint64_t)SF37_EMBD * sizeof(float)),
               "initial hidden upload");
    kv_cache_init(&cpu_cache, 1);

    fprintf(fp, "sf37 cuda smoke token=%d decode layers=%d/%u\n",
            token, layers, SF37_MAIN_LAYERS);
    print_vec_stats(fp, "embed", seed, SF37_EMBD);
    for (int il = 0; il < layers; il++) {
        char cpu_label[64];
        char gpu_label[64];
        char cmp_label[64];
        CUDA_CHECK(cuda_decode_layer(e, &state, (uint32_t)il, 0), "decode layer");
        CUDA_CHECK(sf37_cuda_tensor_read(state.hidden, 0, gpu,
                                         (uint64_t)SF37_EMBD * sizeof(float)),
                   "read hidden");
        run_layer_decode(e, &cpu_cache, (uint32_t)il, 0, cpu);
        snprintf(cpu_label, sizeof(cpu_label), "cpu_after_layer%d%s", il, il >= 3 ? "_moe" : "");
        snprintf(gpu_label, sizeof(gpu_label), "cuda_after_layer%d%s", il, il >= 3 ? "_moe" : "");
        snprintf(cmp_label, sizeof(cmp_label), "cuda_layer%d_vs_cpu", il);
        print_vec_stats(fp, cpu_label, cpu, SF37_EMBD);
        print_vec_stats(fp, gpu_label, gpu, SF37_EMBD);
        compare_vec(fp, cmp_label, gpu, cpu, SF37_EMBD);
    }

    if (topk > 0) {
        if (layers != SF37_MAIN_LAYERS) {
            fprintf(fp, "logits note: partial transformer stack (%d/%u layers); use --smoke-layers %u for a full CUDA smoke.\n",
                    layers, SF37_MAIN_LAYERS, SF37_MAIN_LAYERS);
        }
        float *logits = xmalloc_f32(SF37_VOCAB);
        CUDA_CHECK(cuda_output_logits(e, &state), "cuda output logits");
        CUDA_CHECK(sf37_cuda_tensor_read(state.logits, 0, logits,
                                         (uint64_t)SF37_VOCAB * sizeof(float)),
                   "read cuda logits");
        print_vec_stats(fp, "cuda_logits", logits, SF37_VOCAB);
        print_top_logits(fp, logits, topk);
        free(logits);
    }

cleanup:
    kv_cache_free(&cpu_cache);
    cuda_decode_state_free(&state);
    free(gpu);
    free(cpu);
    free(seed);
    return rc;
}

int sf37_engine_cuda_bench_decode(sf37_engine *e, int token, int layers,
                                  int repeat, int cache_cap,
                                  bool include_logits, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA bench requires --backend cuda in a CUDA build");
        return -1;
    }
    if (layers < 0 || layers > SF37_MAIN_LAYERS) {
        sf37_log(stderr, SF37_LOG_ERROR, "--smoke-layers must be in [0,%u]", SF37_MAIN_LAYERS);
        return -1;
    }
    if (repeat <= 0) repeat = 1;
    if (cache_cap <= 0) cache_cap = 1;
    if (cache_cap > SF37_CTX) cache_cap = SF37_CTX;

    int rc = 0;
    const uint64_t embd_bytes = (uint64_t)SF37_EMBD * sizeof(float);
    float *seed = xmalloc_f32(SF37_EMBD);
    sf37_cuda_decode_state state;
    memset(&state, 0, sizeof(state));

    embed_token_bf16(e, token, seed);
    CUDA_CHECK(cuda_decode_state_init(&state, (uint32_t)cache_cap), "bench decode state allocation");
    CUDA_CHECK(sf37_cuda_tensor_write(state.hidden, 0, seed, embd_bytes),
               "bench initial hidden upload");
    CUDA_CHECK(sf37_cuda_synchronize(), "bench warmup synchronize");

    const double t0 = sf37_now_sec();
    for (int r = 0; r < repeat; r++) {
        CUDA_CHECK(sf37_cuda_tensor_write(state.hidden, 0, seed, embd_bytes),
                   "bench hidden upload");
        for (int il = 0; il < layers; il++) {
            CUDA_CHECK(cuda_decode_layer(e, &state, (uint32_t)il, (uint32_t)r),
                       "bench decode layer");
        }
        if (include_logits) CUDA_CHECK(cuda_output_logits(e, &state), "bench output logits");
    }
    CUDA_CHECK(sf37_cuda_synchronize(), "bench synchronize");
    const double t1 = sf37_now_sec();

    const double elapsed = t1 - t0;
    const double avg_ms = elapsed > 0.0 ? elapsed * 1000.0 / (double)repeat : 0.0;
    const double tok_s = elapsed > 0.0 ? (double)repeat / elapsed : 0.0;
    fprintf(fp,
            "sf37 cuda bench token=%d layers=%d/%u repeat=%d cache_cap=%d logits=%d\n",
            token, layers, SF37_MAIN_LAYERS, repeat, cache_cap, include_logits ? 1 : 0);
    fprintf(fp,
            "sf37 cuda bench elapsed=%.6f s avg=%.3f ms/token throughput=%.3f tok/s\n",
            elapsed, avg_ms, tok_s);

cleanup:
    cuda_decode_state_free(&state);
    free(seed);
    return rc;
}

int sf37_engine_cuda_layer0_smoke(sf37_engine *e, int token, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer0 smoke requires --backend cuda in a CUDA build");
        return -1;
    }

    int rc = 0;
    const uint32_t il = 0;
    sf37_layer_weights *l = &e->layer[il];
    const uint32_t q_heads = layer_q_heads(il);
    const uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    const uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;

    float *seed = xmalloc_f32(SF37_EMBD);
    float *cpu = xmalloc_f32(SF37_EMBD);
    float *gpu = xmalloc_f32(SF37_EMBD);
    float *gpu_attn_hidden = xmalloc_f32(SF37_EMBD);
    float *stage = NULL;
    float *w_embd_f = NULL;
    float *cpu_attn_norm = NULL;
    float *cpu_q = NULL;
    float *cpu_k = NULL;
    float *cpu_v = NULL;
    float *cpu_head_gate = NULL;
    float *cpu_attn_heads = NULL;
    float *cpu_attn_out = NULL;
    float *cpu_attn_hidden = NULL;
    float *cpu_ffn_norm = NULL;
    float *cpu_gate = NULL;
    float *cpu_up = NULL;
    float *cpu_mid = NULL;
    float *cpu_ffn_out = NULL;
    embed_token_bf16(e, token, seed);
    memcpy(cpu, seed, (size_t)SF37_EMBD * sizeof(cpu[0]));

    sf37_cuda_tensor *d_hidden = NULL;
    sf37_cuda_tensor *d_attn_norm = NULL;
    sf37_cuda_tensor *d_q = NULL;
    sf37_cuda_tensor *d_k = NULL;
    sf37_cuda_tensor *d_v = NULL;
    sf37_cuda_tensor *d_head_gate = NULL;
    sf37_cuda_tensor *d_attn_heads = NULL;
    sf37_cuda_tensor *d_attn_out = NULL;
    sf37_cuda_tensor *d_ffn_norm = NULL;
    sf37_cuda_tensor *d_gate = NULL;
    sf37_cuda_tensor *d_up = NULL;
    sf37_cuda_tensor *d_mid = NULL;
    sf37_cuda_tensor *d_ffn_out = NULL;
    sf37_cuda_tensor *w_input_norm = NULL;
    sf37_cuda_tensor *w_post_norm = NULL;
    sf37_cuda_tensor *w_q = NULL;
    sf37_cuda_tensor *w_k = NULL;
    sf37_cuda_tensor *w_v = NULL;
    sf37_cuda_tensor *w_g = NULL;
    sf37_cuda_tensor *w_o = NULL;
    sf37_cuda_tensor *w_gate = NULL;
    sf37_cuda_tensor *w_up = NULL;
    sf37_cuda_tensor *w_down = NULL;

    d_hidden = cuda_tensor_from_host(seed, (uint64_t)SF37_EMBD * sizeof(float), "hidden");
    d_attn_norm = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_q = sf37_cuda_tensor_alloc((uint64_t)q_dim * sizeof(float));
    d_k = sf37_cuda_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    d_v = sf37_cuda_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    d_head_gate = sf37_cuda_tensor_alloc((uint64_t)q_heads * sizeof(float));
    d_attn_heads = sf37_cuda_tensor_alloc((uint64_t)q_dim * sizeof(float));
    d_attn_out = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_ffn_norm = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_gate = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_up = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_mid = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_ffn_out = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    CUDA_CHECK(d_hidden && d_attn_norm && d_q && d_k && d_v && d_head_gate &&
               d_attn_heads && d_attn_out && d_ffn_norm && d_gate && d_up &&
               d_mid && d_ffn_out, "activation allocation");

    w_input_norm = cuda_tensor_from_host(tensor_data(e, l->input_norm), l->input_norm->nbytes, "layer0 input norm");
    w_post_norm = cuda_tensor_from_host(tensor_data(e, l->post_norm), l->post_norm->nbytes, "layer0 post norm");
    w_q = cuda_tensor_from_host(tensor_data(e, l->q_proj), l->q_proj->nbytes, "layer0 q_proj");
    w_k = cuda_tensor_from_host(tensor_data(e, l->k_proj), l->k_proj->nbytes, "layer0 k_proj");
    w_v = cuda_tensor_from_host(tensor_data(e, l->v_proj), l->v_proj->nbytes, "layer0 v_proj");
    w_g = cuda_tensor_from_host(tensor_data(e, l->g_proj), l->g_proj->nbytes, "layer0 g_proj");
    w_o = cuda_tensor_from_host(tensor_data(e, l->o_proj), l->o_proj->nbytes, "layer0 o_proj");
    w_gate = cuda_tensor_from_host(tensor_data(e, l->mlp_gate), l->mlp_gate->nbytes, "layer0 mlp gate");
    w_up = cuda_tensor_from_host(tensor_data(e, l->mlp_up), l->mlp_up->nbytes, "layer0 mlp up");
    w_down = cuda_tensor_from_host(tensor_data(e, l->mlp_down), l->mlp_down->nbytes, "layer0 mlp down");
    CUDA_CHECK(w_input_norm && w_post_norm && w_q && w_k && w_v && w_g &&
               w_o && w_gate && w_up && w_down, "weight upload");

    CUDA_CHECK(sf37_cuda_rms_norm_weight1_bf16(d_attn_norm, d_hidden, w_input_norm,
                                               SF37_EMBD, SF37_RMS_EPS), "input rms");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_q, w_q, SF37_EMBD, q_dim, d_attn_norm), "q_proj");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_k, w_k, SF37_EMBD, kv_dim, d_attn_norm), "k_proj");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_v, w_v, SF37_EMBD, kv_dim, d_attn_norm), "v_proj");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_head_gate, w_g, SF37_EMBD, q_heads, d_attn_norm), "g_proj");
    CUDA_CHECK(sf37_cuda_gqa_single_token_heads(d_attn_heads, d_v, d_head_gate,
                                                q_heads, SF37_KV_HEADS, SF37_HEAD_DIM),
               "single-token attention heads");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_attn_out, w_o, q_dim, SF37_EMBD, d_attn_heads), "o_proj");
    CUDA_CHECK(sf37_cuda_add_inplace_f32(d_hidden, d_attn_out, SF37_EMBD), "attention residual");
    CUDA_CHECK(sf37_cuda_tensor_read(d_hidden, 0, gpu_attn_hidden,
                                     (uint64_t)SF37_EMBD * sizeof(float)),
               "read attention hidden");

    CUDA_CHECK(sf37_cuda_rms_norm_weight1_bf16(d_ffn_norm, d_hidden, w_post_norm,
                                               SF37_EMBD, SF37_RMS_EPS), "post rms");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_gate, w_gate, SF37_EMBD, SF37_DENSE_FF, d_ffn_norm), "dense gate");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_up, w_up, SF37_EMBD, SF37_DENSE_FF, d_ffn_norm), "dense up");
    CUDA_CHECK(sf37_cuda_swiglu_f32(d_mid, d_gate, d_up, SF37_DENSE_FF,
                                    layer_swiglu_limit(il, true)), "dense swiglu");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_ffn_out, w_down, SF37_DENSE_FF, SF37_EMBD, d_mid), "dense down");
    CUDA_CHECK(sf37_cuda_add_inplace_f32(d_hidden, d_ffn_out, SF37_EMBD), "ffn residual");
    CUDA_CHECK(sf37_cuda_tensor_read(d_hidden, 0, gpu, (uint64_t)SF37_EMBD * sizeof(float)), "read hidden");

    sf37_kv_cache cache;
    kv_cache_init(&cache, 1);
    run_layer_decode(e, &cache, il, 0, cpu);
    kv_cache_free(&cache);

    fprintf(fp, "sf37 cuda layer0 smoke token=%d\n", token);
    print_vec_stats(fp, "cpu_after_layer0", cpu, SF37_EMBD);
    print_vec_stats(fp, "cuda_after_layer0", gpu, SF37_EMBD);
    compare_vec(fp, "cuda_layer0_vs_cpu", gpu, cpu, SF37_EMBD);

    uint64_t stage_cap = SF37_DENSE_FF;
    if ((uint64_t)q_dim > stage_cap) stage_cap = q_dim;
    stage = xmalloc_f32(stage_cap);
    w_embd_f = xmalloc_f32(SF37_EMBD);
    cpu_attn_norm = xmalloc_f32(SF37_EMBD);
    cpu_q = xmalloc_f32(q_dim);
    cpu_k = xmalloc_f32(kv_dim);
    cpu_v = xmalloc_f32(kv_dim);
    cpu_head_gate = xmalloc_f32(q_heads);
    cpu_attn_heads = xmalloc_f32(q_dim);
    cpu_attn_out = xmalloc_f32(SF37_EMBD);
    cpu_attn_hidden = xmalloc_f32(SF37_EMBD);
    cpu_ffn_norm = xmalloc_f32(SF37_EMBD);
    cpu_gate = xmalloc_f32(SF37_DENSE_FF);
    cpu_up = xmalloc_f32(SF37_DENSE_FF);
    cpu_mid = xmalloc_f32(SF37_DENSE_FF);
    cpu_ffn_out = xmalloc_f32(SF37_EMBD);

    tensor_bf16_to_f32(e, l->input_norm, w_embd_f, SF37_EMBD);
    sf37_rms_norm_weight1(cpu_attn_norm, seed, w_embd_f, SF37_EMBD, SF37_RMS_EPS);
    sf37_q8_0_matvec(cpu_q, tensor_data(e, l->q_proj), SF37_EMBD, q_dim, cpu_attn_norm);
    sf37_q8_0_matvec(cpu_k, tensor_data(e, l->k_proj), SF37_EMBD, kv_dim, cpu_attn_norm);
    sf37_q8_0_matvec(cpu_v, tensor_data(e, l->v_proj), SF37_EMBD, kv_dim, cpu_attn_norm);
    sf37_q8_0_matvec(cpu_head_gate, tensor_data(e, l->g_proj), SF37_EMBD, q_heads, cpu_attn_norm);
    for (uint32_t h = 0; h < q_heads; h++) {
        const uint32_t kvh = h / (q_heads / SF37_KV_HEADS);
        const float g = sf37_sigmoid(cpu_head_gate[h]);
        for (uint32_t d = 0; d < SF37_HEAD_DIM; d++) {
            cpu_attn_heads[(uint64_t)h * SF37_HEAD_DIM + d] =
                cpu_v[(uint64_t)kvh * SF37_HEAD_DIM + d] * g;
        }
    }
    sf37_q8_0_matvec(cpu_attn_out, tensor_data(e, l->o_proj), q_dim, SF37_EMBD, cpu_attn_heads);
    for (uint32_t i = 0; i < SF37_EMBD; i++) cpu_attn_hidden[i] = seed[i] + cpu_attn_out[i];
    tensor_bf16_to_f32(e, l->post_norm, w_embd_f, SF37_EMBD);
    sf37_rms_norm_weight1(cpu_ffn_norm, cpu_attn_hidden, w_embd_f, SF37_EMBD, SF37_RMS_EPS);
    sf37_q8_0_matvec(cpu_gate, tensor_data(e, l->mlp_gate), SF37_EMBD, SF37_DENSE_FF, cpu_ffn_norm);
    sf37_q8_0_matvec(cpu_up, tensor_data(e, l->mlp_up), SF37_EMBD, SF37_DENSE_FF, cpu_ffn_norm);
    sf37_swiglu(cpu_mid, cpu_gate, cpu_up, SF37_DENSE_FF, layer_swiglu_limit(il, true));
    sf37_q8_0_matvec(cpu_ffn_out, tensor_data(e, l->mlp_down), SF37_DENSE_FF, SF37_EMBD, cpu_mid);

#define READ_COMPARE_STAGE(tensor_, cpu_, n_, label_) do { \
        CUDA_CHECK(sf37_cuda_tensor_read((tensor_), 0, stage, (uint64_t)(n_) * sizeof(float)), \
                   "read " label_); \
        compare_vec(fp, "cuda_" label_ "_vs_cpu", stage, (cpu_), (n_)); \
    } while (0)

    READ_COMPARE_STAGE(d_attn_norm, cpu_attn_norm, SF37_EMBD, "layer0_attn_norm");
    READ_COMPARE_STAGE(d_q, cpu_q, q_dim, "layer0_q_proj");
    READ_COMPARE_STAGE(d_k, cpu_k, kv_dim, "layer0_k_proj");
    READ_COMPARE_STAGE(d_v, cpu_v, kv_dim, "layer0_v_proj");
    READ_COMPARE_STAGE(d_head_gate, cpu_head_gate, q_heads, "layer0_g_proj");
    READ_COMPARE_STAGE(d_attn_heads, cpu_attn_heads, q_dim, "layer0_attn_heads");
    READ_COMPARE_STAGE(d_attn_out, cpu_attn_out, SF37_EMBD, "layer0_o_proj");
    compare_vec(fp, "cuda_layer0_attn_residual_vs_cpu", gpu_attn_hidden,
                cpu_attn_hidden, SF37_EMBD);
    READ_COMPARE_STAGE(d_ffn_norm, cpu_ffn_norm, SF37_EMBD, "layer0_ffn_norm");
    READ_COMPARE_STAGE(d_gate, cpu_gate, SF37_DENSE_FF, "layer0_mlp_gate");
    READ_COMPARE_STAGE(d_up, cpu_up, SF37_DENSE_FF, "layer0_mlp_up");
    READ_COMPARE_STAGE(d_mid, cpu_mid, SF37_DENSE_FF, "layer0_mlp_mid");
    READ_COMPARE_STAGE(d_ffn_out, cpu_ffn_out, SF37_EMBD, "layer0_mlp_down");

#undef READ_COMPARE_STAGE

cleanup:
    sf37_cuda_tensor_free(w_down);
    sf37_cuda_tensor_free(w_up);
    sf37_cuda_tensor_free(w_gate);
    sf37_cuda_tensor_free(w_o);
    sf37_cuda_tensor_free(w_g);
    sf37_cuda_tensor_free(w_v);
    sf37_cuda_tensor_free(w_k);
    sf37_cuda_tensor_free(w_q);
    sf37_cuda_tensor_free(w_post_norm);
    sf37_cuda_tensor_free(w_input_norm);
    sf37_cuda_tensor_free(d_ffn_out);
    sf37_cuda_tensor_free(d_mid);
    sf37_cuda_tensor_free(d_up);
    sf37_cuda_tensor_free(d_gate);
    sf37_cuda_tensor_free(d_ffn_norm);
    sf37_cuda_tensor_free(d_attn_out);
    sf37_cuda_tensor_free(d_attn_heads);
    sf37_cuda_tensor_free(d_head_gate);
    sf37_cuda_tensor_free(d_v);
    sf37_cuda_tensor_free(d_k);
    sf37_cuda_tensor_free(d_q);
    sf37_cuda_tensor_free(d_attn_norm);
    sf37_cuda_tensor_free(d_hidden);
    free(cpu_ffn_out);
    free(cpu_mid);
    free(cpu_up);
    free(cpu_gate);
    free(cpu_ffn_norm);
    free(cpu_attn_hidden);
    free(cpu_attn_out);
    free(cpu_attn_heads);
    free(cpu_head_gate);
    free(cpu_v);
    free(cpu_k);
    free(cpu_q);
    free(cpu_attn_norm);
    free(w_embd_f);
    free(stage);
    free(gpu_attn_hidden);
    free(gpu);
    free(cpu);
    free(seed);
    return rc;
}

int sf37_engine_cuda_layer0_seq_smoke(sf37_engine *e, int token0, int token1, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer0 sequence smoke requires --backend cuda in a CUDA build");
        return -1;
    }

    int rc = 0;
    const uint32_t il = 0;
    sf37_layer_weights *l = &e->layer[il];
    const uint32_t q_heads = layer_q_heads(il);
    const uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    const uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t rotary_dim = layer_rotary_dim(il);
    const double theta = layer_rope_theta(il);
    const int llama3 = layer_is_full(il) ? 1 : 0;
    const int tokens[2] = { token0, token1 };

    float *seed = xmalloc_f32(SF37_EMBD);
    float *cpu = xmalloc_f32((uint64_t)2 * SF37_EMBD);
    float *gpu = xmalloc_f32((uint64_t)2 * SF37_EMBD);

    sf37_cuda_tensor *d_hidden = NULL;
    sf37_cuda_tensor *d_attn_norm = NULL;
    sf37_cuda_tensor *d_q = NULL;
    sf37_cuda_tensor *d_k = NULL;
    sf37_cuda_tensor *d_v = NULL;
    sf37_cuda_tensor *d_head_gate = NULL;
    sf37_cuda_tensor *d_attn_heads = NULL;
    sf37_cuda_tensor *d_attn_out = NULL;
    sf37_cuda_tensor *d_ffn_norm = NULL;
    sf37_cuda_tensor *d_gate = NULL;
    sf37_cuda_tensor *d_up = NULL;
    sf37_cuda_tensor *d_mid = NULL;
    sf37_cuda_tensor *d_ffn_out = NULL;
    sf37_cuda_tensor *d_k_cache = NULL;
    sf37_cuda_tensor *d_v_cache = NULL;
    sf37_cuda_tensor *w_input_norm = NULL;
    sf37_cuda_tensor *w_post_norm = NULL;
    sf37_cuda_tensor *w_q_norm = NULL;
    sf37_cuda_tensor *w_k_norm = NULL;
    sf37_cuda_tensor *w_q = NULL;
    sf37_cuda_tensor *w_k = NULL;
    sf37_cuda_tensor *w_v = NULL;
    sf37_cuda_tensor *w_g = NULL;
    sf37_cuda_tensor *w_o = NULL;
    sf37_cuda_tensor *w_gate = NULL;
    sf37_cuda_tensor *w_up = NULL;
    sf37_cuda_tensor *w_down = NULL;

    d_hidden = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_attn_norm = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_q = sf37_cuda_tensor_alloc((uint64_t)q_dim * sizeof(float));
    d_k = sf37_cuda_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    d_v = sf37_cuda_tensor_alloc((uint64_t)kv_dim * sizeof(float));
    d_head_gate = sf37_cuda_tensor_alloc((uint64_t)q_heads * sizeof(float));
    d_attn_heads = sf37_cuda_tensor_alloc((uint64_t)q_dim * sizeof(float));
    d_attn_out = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_ffn_norm = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_gate = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_up = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_mid = sf37_cuda_tensor_alloc((uint64_t)SF37_DENSE_FF * sizeof(float));
    d_ffn_out = sf37_cuda_tensor_alloc((uint64_t)SF37_EMBD * sizeof(float));
    d_k_cache = sf37_cuda_tensor_alloc((uint64_t)2 * kv_dim * sizeof(float));
    d_v_cache = sf37_cuda_tensor_alloc((uint64_t)2 * kv_dim * sizeof(float));
    CUDA_CHECK(d_hidden && d_attn_norm && d_q && d_k && d_v && d_head_gate &&
               d_attn_heads && d_attn_out && d_ffn_norm && d_gate && d_up &&
               d_mid && d_ffn_out && d_k_cache && d_v_cache, "sequence activation allocation");

    w_input_norm = cuda_tensor_from_host(tensor_data(e, l->input_norm), l->input_norm->nbytes, "layer0 input norm");
    w_post_norm = cuda_tensor_from_host(tensor_data(e, l->post_norm), l->post_norm->nbytes, "layer0 post norm");
    w_q_norm = cuda_tensor_from_host(tensor_data(e, l->q_norm), l->q_norm->nbytes, "layer0 q norm");
    w_k_norm = cuda_tensor_from_host(tensor_data(e, l->k_norm), l->k_norm->nbytes, "layer0 k norm");
    w_q = cuda_tensor_from_host(tensor_data(e, l->q_proj), l->q_proj->nbytes, "layer0 q_proj");
    w_k = cuda_tensor_from_host(tensor_data(e, l->k_proj), l->k_proj->nbytes, "layer0 k_proj");
    w_v = cuda_tensor_from_host(tensor_data(e, l->v_proj), l->v_proj->nbytes, "layer0 v_proj");
    w_g = cuda_tensor_from_host(tensor_data(e, l->g_proj), l->g_proj->nbytes, "layer0 g_proj");
    w_o = cuda_tensor_from_host(tensor_data(e, l->o_proj), l->o_proj->nbytes, "layer0 o_proj");
    w_gate = cuda_tensor_from_host(tensor_data(e, l->mlp_gate), l->mlp_gate->nbytes, "layer0 mlp gate");
    w_up = cuda_tensor_from_host(tensor_data(e, l->mlp_up), l->mlp_up->nbytes, "layer0 mlp up");
    w_down = cuda_tensor_from_host(tensor_data(e, l->mlp_down), l->mlp_down->nbytes, "layer0 mlp down");
    CUDA_CHECK(w_input_norm && w_post_norm && w_q_norm && w_k_norm && w_q && w_k &&
               w_v && w_g && w_o && w_gate && w_up && w_down, "sequence weight upload");

    for (uint32_t t = 0; t < 2; t++) {
        embed_token_bf16(e, tokens[t], seed);
        CUDA_CHECK(sf37_cuda_tensor_write(d_hidden, 0, seed,
                                          (uint64_t)SF37_EMBD * sizeof(float)),
                   "sequence hidden upload");
        CUDA_CHECK(sf37_cuda_rms_norm_weight1_bf16(d_attn_norm, d_hidden, w_input_norm,
                                                   SF37_EMBD, SF37_RMS_EPS), "sequence input rms");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_q, w_q, SF37_EMBD, q_dim, d_attn_norm), "sequence q_proj");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_k, w_k, SF37_EMBD, kv_dim, d_attn_norm), "sequence k_proj");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_v, w_v, SF37_EMBD, kv_dim, d_attn_norm), "sequence v_proj");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_head_gate, w_g, SF37_EMBD, q_heads, d_attn_norm), "sequence g_proj");
        CUDA_CHECK(sf37_cuda_head_rms_norm_weight1_bf16(d_q, w_q_norm, q_heads,
                                                        SF37_HEAD_DIM, SF37_RMS_EPS),
                   "sequence q head norm");
        CUDA_CHECK(sf37_cuda_head_rms_norm_weight1_bf16(d_k, w_k_norm, SF37_KV_HEADS,
                                                        SF37_HEAD_DIM, SF37_RMS_EPS),
                   "sequence k head norm");
        CUDA_CHECK(sf37_cuda_rope_split_half(d_q, q_heads, SF37_HEAD_DIM,
                                             rotary_dim, theta, llama3, t),
                   "sequence q rope");
        CUDA_CHECK(sf37_cuda_rope_split_half(d_k, SF37_KV_HEADS, SF37_HEAD_DIM,
                                             rotary_dim, theta, llama3, t),
                   "sequence k rope");
        CUDA_CHECK(sf37_cuda_tensor_copy(d_k_cache, (uint64_t)t * kv_dim * sizeof(float),
                                         d_k, 0, (uint64_t)kv_dim * sizeof(float)),
                   "sequence k cache store");
        CUDA_CHECK(sf37_cuda_tensor_copy(d_v_cache, (uint64_t)t * kv_dim * sizeof(float),
                                         d_v, 0, (uint64_t)kv_dim * sizeof(float)),
                   "sequence v cache store");
        CUDA_CHECK(sf37_cuda_attention_decode_heads(d_attn_heads, d_q, d_k_cache, d_v_cache,
                                                    d_head_gate, t + 1u, 2u, q_heads,
                                                    SF37_KV_HEADS, SF37_HEAD_DIM,
                                                    layer_is_full(il) ? 0 : 1,
                                                    SF37_SLIDING_WINDOW),
                   "sequence attention decode");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_attn_out, w_o, q_dim, SF37_EMBD, d_attn_heads),
                   "sequence o_proj");
        CUDA_CHECK(sf37_cuda_add_inplace_f32(d_hidden, d_attn_out, SF37_EMBD),
                   "sequence attention residual");
        CUDA_CHECK(sf37_cuda_rms_norm_weight1_bf16(d_ffn_norm, d_hidden, w_post_norm,
                                                   SF37_EMBD, SF37_RMS_EPS), "sequence post rms");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_gate, w_gate, SF37_EMBD, SF37_DENSE_FF, d_ffn_norm),
                   "sequence dense gate");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_up, w_up, SF37_EMBD, SF37_DENSE_FF, d_ffn_norm),
                   "sequence dense up");
        CUDA_CHECK(sf37_cuda_swiglu_f32(d_mid, d_gate, d_up, SF37_DENSE_FF,
                                        layer_swiglu_limit(il, true)),
                   "sequence dense swiglu");
        CUDA_CHECK(sf37_cuda_matvec_q8_0(d_ffn_out, w_down, SF37_DENSE_FF, SF37_EMBD, d_mid),
                   "sequence dense down");
        CUDA_CHECK(sf37_cuda_add_inplace_f32(d_hidden, d_ffn_out, SF37_EMBD),
                   "sequence ffn residual");
        CUDA_CHECK(sf37_cuda_tensor_read(d_hidden, 0,
                                         gpu + (uint64_t)t * SF37_EMBD,
                                         (uint64_t)SF37_EMBD * sizeof(float)),
                   "sequence read hidden");
    }

    sf37_kv_cache cache;
    kv_cache_init(&cache, 2);
    for (uint32_t t = 0; t < 2; t++) {
        embed_token_bf16(e, tokens[t], cpu + (uint64_t)t * SF37_EMBD);
        run_layer_decode(e, &cache, il, t, cpu + (uint64_t)t * SF37_EMBD);
    }
    kv_cache_free(&cache);

    fprintf(fp, "sf37 cuda layer0 seq smoke tokens=%d,%d\n", token0, token1);
    print_vec_stats(fp, "cpu_after_layer0_token0", cpu, SF37_EMBD);
    print_vec_stats(fp, "cuda_after_layer0_token0", gpu, SF37_EMBD);
    compare_vec(fp, "cuda_layer0_token0_vs_cpu", gpu, cpu, SF37_EMBD);
    print_vec_stats(fp, "cpu_after_layer0_token1", cpu + SF37_EMBD, SF37_EMBD);
    print_vec_stats(fp, "cuda_after_layer0_token1", gpu + SF37_EMBD, SF37_EMBD);
    compare_vec(fp, "cuda_layer0_token1_vs_cpu", gpu + SF37_EMBD, cpu + SF37_EMBD, SF37_EMBD);

cleanup:
    sf37_cuda_tensor_free(w_down);
    sf37_cuda_tensor_free(w_up);
    sf37_cuda_tensor_free(w_gate);
    sf37_cuda_tensor_free(w_o);
    sf37_cuda_tensor_free(w_g);
    sf37_cuda_tensor_free(w_v);
    sf37_cuda_tensor_free(w_k);
    sf37_cuda_tensor_free(w_q);
    sf37_cuda_tensor_free(w_k_norm);
    sf37_cuda_tensor_free(w_q_norm);
    sf37_cuda_tensor_free(w_post_norm);
    sf37_cuda_tensor_free(w_input_norm);
    sf37_cuda_tensor_free(d_v_cache);
    sf37_cuda_tensor_free(d_k_cache);
    sf37_cuda_tensor_free(d_ffn_out);
    sf37_cuda_tensor_free(d_mid);
    sf37_cuda_tensor_free(d_up);
    sf37_cuda_tensor_free(d_gate);
    sf37_cuda_tensor_free(d_ffn_norm);
    sf37_cuda_tensor_free(d_attn_out);
    sf37_cuda_tensor_free(d_attn_heads);
    sf37_cuda_tensor_free(d_head_gate);
    sf37_cuda_tensor_free(d_v);
    sf37_cuda_tensor_free(d_k);
    sf37_cuda_tensor_free(d_q);
    sf37_cuda_tensor_free(d_attn_norm);
    sf37_cuda_tensor_free(d_hidden);
    free(gpu);
    free(cpu);
    free(seed);
    return rc;
}

int sf37_engine_cuda_layer3_moe_smoke(sf37_engine *e, int token, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer3 MoE smoke requires --backend cuda in a CUDA build");
        return -1;
    }

    const uint32_t il = 3;
    sf37_layer_weights *l = &e->layer[il];
    int rc = 0;
    float *seed = xmalloc_f32(SF37_EMBD);
    float *cpu = xmalloc_f32(SF37_EMBD);
    float *gpu = xmalloc_f32(SF37_EMBD);
    int32_t selected[SF37_EXPERT_USED];
    float weights[SF37_EXPERT_USED];
    memset(selected, 0, sizeof(selected));
    memset(weights, 0, sizeof(weights));

    sf37_cuda_tensor *d_x = NULL;
    sf37_cuda_tensor *d_router_logits = NULL;
    sf37_cuda_tensor *d_router_probs = NULL;
    sf37_cuda_tensor *d_selected = NULL;
    sf37_cuda_tensor *d_weights = NULL;
    sf37_cuda_tensor *d_shared_gate = NULL;
    sf37_cuda_tensor *d_shared_up = NULL;
    sf37_cuda_tensor *d_shared_mid = NULL;
    sf37_cuda_tensor *d_shared_out = NULL;
    sf37_cuda_tensor *d_routed_gate = NULL;
    sf37_cuda_tensor *d_routed_up = NULL;
    sf37_cuda_tensor *d_routed_mid = NULL;
    sf37_cuda_tensor *d_routed_down = NULL;
    sf37_cuda_tensor *d_routed_out = NULL;
    sf37_cuda_tensor *w_router = NULL;
    sf37_cuda_tensor *w_router_bias = NULL;
    sf37_cuda_tensor *w_share_gate = NULL;
    sf37_cuda_tensor *w_share_up = NULL;
    sf37_cuda_tensor *w_share_down = NULL;
    sf37_cuda_tensor *w_moe_gate = NULL;
    sf37_cuda_tensor *w_moe_up = NULL;
    sf37_cuda_tensor *w_moe_down = NULL;

    embed_token_bf16(e, token, seed);
    layer_moe_ffn(cpu, e, l, seed, il);

    const uint64_t embd_bytes = (uint64_t)SF37_EMBD * sizeof(float);
    const uint64_t expert_bytes = (uint64_t)SF37_EXPERT_FF * sizeof(float);
    const uint64_t pair_mid_bytes = (uint64_t)SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float);
    const uint64_t pair_down_bytes = (uint64_t)SF37_EXPERT_USED * SF37_EMBD * sizeof(float);

    d_x = cuda_tensor_from_host(seed, embd_bytes, "layer3 moe input");
    d_router_logits = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERTS * sizeof(float));
    d_router_probs = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERTS * sizeof(float));
    d_selected = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * sizeof(int32_t));
    d_weights = sf37_cuda_tensor_alloc((uint64_t)SF37_EXPERT_USED * sizeof(float));
    d_shared_gate = sf37_cuda_tensor_alloc(expert_bytes);
    d_shared_up = sf37_cuda_tensor_alloc(expert_bytes);
    d_shared_mid = sf37_cuda_tensor_alloc(expert_bytes);
    d_shared_out = sf37_cuda_tensor_alloc(embd_bytes);
    d_routed_gate = sf37_cuda_tensor_alloc(pair_mid_bytes);
    d_routed_up = sf37_cuda_tensor_alloc(pair_mid_bytes);
    d_routed_mid = sf37_cuda_tensor_alloc(pair_mid_bytes);
    d_routed_down = sf37_cuda_tensor_alloc(pair_down_bytes);
    d_routed_out = sf37_cuda_tensor_alloc(embd_bytes);
    CUDA_CHECK(d_x && d_router_logits && d_router_probs && d_selected && d_weights &&
               d_shared_gate && d_shared_up && d_shared_mid && d_shared_out &&
               d_routed_gate && d_routed_up && d_routed_mid && d_routed_down &&
               d_routed_out, "layer3 MoE scratch allocation");

    w_router = cuda_tensor_from_host(tensor_data(e, l->router_gate), l->router_gate->nbytes, "layer3 router gate");
    w_router_bias = cuda_tensor_from_host(tensor_data(e, l->router_bias), l->router_bias->nbytes, "layer3 router bias");
    w_share_gate = cuda_tensor_from_host(tensor_data(e, l->share_gate), l->share_gate->nbytes, "layer3 shared gate");
    w_share_up = cuda_tensor_from_host(tensor_data(e, l->share_up), l->share_up->nbytes, "layer3 shared up");
    w_share_down = cuda_tensor_from_host(tensor_data(e, l->share_down), l->share_down->nbytes, "layer3 shared down");
    w_moe_gate = cuda_tensor_from_host(tensor_data(e, l->moe_gate), l->moe_gate->nbytes, "layer3 routed gate");
    w_moe_up = cuda_tensor_from_host(tensor_data(e, l->moe_up), l->moe_up->nbytes, "layer3 routed up");
    w_moe_down = cuda_tensor_from_host(tensor_data(e, l->moe_down), l->moe_down->nbytes, "layer3 routed down");
    CUDA_CHECK(w_router && w_router_bias && w_share_gate && w_share_up && w_share_down &&
               w_moe_gate && w_moe_up && w_moe_down, "layer3 MoE weight upload");

    CUDA_CHECK(cuda_matvec_dense_uploaded(d_router_logits, l->router_gate, w_router, d_x),
               "layer3 router gate");
    CUDA_CHECK(sf37_cuda_router_select(d_selected, d_weights, d_router_probs,
                                       d_router_logits, w_router_bias,
                                       SF37_EXPERTS, SF37_EXPERT_USED, 3.0f),
               "layer3 router select");

    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_shared_gate, w_share_gate, SF37_EMBD, SF37_EXPERT_FF, d_x),
               "layer3 shared gate");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_shared_up, w_share_up, SF37_EMBD, SF37_EXPERT_FF, d_x),
               "layer3 shared up");
    CUDA_CHECK(sf37_cuda_swiglu_f32(d_shared_mid, d_shared_gate, d_shared_up,
                                    SF37_EXPERT_FF, layer_swiglu_limit(il, true)),
               "layer3 shared swiglu");
    CUDA_CHECK(sf37_cuda_matvec_q8_0(d_shared_out, w_share_down, SF37_EXPERT_FF, SF37_EMBD, d_shared_mid),
               "layer3 shared down");

    CUDA_CHECK(sf37_cuda_routed_moe_one(d_routed_out, d_routed_gate, d_routed_up,
                                        d_routed_mid, d_routed_down,
                                        w_moe_gate, w_moe_up, w_moe_down,
                                        d_selected, d_weights,
                                        SF37_EXPERTS, SF37_EXPERT_USED,
                                        SF37_EMBD, SF37_EXPERT_FF, SF37_EMBD,
                                        layer_swiglu_limit(il, false), d_x),
               "layer3 routed MoE");
    CUDA_CHECK(sf37_cuda_add_inplace_f32(d_shared_out, d_routed_out, SF37_EMBD),
               "layer3 shared+routed add");
    CUDA_CHECK(sf37_cuda_tensor_read(d_shared_out, 0, gpu, embd_bytes),
               "layer3 MoE read output");
    CUDA_CHECK(sf37_cuda_tensor_read(d_selected, 0, selected, sizeof(selected)),
               "layer3 MoE read selected");
    CUDA_CHECK(sf37_cuda_tensor_read(d_weights, 0, weights, sizeof(weights)),
               "layer3 MoE read weights");

    fprintf(fp, "sf37 cuda layer3 isolated MoE smoke token=%d input=embedding router_gate_type=%s\n",
            token, type_name(l->router_gate->type));
    fprintf(fp, "router selected:");
    for (uint32_t i = 0; i < SF37_EXPERT_USED; i++) {
        fprintf(fp, " %d(%.6g)", selected[i], weights[i]);
    }
    fprintf(fp, "\n");
    print_vec_stats(fp, "cpu_layer3_moe", cpu, SF37_EMBD);
    print_vec_stats(fp, "cuda_layer3_moe", gpu, SF37_EMBD);
    compare_vec(fp, "cuda_layer3_moe_vs_cpu", gpu, cpu, SF37_EMBD);

cleanup:
    sf37_cuda_tensor_free(w_moe_down);
    sf37_cuda_tensor_free(w_moe_up);
    sf37_cuda_tensor_free(w_moe_gate);
    sf37_cuda_tensor_free(w_share_down);
    sf37_cuda_tensor_free(w_share_up);
    sf37_cuda_tensor_free(w_share_gate);
    sf37_cuda_tensor_free(w_router_bias);
    sf37_cuda_tensor_free(w_router);
    sf37_cuda_tensor_free(d_routed_out);
    sf37_cuda_tensor_free(d_routed_down);
    sf37_cuda_tensor_free(d_routed_mid);
    sf37_cuda_tensor_free(d_routed_up);
    sf37_cuda_tensor_free(d_routed_gate);
    sf37_cuda_tensor_free(d_shared_out);
    sf37_cuda_tensor_free(d_shared_mid);
    sf37_cuda_tensor_free(d_shared_up);
    sf37_cuda_tensor_free(d_shared_gate);
    sf37_cuda_tensor_free(d_weights);
    sf37_cuda_tensor_free(d_selected);
    sf37_cuda_tensor_free(d_router_probs);
    sf37_cuda_tensor_free(d_router_logits);
    sf37_cuda_tensor_free(d_x);
    free(gpu);
    free(cpu);
    free(seed);
    return rc;
}

int sf37_engine_cuda_layer_replay_smoke(sf37_engine *e, int token, int layer, FILE *fp) {
    if (!e) return -1;
    if (!fp) fp = stdout;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer replay smoke requires --backend cuda in a CUDA build");
        return -1;
    }
    if (layer < 0 || layer >= (int)SF37_MAIN_LAYERS) {
        sf37_log(stderr, SF37_LOG_ERROR, "--cuda-layer-replay-smoke layer must be in [0,%u]",
                 SF37_MAIN_LAYERS - 1u);
        return -1;
    }

    int rc = 0;
    float *prefix = xmalloc_f32(SF37_EMBD);
    float *cpu = xmalloc_f32(SF37_EMBD);
    float *gpu = xmalloc_f32(SF37_EMBD);
    sf37_kv_cache cpu_prefix_cache;
    sf37_kv_cache cpu_layer_cache;
    sf37_cuda_decode_state state;
    memset(&cpu_prefix_cache, 0, sizeof(cpu_prefix_cache));
    memset(&cpu_layer_cache, 0, sizeof(cpu_layer_cache));
    memset(&state, 0, sizeof(state));

    embed_token_bf16(e, token, prefix);
    kv_cache_init(&cpu_prefix_cache, 1);
    for (int il = 0; il < layer; il++) {
        run_layer_decode(e, &cpu_prefix_cache, (uint32_t)il, 0, prefix);
    }
    kv_cache_free(&cpu_prefix_cache);

    memcpy(cpu, prefix, (size_t)SF37_EMBD * sizeof(cpu[0]));
    kv_cache_init(&cpu_layer_cache, 1);
    run_layer_decode(e, &cpu_layer_cache, (uint32_t)layer, 0, cpu);

    CUDA_CHECK(cuda_decode_state_init(&state, 1), "layer replay state allocation");
    CUDA_CHECK(sf37_cuda_tensor_write(state.hidden, 0, prefix,
                                      (uint64_t)SF37_EMBD * sizeof(float)),
               "layer replay hidden upload");
    CUDA_CHECK(cuda_decode_layer(e, &state, (uint32_t)layer, 0), "layer replay decode");
    CUDA_CHECK(sf37_cuda_tensor_read(state.hidden, 0, gpu,
                                     (uint64_t)SF37_EMBD * sizeof(float)),
               "layer replay hidden read");

    fprintf(fp, "sf37 cuda layer replay smoke token=%d layer=%d input=cpu_prefix\n",
            token, layer);
    print_vec_stats(fp, "cpu_prefix_hidden", prefix, SF37_EMBD);
    print_vec_stats(fp, "cpu_after_replay_layer", cpu, SF37_EMBD);
    print_vec_stats(fp, "cuda_after_replay_layer", gpu, SF37_EMBD);
    compare_vec(fp, "cuda_replay_layer_vs_cpu", gpu, cpu, SF37_EMBD);

cleanup:
    cuda_decode_state_free(&state);
    kv_cache_free(&cpu_layer_cache);
    kv_cache_free(&cpu_prefix_cache);
    free(gpu);
    free(cpu);
    free(prefix);
    return rc;
}

#undef CUDA_CHECK
#else
int sf37_engine_cuda_smoke_decode(sf37_engine *e, int token, int layers, int topk, FILE *fp) {
    (void)e;
    (void)token;
    (void)layers;
    (void)topk;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA decode smoke requested from a CPU-only build");
    return -1;
}

int sf37_engine_cuda_layer0_smoke(sf37_engine *e, int token, FILE *fp) {
    (void)e;
    (void)token;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer0 smoke requested from a CPU-only build");
    return -1;
}

int sf37_engine_cuda_layer0_seq_smoke(sf37_engine *e, int token0, int token1, FILE *fp) {
    (void)e;
    (void)token0;
    (void)token1;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer0 sequence smoke requested from a CPU-only build");
    return -1;
}

int sf37_engine_cuda_layer3_moe_smoke(sf37_engine *e, int token, FILE *fp) {
    (void)e;
    (void)token;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer3 MoE smoke requested from a CPU-only build");
    return -1;
}

int sf37_engine_cuda_layer_replay_smoke(sf37_engine *e, int token, int layer, FILE *fp) {
    (void)e;
    (void)token;
    (void)layer;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA layer replay smoke requested from a CPU-only build");
    return -1;
}

int sf37_engine_cuda_bench_decode(sf37_engine *e, int token, int layers,
                                  int repeat, int cache_cap,
                                  bool include_logits, FILE *fp) {
    (void)e;
    (void)token;
    (void)layers;
    (void)repeat;
    (void)cache_cap;
    (void)include_logits;
    (void)fp;
    sf37_log(stderr, SF37_LOG_ERROR, "CUDA bench requested from a CPU-only build");
    return -1;
}
#endif

#ifdef SF37_USE_CUDA
typedef struct sf37_imatrix_collector sf37_imatrix_collector;
#endif

struct sf37_session {
    sf37_engine *engine;
    int ctx_size;
    sf37_tokens checkpoint;
    bool checkpoint_valid;
    float *hidden;
    float *logits;
    sf37_kv_cache cpu_cache;
    sf37_session_progress_fn progress;
    void *progress_ud;
    sf37_session_cancel_fn cancel;
    void *cancel_ud;
#ifdef SF37_USE_CUDA
    sf37_cuda_decode_state cuda_state;
    sf37_cuda_prefill_state cuda_prefill;
    sf37_imatrix_collector *imatrix_collector;
    bool cuda_state_ready;
    bool cuda_prefill_ready;
#endif
};

static void session_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static bool session_uses_cuda(const sf37_session *s) {
#ifdef SF37_USE_CUDA
    return s && s->engine && s->engine->backend == SF37_BACKEND_CUDA && s->engine->cuda_ready;
#else
    (void)s;
    return false;
#endif
}

static void session_reset(sf37_session *s);

static int sample_argmax(const float *logits, uint32_t n_vocab) {
    if (!logits || n_vocab == 0) return -1;
    int best = 0;
    float best_v = logits[0];
    for (uint32_t i = 1; i < n_vocab; i++) {
        if (logits[i] > best_v) {
            best_v = logits[i];
            best = (int)i;
        }
    }
    return best;
}

static uint64_t sample_rng_next(uint64_t *state) {
    uint64_t x = state && *state ? *state : 0x9e3779b97f4a7c15ull;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    if (state) *state = x;
    return x * 2685821657736338717ull;
}

static float sample_rng_f32(uint64_t *state) {
    const uint64_t x = sample_rng_next(state);
    return (float)((x >> 40) & 0xffffffu) / 16777216.0f;
}

typedef struct {
    int id;
    float logit;
    float prob;
} sf37_sample_candidate;

static int sample_candidate_cmp_desc(const void *a, const void *b) {
    const sf37_sample_candidate *ca = a;
    const sf37_sample_candidate *cb = b;
    return (cb->logit > ca->logit) - (cb->logit < ca->logit);
}

static int sample_full_vocab(const float *logits,
                             uint32_t n_vocab,
                             float temperature,
                             float top_p,
                             float min_p,
                             uint64_t *rng) {
    float max_logit = -INFINITY;
    int best = 0;
    uint32_t finite = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        finite++;
        if (v > max_logit) {
            max_logit = v;
            best = (int)i;
        }
    }
    if (finite == 0) return sample_argmax(logits, n_vocab);

    if (top_p >= 1.0f) {
        float sum = 0.0f;
        const float min_rel = min_p > 0.0f ? min_p : 0.0f;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            sum += p;
        }
        if (sum <= 0.0f || !isfinite(sum)) return best;
        float r = sample_rng_f32(rng) * sum;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const float v = logits[i];
            if (!isfinite(v)) continue;
            const float p = expf((v - max_logit) / temperature);
            if (p < min_rel) continue;
            r -= p;
            if (r <= 0.0f) return (int)i;
        }
        return best;
    }

    sf37_sample_candidate *cand = xmalloc((size_t)finite * sizeof(cand[0]));
    uint32_t n = 0;
    float sum = 0.0f;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        const float p = expf((v - max_logit) / temperature);
        cand[n++] = (sf37_sample_candidate){ .id = (int)i, .logit = v, .prob = p };
        sum += p;
    }
    if (sum <= 0.0f || !isfinite(sum)) {
        free(cand);
        return best;
    }

    qsort(cand, n, sizeof(cand[0]), sample_candidate_cmp_desc);
    const float min_prob = (cand[0].prob / sum) * (min_p > 0.0f ? min_p : 0.0f);
    float filtered_sum = 0.0f;
    uint32_t filtered = 0;
    for (uint32_t i = 0; i < n; i++) {
        const float p = cand[i].prob / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += cand[i].prob;
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered == 0) {
        free(cand);
        return best;
    }

    float r = sample_rng_f32(rng) * filtered_sum;
    for (uint32_t i = 0; i < filtered; i++) {
        r -= cand[i].prob;
        if (r <= 0.0f) {
            const int id = cand[i].id;
            free(cand);
            return id;
        }
    }
    const int id = cand[filtered - 1].id;
    free(cand);
    return id;
}

static int sample_top_p_min_p(const float *logits,
                              uint32_t n_vocab,
                              float temperature,
                              int top_k,
                              float top_p,
                              float min_p,
                              uint64_t *rng) {
    if (temperature <= 0.0f) return sample_argmax(logits, n_vocab);
    if (top_p <= 0.0f || top_p > 1.0f) top_p = 1.0f;
    if (min_p < 0.0f) min_p = 0.0f;
    if (top_k <= 0) return sample_full_vocab(logits, n_vocab, temperature, top_p, min_p, rng);
    if (top_k > 1024) top_k = 1024;
    if ((uint32_t)top_k > n_vocab) top_k = (int)n_vocab;

    int ids[1024];
    float vals[1024];
    int n = 0;
    for (uint32_t i = 0; i < n_vocab; i++) {
        const float v = logits[i];
        if (!isfinite(v)) continue;
        if (n == top_k && v <= vals[n - 1]) continue;
        int j = n < top_k ? n++ : n - 1;
        while (j > 0 && vals[j - 1] < v) {
            vals[j] = vals[j - 1];
            ids[j] = ids[j - 1];
            j--;
        }
        vals[j] = v;
        ids[j] = (int)i;
    }
    if (n == 0) return sample_argmax(logits, n_vocab);

    float probs[1024];
    const float max_logit = vals[0];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        probs[i] = expf((vals[i] - max_logit) / temperature);
        sum += probs[i];
    }
    if (sum <= 0.0f || !isfinite(sum)) return ids[0];

    const float min_prob = (probs[0] / sum) * min_p;
    float filtered_sum = 0.0f;
    int filtered = 0;
    for (int i = 0; i < n; i++) {
        const float p = probs[i] / sum;
        if (i > 0 && p < min_prob) break;
        filtered_sum += probs[i];
        filtered++;
        if (filtered_sum / sum >= top_p) break;
    }
    if (filtered <= 0) return ids[0];

    float r = sample_rng_f32(rng) * filtered_sum;
    for (int i = 0; i < filtered; i++) {
        r -= probs[i];
        if (r <= 0.0f) return ids[i];
    }
    return ids[filtered - 1];
}

int sf37_session_create(sf37_session **out, sf37_engine *e, int ctx_size) {
    if (!out || !e || ctx_size <= 0 || ctx_size > SF37_CTX) return 1;
    *out = NULL;
    sf37_session *s = xcalloc(1, sizeof(*s));
    s->engine = e;
    s->ctx_size = ctx_size;
    s->hidden = xmalloc_f32(SF37_EMBD);
    s->logits = xmalloc_f32(SF37_VOCAB);
    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        if (!cuda_decode_state_init(&s->cuda_state, (uint32_t)ctx_size)) {
            free(s->logits);
            free(s->hidden);
            free(s);
            return 1;
        }
        s->cuda_state_ready = true;
#endif
    } else {
        kv_cache_init(&s->cpu_cache, (uint32_t)ctx_size);
    }
    s->checkpoint_valid = true;
    *out = s;
    return 0;
}

void sf37_session_free(sf37_session *s) {
    if (!s) return;
#ifdef SF37_USE_CUDA
    if (s->cuda_prefill_ready) cuda_prefill_state_free(&s->cuda_prefill);
    if (s->cuda_state_ready) cuda_decode_state_free(&s->cuda_state);
#endif
    kv_cache_free(&s->cpu_cache);
    sf37_tokens_free(&s->checkpoint);
    free(s->logits);
    free(s->hidden);
    free(s);
}

void sf37_session_set_progress(sf37_session *s, sf37_session_progress_fn fn, void *ud) {
    if (!s) return;
    s->progress = fn;
    s->progress_ud = ud;
}

void sf37_session_set_cancel(sf37_session *s, sf37_session_cancel_fn fn, void *ud) {
    if (!s) return;
    s->cancel = fn;
    s->cancel_ud = ud;
}

static bool sf37_session_cancel_requested(sf37_session *s) {
    return s && s->cancel && s->cancel(s->cancel_ud);
}

void sf37_session_report_progress(sf37_session *s, const char *event, int current, int total) {
    if (!s || !s->progress) return;
    s->progress(s->progress_ud, event, current, total);
}

static uint32_t prompt_count_im_patch(sf37_engine *e,
                                      const sf37_tokens *prompt,
                                      uint32_t start,
                                      uint32_t end) {
    if (!e || !prompt || e->vocab.im_patch_id < 0 || start >= end) return 0;
    if (end > (uint32_t)prompt->len) end = (uint32_t)prompt->len;
    uint32_t n = 0;
    for (uint32_t i = start; i < end; i++) {
        if (prompt->v[i] == e->vocab.im_patch_id) n++;
    }
    return n;
}

static bool image_pixels_feature_rows(const sf37_image_features *features,
                                      uint32_t *rows_per_image,
                                      char *err,
                                      size_t errlen) {
    if (rows_per_image) *rows_per_image = 0;
    if (!features || !features->pixel_values || features->images == 0) return true;
    if (features->pixel_channels != 3u) {
        session_set_err(err, errlen, "raw image pixel_values must be NCHW with 3 channels");
        return false;
    }
    if (features->pixel_height != features->pixel_width ||
        (features->pixel_height != SF37_VISION_IMAGE &&
         features->pixel_height != SF37_VISION_PATCH_IMAGE)) {
        session_set_err(err, errlen,
                        "raw CUDA vision currently supports normalized %ux%u or %ux%u square images",
                        SF37_VISION_IMAGE, SF37_VISION_IMAGE,
                        SF37_VISION_PATCH_IMAGE, SF37_VISION_PATCH_IMAGE);
        return false;
    }
    if (features->pixel_height % SF37_VISION_PATCH != 0) {
        session_set_err(err, errlen, "raw image size must be divisible by vision patch size %u",
                        SF37_VISION_PATCH);
        return false;
    }
    const uint32_t grid = features->pixel_height / SF37_VISION_PATCH;
    if ((grid & 3u) != 0) {
        session_set_err(err, errlen, "raw image patch grid %u is not compatible with two stride-2 downsamplers",
                        grid);
        return false;
    }
    const uint32_t out_grid = grid / 4u;
    if (rows_per_image) *rows_per_image = out_grid * out_grid;
    return true;
}

static bool image_pixels_mixed_feature_rows(const sf37_image_features *features,
                                            uint32_t *rows,
                                            char *err,
                                            size_t errlen) {
    if (rows) *rows = 0;
    if (!features) return true;
    const bool has_main = features->pixel_values && features->images > 0;
    const bool has_patch = features->patch_pixel_values && features->patch_images > 0;
    if (!has_patch) {
        uint32_t per = 0;
        if (!image_pixels_feature_rows(features, &per, err, errlen)) return false;
        if (has_main && features->images > UINT32_MAX / per) {
            session_set_err(err, errlen, "raw image feature row count overflow");
            return false;
        }
        if (rows) *rows = has_main ? features->images * per : 0u;
        return true;
    }
    if (!has_main) {
        session_set_err(err, errlen, "mixed image pixels require 728x728 main images");
        return false;
    }
    if (features->pixel_channels != 3u ||
        features->pixel_height != SF37_VISION_IMAGE ||
        features->pixel_width != SF37_VISION_IMAGE) {
        session_set_err(err, errlen,
                        "mixed image pixels require main NCHW 3x%ux%u images",
                        SF37_VISION_IMAGE, SF37_VISION_IMAGE);
        return false;
    }
    if (!features->patches_per_image) {
        session_set_err(err, errlen,
                        "mixed image pixels require patches_per_image metadata");
        return false;
    }
    uint32_t patches = 0;
    for (uint32_t i = 0; i < features->images; i++) {
        if (features->patches_per_image[i] > UINT32_MAX - patches) {
            session_set_err(err, errlen, "patch image count overflow");
            return false;
        }
        patches += features->patches_per_image[i];
    }
    if (patches != features->patch_images) {
        session_set_err(err, errlen,
                        "patches_per_image sum %u does not match patch_images %u",
                        patches, features->patch_images);
        return false;
    }
    const uint64_t total = (uint64_t)features->images * SF37_VISION_FEATURES +
                           (uint64_t)features->patch_images * SF37_VISION_PATCH_FEATURES;
    if (total > UINT32_MAX) {
        session_set_err(err, errlen, "mixed image feature row count overflow");
        return false;
    }
    if (rows) *rows = (uint32_t)total;
    return true;
}

static bool image_features_effective_rows(const sf37_image_features *features,
                                          uint32_t *rows,
                                          char *err,
                                          size_t errlen) {
    if (rows) *rows = 0;
    if (!features) return true;
    if ((features->pixel_values && features->images > 0) ||
        (features->patch_pixel_values && features->patch_images > 0)) {
        if (!image_pixels_mixed_feature_rows(features, rows, err, errlen)) return false;
        return true;
    }
    if (rows) *rows = features->rows;
    return true;
}

#ifdef SF37_USE_CUDA
static bool cuda_batch_prefill_disabled(void) {
    return sf37_env_present("SF37_CUDA_NO_BATCH_PREFILL", NULL);
}

static bool cuda_prefill_profile_enabled(void) {
    return sf37_env_present("SF37_CUDA_PREFILL_PROFILE",
                            "DS4_METAL_GRAPH_PREFILL_PROFILE");
}

static uint32_t cuda_resume_prefill_min_tokens(void) {
    uint32_t v = 4;
    const char *env = sf37_env_value("SF37_CUDA_RESUME_PREFILL_MIN",
                                     "DS4_METAL_RESUME_PREFILL_MIN");
    if (env) {
        char *end = NULL;
        unsigned long parsed = strtoul(env, &end, 10);
        if (end != env && parsed <= 262144ul) v = (uint32_t)parsed;
    }
    return v;
}

static uint32_t cuda_prefill_default_chunk(uint32_t suffix) {
    const char *env = sf37_env_value("SF37_CUDA_PREFILL_CHUNK",
                                     "DS4_METAL_PREFILL_CHUNK");
    if (env) {
        char *end = NULL;
        long parsed = strtol(env, &end, 10);
        if (end != env) {
            if (parsed <= 0) return suffix ? suffix : 1u;
            if (parsed > 262144l) parsed = 262144l;
            return (uint32_t)parsed;
        }
    }
    uint64_t free_b = 0;
    uint64_t total_b = 0;
    if (sf37_cuda_memory_info(&free_b, &total_b)) {
        (void)total_b;
        const uint64_t gib = 1024ull * 1024ull * 1024ull;
        if (free_b >= 24ull * gib) return 4096;
        if (free_b >= 12ull * gib) return 2048;
        if (free_b >= 6ull * gib) return 1024;
    }
    return 512;
}

static bool cuda_prefill_ensure_state(sf37_session *s, uint32_t cap) {
    if (!s || cap == 0) return false;
    if (s->cuda_prefill_ready && s->cuda_prefill.cap >= cap) return true;
    if (s->cuda_prefill_ready) {
        cuda_prefill_state_free(&s->cuda_prefill);
        s->cuda_prefill_ready = false;
    }
    if (!cuda_prefill_state_init(&s->cuda_prefill, cap)) return false;
    s->cuda_prefill_ready = true;
    return true;
}

static bool cuda_prefill_ensure_routed_mid_pairs(sf37_session *s, uint32_t cap) {
    if (!s || !s->cuda_prefill_ready || cap == 0) return false;
    const uint64_t bytes = (uint64_t)cap * SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float);
    if (s->cuda_prefill.routed_mid &&
        sf37_cuda_tensor_bytes(s->cuda_prefill.routed_mid) >= bytes) {
        return true;
    }
    sf37_cuda_tensor_free(s->cuda_prefill.routed_mid);
    s->cuda_prefill.routed_mid = sf37_cuda_tensor_alloc(bytes);
    return s->cuda_prefill.routed_mid != NULL;
}

static uint32_t cuda_prefill_select_chunk_cap(sf37_session *s, uint32_t suffix) {
    if (!s || suffix == 0) return 0;
    uint32_t wanted = cuda_prefill_default_chunk(suffix);
    const uint32_t state_cap = s->cuda_state.prefill_cap;
    if (state_cap != 0 && wanted > state_cap) wanted = state_cap;
    if (wanted > suffix) wanted = suffix;
    if (wanted == 0) wanted = suffix;
    const uint32_t fixed[] = {4096, 2048, 1024, 512, 256};
    uint32_t candidates[6];
    uint32_t n = 0;
    candidates[n++] = wanted;
    for (uint32_t i = 0; i < (uint32_t)(sizeof(fixed) / sizeof(fixed[0])); i++) {
        uint32_t c = fixed[i];
        if (state_cap != 0 && c > state_cap) c = state_cap;
        if (c > suffix) c = suffix;
        if (c == 0) continue;
        bool seen = false;
        for (uint32_t j = 0; j < n; j++) {
            if (candidates[j] == c) seen = true;
        }
        if (!seen && n < (uint32_t)(sizeof(candidates) / sizeof(candidates[0]))) {
            candidates[n++] = c;
        }
    }
    for (uint32_t i = 0; i < n; i++) {
        if (cuda_prefill_ensure_state(s, candidates[i])) return candidates[i];
    }
    return 0;
}

typedef struct {
    sf37_cuda_tensor *pixels;
    sf37_cuda_tensor *hidden;
    sf37_cuda_tensor *norm;
    sf37_cuda_tensor *qkv;
    sf37_cuda_tensor *q;
    sf37_cuda_tensor *k;
    sf37_cuda_tensor *v;
    sf37_cuda_tensor *attn;
    sf37_cuda_tensor *mlp;
    sf37_cuda_tensor *down1;
    sf37_cuda_tensor *down2;
    sf37_cuda_tensor *projected;
    uint32_t image_size;
    uint32_t grid;
    uint32_t seq;
    uint32_t down1_grid;
    uint32_t down2_grid;
    uint32_t rows;
} sf37_cuda_vision_state;

static void cuda_vision_state_free(sf37_cuda_vision_state *v) {
    if (!v) return;
    sf37_cuda_tensor_free(v->projected);
    sf37_cuda_tensor_free(v->down2);
    sf37_cuda_tensor_free(v->down1);
    sf37_cuda_tensor_free(v->mlp);
    sf37_cuda_tensor_free(v->attn);
    sf37_cuda_tensor_free(v->v);
    sf37_cuda_tensor_free(v->k);
    sf37_cuda_tensor_free(v->q);
    sf37_cuda_tensor_free(v->qkv);
    sf37_cuda_tensor_free(v->norm);
    sf37_cuda_tensor_free(v->hidden);
    sf37_cuda_tensor_free(v->pixels);
    memset(v, 0, sizeof(*v));
}

static bool cuda_vision_state_init(sf37_cuda_vision_state *v, uint32_t image_size) {
    memset(v, 0, sizeof(*v));
    if (image_size != SF37_VISION_IMAGE && image_size != SF37_VISION_PATCH_IMAGE) return false;
    if (image_size % SF37_VISION_PATCH != 0) return false;
    v->image_size = image_size;
    v->grid = image_size / SF37_VISION_PATCH;
    if ((v->grid & 3u) != 0) return false;
    v->seq = v->grid * v->grid;
    v->down1_grid = v->grid / 2u;
    v->down2_grid = v->grid / 4u;
    v->rows = v->down2_grid * v->down2_grid;
    const uint64_t seq = v->seq;
    const uint64_t rows = v->rows;
    v->pixels = sf37_cuda_tensor_alloc((uint64_t)3u * image_size * image_size * sizeof(float));
    v->hidden = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->norm = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->qkv = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * 3u * sizeof(float));
    v->q = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->k = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->v = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->attn = sf37_cuda_tensor_alloc(seq * SF37_VISION_WIDTH * sizeof(float));
    v->mlp = sf37_cuda_tensor_alloc(seq * SF37_VISION_MLP * sizeof(float));
    v->down1 = sf37_cuda_tensor_alloc((uint64_t)v->down1_grid * v->down1_grid *
                                      SF37_VISION_WIDTH * 2u * sizeof(float));
    v->down2 = sf37_cuda_tensor_alloc(rows * SF37_VISION_PROJECTOR_IN * sizeof(float));
    v->projected = sf37_cuda_tensor_alloc(rows * SF37_EMBD * sizeof(float));
    if (!v->pixels || !v->hidden || !v->norm || !v->qkv || !v->q || !v->k ||
        !v->v || !v->attn || !v->mlp || !v->down1 || !v->down2 || !v->projected) {
        cuda_vision_state_free(v);
        return false;
    }
    return true;
}

static bool tensor_must_be_bf16(const sf37_tensor *t) {
    return t && t->type == SF37_TENSOR_BF16;
}

static int cuda_vision_encode_one(sf37_engine *e,
                                  sf37_cuda_vision_state *v,
                                  const float *pixels,
                                  sf37_cuda_tensor *dst,
                                  uint64_t dst_row,
                                  char *err,
                                  size_t errlen) {
    if (!e || !v || !pixels || !dst) return 0;
    sf37_vision_weights *vw = &e->vision;
    const uint64_t pixel_bytes = (uint64_t)3u * v->image_size * v->image_size * sizeof(float);
    const uint64_t projected_bytes = (uint64_t)v->rows * SF37_EMBD * sizeof(float);
    if (sf37_cuda_tensor_bytes(dst) < (dst_row + v->rows) * (uint64_t)SF37_EMBD * sizeof(float)) {
        session_set_err(err, errlen, "CUDA vision output buffer is too small");
        return 0;
    }
    if (!tensor_must_be_bf16(vw->conv1_weight) ||
        !tensor_must_be_bf16(vw->ln_pre_weight) ||
        !tensor_must_be_bf16(vw->ln_pre_bias) ||
        !tensor_must_be_bf16(vw->down1_weight) ||
        !tensor_must_be_bf16(vw->down1_bias) ||
        !tensor_must_be_bf16(vw->down2_weight) ||
        !tensor_must_be_bf16(vw->down2_bias) ||
        !tensor_must_be_bf16(vw->projector)) {
        session_set_err(err, errlen, "CUDA native vision currently requires BF16 conv/norm/projector tensors");
        return 0;
    }
    if (!sf37_cuda_tensor_write(v->pixels, 0, pixels, pixel_bytes)) {
        session_set_err(err, errlen, "CUDA vision pixel upload failed");
        return 0;
    }
    if (!sf37_cuda_vision_conv2d_bf16_mapped(v->hidden, v->pixels,
                                             e->gguf.map, e->gguf.size,
                                             vw->conv1_weight->abs_offset,
                                             UINT64_MAX,
                                             1u, 3u,
                                             v->image_size, v->image_size,
                                             SF37_VISION_WIDTH,
                                             SF37_VISION_PATCH,
                                             SF37_VISION_PATCH,
                                             0u, 0)) {
        session_set_err(err, errlen, "CUDA vision conv1 failed");
        return 0;
    }
    if (!sf37_cuda_vision_add_pos_q8_0_mapped(v->hidden,
                                              e->gguf.map, e->gguf.size,
                                              vw->positional_embedding->abs_offset,
                                              SF37_VISION_WIDTH,
                                              v->grid, v->grid,
                                              SF37_VISION_GRID)) {
        session_set_err(err, errlen, "CUDA vision positional embedding failed");
        return 0;
    }
    if (!sf37_cuda_layer_norm_bf16_mapped(v->norm, v->hidden,
                                          e->gguf.map, e->gguf.size,
                                          vw->ln_pre_weight->abs_offset,
                                          vw->ln_pre_bias->abs_offset,
                                          SF37_VISION_WIDTH, v->seq,
                                          SF37_VISION_LN_EPS)) {
        session_set_err(err, errlen, "CUDA vision ln_pre failed");
        return 0;
    }
    sf37_cuda_tensor *tmp = v->hidden;
    v->hidden = v->norm;
    v->norm = tmp;

    for (uint32_t il = 0; il < SF37_VISION_LAYERS; il++) {
        sf37_vision_block *b = &vw->block[il];
        if (!tensor_must_be_bf16(b->in_proj_bias) ||
            !tensor_must_be_bf16(b->out_proj_bias) ||
            !tensor_must_be_bf16(b->ln1_weight) ||
            !tensor_must_be_bf16(b->ln1_bias) ||
            !tensor_must_be_bf16(b->ln2_weight) ||
            !tensor_must_be_bf16(b->ln2_bias) ||
            !tensor_must_be_bf16(b->ls1_gamma) ||
            !tensor_must_be_bf16(b->ls2_gamma) ||
            !tensor_must_be_bf16(b->mlp_fc_bias) ||
            !tensor_must_be_bf16(b->mlp_proj_bias)) {
            session_set_err(err, errlen, "CUDA native vision block %u has non-BF16 bias/norm tensors", il);
            return 0;
        }
        if (!sf37_cuda_layer_norm_bf16_mapped(v->norm, v->hidden,
                                              e->gguf.map, e->gguf.size,
                                              b->ln1_weight->abs_offset,
                                              b->ln1_bias->abs_offset,
                                              SF37_VISION_WIDTH, v->seq,
                                              SF37_VISION_LN_EPS) ||
            !sf37_cuda_matmul_q8_0_mapped(v->qkv,
                                          e->gguf.map, e->gguf.size,
                                          b->in_proj_weight->abs_offset,
                                          SF37_VISION_WIDTH,
                                          SF37_VISION_WIDTH * 3u,
                                          v->norm, v->seq) ||
            !sf37_cuda_vision_qkv_split_rope_bf16_mapped(v->q, v->k, v->v, v->qkv,
                                                         e->gguf.map, e->gguf.size,
                                                         b->in_proj_bias->abs_offset,
                                                         v->seq, v->grid, v->grid,
                                                         SF37_VISION_HEADS,
                                                         SF37_VISION_HEAD_DIM,
                                                         SF37_VISION_GRID,
                                                         SF37_VISION_ROPE_THETA) ||
            !sf37_cuda_vision_attention(v->attn, v->q, v->k, v->v,
                                        v->seq,
                                        SF37_VISION_HEADS,
                                        SF37_VISION_HEAD_DIM) ||
            !sf37_cuda_matmul_q8_0_mapped(v->norm,
                                          e->gguf.map, e->gguf.size,
                                          b->out_proj_weight->abs_offset,
                                          SF37_VISION_WIDTH,
                                          SF37_VISION_WIDTH,
                                          v->attn, v->seq) ||
            !sf37_cuda_add_bias_bf16_mapped(v->norm,
                                            e->gguf.map, e->gguf.size,
                                            b->out_proj_bias->abs_offset,
                                            SF37_VISION_WIDTH, v->seq) ||
            !sf37_cuda_add_scaled_bf16_mapped(v->hidden, v->norm,
                                              e->gguf.map, e->gguf.size,
                                              b->ls1_gamma->abs_offset,
                                              SF37_VISION_WIDTH, v->seq)) {
            session_set_err(err, errlen, "CUDA vision attention block %u failed", il);
            return 0;
        }
        if (!sf37_cuda_layer_norm_bf16_mapped(v->norm, v->hidden,
                                              e->gguf.map, e->gguf.size,
                                              b->ln2_weight->abs_offset,
                                              b->ln2_bias->abs_offset,
                                              SF37_VISION_WIDTH, v->seq,
                                              SF37_VISION_LN_EPS) ||
            !sf37_cuda_matmul_q8_0_mapped(v->mlp,
                                          e->gguf.map, e->gguf.size,
                                          b->mlp_fc_weight->abs_offset,
                                          SF37_VISION_WIDTH,
                                          SF37_VISION_MLP,
                                          v->norm, v->seq) ||
            !sf37_cuda_add_bias_bf16_mapped(v->mlp,
                                            e->gguf.map, e->gguf.size,
                                            b->mlp_fc_bias->abs_offset,
                                            SF37_VISION_MLP, v->seq) ||
            !sf37_cuda_quick_gelu_f32(v->mlp, (uint64_t)v->seq * SF37_VISION_MLP) ||
            !sf37_cuda_matmul_q8_0_mapped(v->norm,
                                          e->gguf.map, e->gguf.size,
                                          b->mlp_proj_weight->abs_offset,
                                          SF37_VISION_MLP,
                                          SF37_VISION_WIDTH,
                                          v->mlp, v->seq) ||
            !sf37_cuda_add_bias_bf16_mapped(v->norm,
                                            e->gguf.map, e->gguf.size,
                                            b->mlp_proj_bias->abs_offset,
                                            SF37_VISION_WIDTH, v->seq) ||
            !sf37_cuda_add_scaled_bf16_mapped(v->hidden, v->norm,
                                              e->gguf.map, e->gguf.size,
                                              b->ls2_gamma->abs_offset,
                                              SF37_VISION_WIDTH, v->seq)) {
            session_set_err(err, errlen, "CUDA vision MLP block %u failed", il);
            return 0;
        }
    }

    if (!sf37_cuda_vision_conv2d_bf16_mapped(v->down1, v->hidden,
                                             e->gguf.map, e->gguf.size,
                                             vw->down1_weight->abs_offset,
                                             vw->down1_bias->abs_offset,
                                             1u,
                                             SF37_VISION_WIDTH,
                                             v->grid, v->grid,
                                             SF37_VISION_WIDTH * 2u,
                                             3u, 2u, 1u, 1) ||
        !sf37_cuda_vision_conv2d_bf16_mapped(v->down2, v->down1,
                                             e->gguf.map, e->gguf.size,
                                             vw->down2_weight->abs_offset,
                                             vw->down2_bias->abs_offset,
                                             1u,
                                             SF37_VISION_WIDTH * 2u,
                                             v->down1_grid, v->down1_grid,
                                             SF37_VISION_PROJECTOR_IN,
                                             3u, 2u, 1u, 1) ||
        !sf37_cuda_matmul_bf16_mapped(v->projected,
                                      e->gguf.map, e->gguf.size,
                                      vw->projector->abs_offset,
                                      SF37_VISION_PROJECTOR_IN,
                                      SF37_EMBD,
                                      v->down2,
                                      v->rows)) {
        session_set_err(err, errlen, "CUDA vision downsampler/projector failed");
        return 0;
    }
    if (!sf37_cuda_tensor_copy(dst, dst_row * (uint64_t)SF37_EMBD * sizeof(float),
                               v->projected, 0, projected_bytes)) {
        session_set_err(err, errlen, "CUDA vision projected feature copy failed");
        return 0;
    }
    return 1;
}

static int cuda_prefill_prepare_image_pixels(sf37_session *s,
                                             const sf37_image_features *features,
                                             char *err,
                                             size_t errlen) {
    if (!features ||
        ((!features->pixel_values || features->images == 0) &&
         (!features->patch_pixel_values || features->patch_images == 0))) return 1;
    if (!s || !s->engine || !s->cuda_prefill_ready) return 0;
    uint32_t total_rows = 0;
    if (!image_pixels_mixed_feature_rows(features, &total_rows, err, errlen)) return 0;
    const uint64_t out_count = (uint64_t)total_rows * SF37_EMBD;
    sf37_cuda_prefill_state *p = &s->cuda_prefill;
    if (p->image_cap < total_rows || p->image_dim != SF37_EMBD || !p->image_proj) {
        sf37_cuda_tensor_free(p->image_in);
        sf37_cuda_tensor_free(p->image_proj);
        p->image_in = NULL;
        p->image_proj = sf37_cuda_tensor_alloc(out_count * sizeof(float));
        p->image_cap = total_rows;
        p->image_dim = SF37_EMBD;
        if (!p->image_proj) {
            session_set_err(err, errlen, "CUDA raw vision output allocation failed");
            return 0;
        }
    }

    int ok = 1;
    if (features->patch_pixel_values && features->patch_images > 0) {
        sf37_cuda_vision_state vs_main = {0};
        sf37_cuda_vision_state vs_patch = {0};
        if (!cuda_vision_state_init(&vs_main, SF37_VISION_IMAGE) ||
            !cuda_vision_state_init(&vs_patch, SF37_VISION_PATCH_IMAGE)) {
            cuda_vision_state_free(&vs_main);
            cuda_vision_state_free(&vs_patch);
            session_set_err(err, errlen, "CUDA mixed vision scratch allocation failed");
            return 0;
        }
        const uint64_t main_floats = (uint64_t)3u * SF37_VISION_IMAGE * SF37_VISION_IMAGE;
        const uint64_t patch_floats = (uint64_t)3u * SF37_VISION_PATCH_IMAGE * SF37_VISION_PATCH_IMAGE;
        uint32_t patch_index = 0;
        uint64_t dst_row = 0;
        for (uint32_t i = 0; ok && i < features->images; i++) {
            const uint32_t np = features->patches_per_image ? features->patches_per_image[i] : 0u;
            for (uint32_t j = 0; j < np; j++) {
                if (patch_index >= features->patch_images) {
                    session_set_err(err, errlen, "mixed image patch index overflow");
                    ok = 0;
                    break;
                }
                const float *pix = features->patch_pixel_values +
                                   (uint64_t)patch_index * patch_floats;
                if (!cuda_vision_encode_one(s->engine, &vs_patch, pix, p->image_proj,
                                            dst_row, err, errlen)) {
                    ok = 0;
                    break;
                }
                dst_row += SF37_VISION_PATCH_FEATURES;
                patch_index++;
            }
            if (!ok) break;
            const float *main_pix = features->pixel_values + (uint64_t)i * main_floats;
            if (!cuda_vision_encode_one(s->engine, &vs_main, main_pix, p->image_proj,
                                        dst_row, err, errlen)) {
                ok = 0;
                break;
            }
            dst_row += SF37_VISION_FEATURES;
        }
        if (ok && patch_index != features->patch_images) {
            session_set_err(err, errlen, "unused mixed image patches: %u of %u",
                            features->patch_images - patch_index, features->patch_images);
            ok = 0;
        }
        cuda_vision_state_free(&vs_main);
        cuda_vision_state_free(&vs_patch);
        return ok;
    }

    uint32_t rows_per_image = 0;
    if (!image_pixels_feature_rows(features, &rows_per_image, err, errlen)) return 0;
    sf37_cuda_vision_state vs;
    if (!cuda_vision_state_init(&vs, features->pixel_height)) {
        session_set_err(err, errlen, "CUDA vision scratch allocation failed");
        return 0;
    }
    const uint64_t image_floats = (uint64_t)features->pixel_channels *
                                  features->pixel_height *
                                  features->pixel_width;
    for (uint32_t i = 0; i < features->images; i++) {
        const float *pix = features->pixel_values + (uint64_t)i * image_floats;
        if (!cuda_vision_encode_one(s->engine, &vs, pix, p->image_proj,
                                    (uint64_t)i * rows_per_image,
                                    err, errlen)) {
            ok = 0;
            break;
        }
    }
    cuda_vision_state_free(&vs);
    return ok;
}

static int cuda_prefill_prepare_image_projection(sf37_session *s,
                                                 const sf37_image_features *features,
                                                 char *err,
                                                 size_t errlen) {
    if (features && features->pixel_values && features->images > 0) {
        return cuda_prefill_prepare_image_pixels(s, features, err, errlen);
    }
    if (!features || features->rows == 0 || features->dim == SF37_EMBD) return 1;
    if (!s || !s->engine || !s->cuda_prefill_ready) return 0;
    if (features->dim != SF37_VISION_PROJECTOR_IN) {
        session_set_err(err, errlen, "unsupported image feature dim %u", features->dim);
        return 0;
    }
    const uint64_t in_count = (uint64_t)features->rows * features->dim;
    const uint64_t out_count = (uint64_t)features->rows * SF37_EMBD;
    if (!features->data || in_count > UINT64_MAX / sizeof(float) ||
        out_count > UINT64_MAX / sizeof(float)) {
        session_set_err(err, errlen, "invalid image feature buffer");
        return 0;
    }
    sf37_cuda_prefill_state *p = &s->cuda_prefill;
    if (p->image_cap < features->rows || p->image_dim != features->dim ||
        !p->image_in || !p->image_proj) {
        sf37_cuda_tensor_free(p->image_in);
        sf37_cuda_tensor_free(p->image_proj);
        p->image_in = sf37_cuda_tensor_alloc(in_count * sizeof(float));
        p->image_proj = sf37_cuda_tensor_alloc(out_count * sizeof(float));
        p->image_cap = features->rows;
        p->image_dim = features->dim;
        if (!p->image_in || !p->image_proj) {
            session_set_err(err, errlen, "CUDA image feature projection scratch allocation failed");
            return 0;
        }
    }
    if (!sf37_cuda_tensor_write(p->image_in, 0, features->data, in_count * sizeof(float))) {
        session_set_err(err, errlen, "CUDA image feature upload failed");
        return 0;
    }
    if (!sf37_cuda_matmul_bf16_mapped(p->image_proj,
                                      s->engine->gguf.map,
                                      s->engine->gguf.size,
                                      s->engine->vision.projector->abs_offset,
                                      SF37_VISION_PROJECTOR_IN,
                                      SF37_EMBD,
                                      p->image_in,
                                      features->rows)) {
        session_set_err(err, errlen, "CUDA image feature projector failed");
        return 0;
    }
    return 1;
}

static int cuda_prefill_apply_image_features(sf37_session *s,
                                             const sf37_tokens *prompt,
                                             uint32_t pos0,
                                             uint32_t n_tok,
                                             const sf37_image_features *features,
                                             char *err,
                                             size_t errlen) {
    if (!features) return 1;
    uint32_t total_rows = 0;
    if (!image_features_effective_rows(features, &total_rows, err, errlen)) return 0;
    if (total_rows == 0) return 1;
    sf37_engine *e = s ? s->engine : NULL;
    if (!e || e->vocab.im_patch_id < 0) {
        session_set_err(err, errlen, "model tokenizer has no <im_patch> token");
        return 0;
    }
    uint32_t feature_row = prompt_count_im_patch(e, prompt, 0, pos0);
    for (uint32_t t = 0; t < n_tok; t++) {
        if (prompt->v[pos0 + t] != e->vocab.im_patch_id) continue;
        if (feature_row >= total_rows) {
            session_set_err(err, errlen, "not enough image feature rows for <im_patch>");
            return 0;
        }
        const uint64_t dst_off = (uint64_t)t * SF37_EMBD * sizeof(float);
        if (features->pixel_values && features->images > 0) {
            const uint64_t src_off = (uint64_t)feature_row * SF37_EMBD * sizeof(float);
            if (!s->cuda_prefill.image_proj ||
                !sf37_cuda_tensor_copy(s->cuda_prefill.hidden, dst_off,
                                       s->cuda_prefill.image_proj, src_off,
                                       (uint64_t)SF37_EMBD * sizeof(float))) {
                session_set_err(err, errlen, "CUDA native image feature copy failed");
                return 0;
            }
        } else if (features->dim == SF37_EMBD) {
            const float *src = features->data + (uint64_t)feature_row * SF37_EMBD;
            if (!sf37_cuda_tensor_write(s->cuda_prefill.hidden, dst_off, src,
                                        (uint64_t)SF37_EMBD * sizeof(float))) {
                session_set_err(err, errlen, "CUDA image feature row upload failed");
                return 0;
            }
        } else if (features->dim == SF37_VISION_PROJECTOR_IN) {
            const uint64_t src_off = (uint64_t)feature_row * SF37_EMBD * sizeof(float);
            if (!s->cuda_prefill.image_proj ||
                !sf37_cuda_tensor_copy(s->cuda_prefill.hidden, dst_off,
                                       s->cuda_prefill.image_proj, src_off,
                                       (uint64_t)SF37_EMBD * sizeof(float))) {
                session_set_err(err, errlen, "CUDA projected image feature copy failed");
                return 0;
            }
        } else {
            session_set_err(err, errlen, "unsupported image feature dim %u", features->dim);
            return 0;
        }
        feature_row++;
    }
    return 1;
}

struct sf37_imatrix_collector {
    float *gate_up_sum2;
    float *down_sum2;
    uint32_t *gate_up_count;
    uint32_t *down_count;
    float *ffn_norm_buf;
    float *routed_mid_buf;
    int32_t *selected_buf;
    float *sq_tmp;
    uint32_t cap_tokens;
    uint64_t observed_tokens;
    uint64_t observed_routes;
    uint32_t chunks;
    const char *dataset_path;
};

static bool imatrix_collector_init(sf37_imatrix_collector *c, const char *dataset_path) {
    memset(c, 0, sizeof(*c));
    c->dataset_path = dataset_path;
    const size_t gate_n = (size_t)SF37_MAIN_LAYERS * SF37_EXPERTS * SF37_EMBD;
    const size_t down_n = (size_t)SF37_MAIN_LAYERS * SF37_EXPERTS * SF37_EXPERT_FF;
    const size_t count_n = (size_t)SF37_MAIN_LAYERS * SF37_EXPERTS;
    c->gate_up_sum2 = xcalloc(gate_n, sizeof(c->gate_up_sum2[0]));
    c->down_sum2 = xcalloc(down_n, sizeof(c->down_sum2[0]));
    c->gate_up_count = xcalloc(count_n, sizeof(c->gate_up_count[0]));
    c->down_count = xcalloc(count_n, sizeof(c->down_count[0]));
    c->sq_tmp = xmalloc((size_t)SF37_EMBD * sizeof(c->sq_tmp[0]));
    return c->gate_up_sum2 && c->down_sum2 && c->gate_up_count &&
           c->down_count && c->sq_tmp;
}

static void imatrix_collector_free(sf37_imatrix_collector *c) {
    if (!c) return;
    free(c->gate_up_sum2);
    free(c->down_sum2);
    free(c->gate_up_count);
    free(c->down_count);
    free(c->ffn_norm_buf);
    free(c->routed_mid_buf);
    free(c->selected_buf);
    free(c->sq_tmp);
    memset(c, 0, sizeof(*c));
}

static bool imatrix_collector_ensure_cap(sf37_imatrix_collector *c, uint32_t n_tok) {
    if (!c) return false;
    if (n_tok <= c->cap_tokens) return true;
    c->ffn_norm_buf = xrealloc(c->ffn_norm_buf,
                               (size_t)n_tok * SF37_EMBD * sizeof(c->ffn_norm_buf[0]));
    c->routed_mid_buf = xrealloc(c->routed_mid_buf,
                                 (size_t)n_tok * SF37_EXPERT_USED *
                                 SF37_EXPERT_FF * sizeof(c->routed_mid_buf[0]));
    c->selected_buf = xrealloc(c->selected_buf,
                               (size_t)n_tok * SF37_EXPERT_USED *
                               sizeof(c->selected_buf[0]));
    c->cap_tokens = n_tok;
    return c->ffn_norm_buf && c->routed_mid_buf && c->selected_buf;
}

static float *imatrix_gate_up_ptr(sf37_imatrix_collector *c,
                                  uint32_t il,
                                  uint32_t expert) {
    return c->gate_up_sum2 + ((size_t)il * SF37_EXPERTS + expert) * SF37_EMBD;
}

static float *imatrix_down_ptr(sf37_imatrix_collector *c,
                               uint32_t il,
                               uint32_t expert) {
    return c->down_sum2 + ((size_t)il * SF37_EXPERTS + expert) * SF37_EXPERT_FF;
}

static uint32_t *imatrix_gate_up_count_ptr(sf37_imatrix_collector *c,
                                           uint32_t il,
                                           uint32_t expert) {
    return c->gate_up_count + (size_t)il * SF37_EXPERTS + expert;
}

static uint32_t *imatrix_down_count_ptr(sf37_imatrix_collector *c,
                                        uint32_t il,
                                        uint32_t expert) {
    return c->down_count + (size_t)il * SF37_EXPERTS + expert;
}

static bool imatrix_collect_layer_batch(sf37_imatrix_collector *c,
                                        sf37_cuda_prefill_state *p,
                                        uint32_t il,
                                        uint32_t n_tok) {
    if (!c || !p || n_tok == 0 || il < 3u || il >= SF37_MAIN_LAYERS) return true;
    if (!imatrix_collector_ensure_cap(c, n_tok)) return false;
    const uint64_t norm_bytes = (uint64_t)n_tok * SF37_EMBD * sizeof(float);
    const uint64_t mid_bytes = (uint64_t)n_tok * SF37_EXPERT_USED *
                               SF37_EXPERT_FF * sizeof(float);
    const uint64_t selected_bytes = (uint64_t)n_tok * SF37_EXPERT_USED * sizeof(int32_t);
    if (!p->routed_mid || sf37_cuda_tensor_bytes(p->routed_mid) < mid_bytes) {
        return false;
    }
    if (!sf37_cuda_tensor_read(p->ffn_norm, 0, c->ffn_norm_buf, norm_bytes) ||
        !sf37_cuda_tensor_read(p->routed_mid, 0, c->routed_mid_buf, mid_bytes) ||
        !sf37_cuda_tensor_read(p->router_selected, 0, c->selected_buf, selected_bytes)) {
        return false;
    }

    for (uint32_t t = 0; t < n_tok; t++) {
        const float *x = c->ffn_norm_buf + (size_t)t * SF37_EMBD;
        for (uint32_t i = 0; i < SF37_EMBD; i++) c->sq_tmp[i] = x[i] * x[i];

        for (uint32_t slot = 0; slot < SF37_EXPERT_USED; slot++) {
            const int32_t expert_i = c->selected_buf[(size_t)t * SF37_EXPERT_USED + slot];
            if (expert_i < 0 || expert_i >= (int32_t)SF37_EXPERTS) continue;
            const uint32_t expert = (uint32_t)expert_i;

            float *gate_up = imatrix_gate_up_ptr(c, il, expert);
            for (uint32_t i = 0; i < SF37_EMBD; i++) gate_up[i] += c->sq_tmp[i];
            uint32_t *gcnt = imatrix_gate_up_count_ptr(c, il, expert);
            if (*gcnt != UINT32_MAX) (*gcnt)++;

            float *down = imatrix_down_ptr(c, il, expert);
            const float *mid = c->routed_mid_buf +
                               ((size_t)t * SF37_EXPERT_USED + slot) * SF37_EXPERT_FF;
            for (uint32_t i = 0; i < SF37_EXPERT_FF; i++) down[i] += mid[i] * mid[i];
            uint32_t *dcnt = imatrix_down_count_ptr(c, il, expert);
            if (*dcnt != UINT32_MAX) (*dcnt)++;
            c->observed_routes++;
        }
    }
    c->observed_tokens += n_tok;
    c->chunks++;
    return true;
}

static int cuda_prefill_layer(sf37_engine *e,
                              sf37_cuda_decode_state *d,
                              sf37_cuda_prefill_state *p,
                              uint32_t il,
                              uint32_t pos0,
                              uint32_t n_tok,
                              sf37_imatrix_collector *imatrix) {
    sf37_layer_weights *l = &e->layer[il];
    const uint32_t q_heads = layer_q_heads(il);
    const uint32_t q_dim = q_heads * SF37_HEAD_DIM;
    const uint32_t kv_dim = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t rotary_dim = layer_rotary_dim(il);
    const double theta = layer_rope_theta(il);
    const int full = layer_is_full(il) ? 1 : 0;
    const uint32_t cache_cap = d->cache_cap[il] ? d->cache_cap[il] : 1u;
    int ok = 1;
    const bool profile = cuda_prefill_profile_enabled();
    const double t0 = profile ? sf37_now_sec() : 0.0;
    (void)sf37_cuda_begin_layer(il);
    cuda_preload_layer_weights(e, il);
    if (cuda_prefill_q8_cache_enabled(n_tok)) {
        cuda_prefill_cache_layer_q8(e, il, NULL, NULL);
    }

#define CUDA_PREFILL_CHECK(expr, label) do { \
        if (!(expr)) { \
            sf37_log(stderr, SF37_LOG_ERROR, "CUDA prefill layer %u failed at %s", il, label); \
            ok = 0; \
            goto cleanup; \
        } \
    } while (0)

    CUDA_PREFILL_CHECK(sf37_cuda_rms_norm_weight1_bf16_batch_mapped(p->attn_norm, p->hidden,
                                                                    e->gguf.map, e->gguf.size,
                                                                    l->input_norm->abs_offset,
                                                                    SF37_EMBD, n_tok,
                                                                    SF37_RMS_EPS),
                       "input rms batch");
    CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->q, e->gguf.map, e->gguf.size,
                                                    l->q_proj->abs_offset,
                                                    SF37_EMBD, q_dim, p->attn_norm, n_tok),
                       "q_proj batch");
    CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->k, e->gguf.map, e->gguf.size,
                                                    l->k_proj->abs_offset,
                                                    SF37_EMBD, kv_dim, p->attn_norm, n_tok),
                       "k_proj batch");
    CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->v, e->gguf.map, e->gguf.size,
                                                    l->v_proj->abs_offset,
                                                    SF37_EMBD, kv_dim, p->attn_norm, n_tok),
                       "v_proj batch");
    CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->head_gate, e->gguf.map, e->gguf.size,
                                                    l->g_proj->abs_offset,
                                                    SF37_EMBD, q_heads, p->attn_norm, n_tok),
                       "g_proj batch");
    CUDA_PREFILL_CHECK(sf37_cuda_head_rms_norm_weight1_bf16_batch_mapped(p->q,
                                                                         e->gguf.map, e->gguf.size,
                                                                         l->q_norm->abs_offset,
                                                                         n_tok, q_heads,
                                                                         SF37_HEAD_DIM,
                                                                         SF37_RMS_EPS),
                       "q head norm batch");
    CUDA_PREFILL_CHECK(sf37_cuda_head_rms_norm_weight1_bf16_batch_mapped(p->k,
                                                                         e->gguf.map, e->gguf.size,
                                                                         l->k_norm->abs_offset,
                                                                         n_tok, SF37_KV_HEADS,
                                                                         SF37_HEAD_DIM,
                                                                         SF37_RMS_EPS),
                       "k head norm batch");
    CUDA_PREFILL_CHECK(sf37_cuda_rope_split_half_batch(p->q, n_tok, q_heads, SF37_HEAD_DIM,
                                                       rotary_dim, theta, full, pos0),
                       "q rope batch");
    CUDA_PREFILL_CHECK(sf37_cuda_rope_split_half_batch(p->k, n_tok, SF37_KV_HEADS, SF37_HEAD_DIM,
                                                       rotary_dim, theta, full, pos0),
                       "k rope batch");
    CUDA_PREFILL_CHECK(sf37_cuda_store_kv_cache_batch(d->k_cache[il], d->v_cache[il],
                                                      p->k, p->v, pos0, n_tok,
                                                      cache_cap, kv_dim),
                       "kv cache store batch");
    CUDA_PREFILL_CHECK(sf37_cuda_attention_prefill_heads(p->attn_heads, p->q,
                                                         d->k_cache[il], d->v_cache[il],
                                                         p->head_gate,
                                                         pos0, n_tok, cache_cap,
                                                         q_heads, SF37_KV_HEADS,
                                                         SF37_HEAD_DIM,
                                                         full ? 0 : 1,
                                                         SF37_SLIDING_WINDOW),
                       "attention prefill batch");
    CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->attn_out, e->gguf.map, e->gguf.size,
                                                    l->o_proj->abs_offset,
                                                    q_dim, SF37_EMBD, p->attn_heads, n_tok),
                       "o_proj batch");
    CUDA_PREFILL_CHECK(sf37_cuda_add_inplace_f32(p->hidden, p->attn_out,
                                                 (uint64_t)n_tok * SF37_EMBD),
                       "attention residual batch");
    CUDA_PREFILL_CHECK(sf37_cuda_rms_norm_weight1_bf16_batch_mapped(p->ffn_norm, p->hidden,
                                                                    e->gguf.map, e->gguf.size,
                                                                    l->post_norm->abs_offset,
                                                                    SF37_EMBD, n_tok,
                                                                    SF37_RMS_EPS),
                       "post rms batch");
    if (il < 3u) {
        CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_pair_mapped(p->gate, p->up,
                                                             e->gguf.map, e->gguf.size,
                                                             l->mlp_gate->abs_offset,
                                                             l->mlp_up->abs_offset,
                                                             SF37_EMBD, SF37_DENSE_FF,
                                                             p->ffn_norm, n_tok),
                           "dense gate/up batch");
        CUDA_PREFILL_CHECK(sf37_cuda_swiglu_f32(p->mid, p->gate, p->up,
                                                (uint64_t)n_tok * SF37_DENSE_FF,
                                                layer_swiglu_limit(il, true)),
                           "dense swiglu batch");
        CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->ffn_out,
                                                        e->gguf.map, e->gguf.size,
                                                        l->mlp_down->abs_offset,
                                                        SF37_DENSE_FF, SF37_EMBD,
                                                        p->mid, n_tok),
                           "dense down batch");
    } else {
        CUDA_PREFILL_CHECK(cuda_matmul_dense_mapped(p->router_logits, e, l->router_gate,
                                                    p->ffn_norm, n_tok),
                           "router gate batch");
        CUDA_PREFILL_CHECK(sf37_cuda_router_select_batch_mapped(p->router_selected,
                                                                p->router_weights,
                                                                p->router_probs,
                                                                p->router_logits,
                                                                e->gguf.map, e->gguf.size,
                                                                l->router_bias->abs_offset,
                                                                SF37_EXPERTS,
                                                                SF37_EXPERT_USED,
                                                                3.0f,
                                                                n_tok),
                           "router select batch");
        CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_pair_mapped(p->gate, p->up,
                                                             e->gguf.map, e->gguf.size,
                                                             l->share_gate->abs_offset,
                                                             l->share_up->abs_offset,
                                                             SF37_EMBD, SF37_EXPERT_FF,
                                                             p->ffn_norm, n_tok),
                           "shared gate/up batch");
        CUDA_PREFILL_CHECK(sf37_cuda_swiglu_f32(p->mid, p->gate, p->up,
                                                (uint64_t)n_tok * SF37_EXPERT_FF,
                                                layer_swiglu_limit(il, true)),
                           "shared swiglu batch");
        CUDA_PREFILL_CHECK(sf37_cuda_matmul_q8_0_mapped(p->ffn_out,
                                                        e->gguf.map, e->gguf.size,
                                                        l->share_down->abs_offset,
                                                        SF37_EXPERT_FF, SF37_EMBD,
                                                        p->mid, n_tok),
                           "shared down batch");
        if (sf37_env_present("SF37_CUDA_NO_BATCH_MOE", NULL)) {
            for (uint32_t t = 0; t < n_tok; t++) {
                const uint64_t x_off = (uint64_t)t * SF37_EMBD * sizeof(float);
                const uint64_t sel_off = (uint64_t)t * SF37_EXPERT_USED * sizeof(int32_t);
                const uint64_t w_off = (uint64_t)t * SF37_EXPERT_USED * sizeof(float);
                const uint64_t out_off = (uint64_t)t * SF37_EMBD * sizeof(float);
                CUDA_PREFILL_CHECK(sf37_cuda_tensor_copy(d->ffn_norm, 0, p->ffn_norm, x_off,
                                                         (uint64_t)SF37_EMBD * sizeof(float)),
                                   "batch moe fallback row copy");
                CUDA_PREFILL_CHECK(sf37_cuda_tensor_copy(d->router_selected, 0,
                                                         p->router_selected, sel_off,
                                                         (uint64_t)SF37_EXPERT_USED * sizeof(int32_t)),
                                   "batch moe fallback selected copy");
                CUDA_PREFILL_CHECK(sf37_cuda_tensor_copy(d->router_weights, 0,
                                                         p->router_weights, w_off,
                                                         (uint64_t)SF37_EXPERT_USED * sizeof(float)),
                                   "batch moe fallback weights copy");
                CUDA_PREFILL_CHECK(sf37_cuda_routed_moe_one_mapped(d->routed_out, d->gate, d->up,
                                                                   d->mid, d->routed_down,
                                                                   e->gguf.map, e->gguf.size,
                                                                   l->moe_gate->abs_offset,
                                                                   l->moe_up->abs_offset,
                                                                   l->moe_down->abs_offset,
                                                                   d->router_selected,
                                                                   d->router_weights,
                                                                   SF37_EXPERTS,
                                                                   SF37_EXPERT_USED,
                                                                   SF37_EMBD,
                                                                   SF37_EXPERT_FF,
                                                                   SF37_EMBD,
                                                                   layer_swiglu_limit(il, false),
                                                                   d->ffn_norm),
                                   "batch moe fallback single");
                CUDA_PREFILL_CHECK(sf37_cuda_tensor_copy(p->routed_out, out_off,
                                                         d->routed_out, 0,
                                                         (uint64_t)SF37_EMBD * sizeof(float)),
                                   "batch moe fallback out copy");
                if (imatrix &&
                    sf37_cuda_tensor_bytes(p->routed_mid) >=
                        (uint64_t)n_tok * SF37_EXPERT_USED * SF37_EXPERT_FF * sizeof(float)) {
                    CUDA_PREFILL_CHECK(sf37_cuda_tensor_copy(
                                           p->routed_mid,
                                           (uint64_t)t * SF37_EXPERT_USED *
                                               SF37_EXPERT_FF * sizeof(float),
                                           d->mid, 0,
                                           (uint64_t)SF37_EXPERT_USED *
                                               SF37_EXPERT_FF * sizeof(float)),
                                       "batch moe fallback mid copy");
                }
            }
        } else {
            CUDA_PREFILL_CHECK(sf37_cuda_routed_moe_batch_mapped(p->routed_out,
                                                                 p->routed_gate,
                                                                 p->routed_up,
                                                                 p->routed_mid,
                                                                 p->routed_down,
                                                                 e->gguf.map, e->gguf.size,
                                                                 l->moe_gate->abs_offset,
                                                                 l->moe_up->abs_offset,
                                                                 l->moe_down->abs_offset,
                                                                 p->router_selected,
                                                                 p->router_weights,
                                                                 SF37_EXPERTS,
                                                                 SF37_EXPERT_USED,
                                                                 SF37_EMBD,
                                                                 SF37_EXPERT_FF,
                                                                 SF37_EMBD,
                                                                 layer_swiglu_limit(il, false),
                                                                 p->ffn_norm,
                                                                 n_tok),
                                "routed moe batch");
        }
        if (imatrix) {
            CUDA_PREFILL_CHECK(imatrix_collect_layer_batch(imatrix, p, il, n_tok),
                               "imatrix collect");
        }
        CUDA_PREFILL_CHECK(sf37_cuda_add_inplace_f32(p->ffn_out, p->routed_out,
                                                     (uint64_t)n_tok * SF37_EMBD),
                           "shared+routed add batch");
    }
    CUDA_PREFILL_CHECK(sf37_cuda_add_inplace_f32(p->hidden, p->ffn_out,
                                                 (uint64_t)n_tok * SF37_EMBD),
                       "ffn residual batch");

cleanup:
#undef CUDA_PREFILL_CHECK
    sf37_cuda_end_layer();
    if (imatrix && !sf37_env_present("SF37_CUDA_IMATRIX_KEEP_WEIGHT_CACHE", NULL)) {
        sf37_cuda_evict_model_cache("imatrix layer streaming");
    }
    if (profile) {
        const double t1 = sf37_now_sec();
        fprintf(stderr, "sf37: CUDA prefill layer=%u pos=%u tokens=%u total=%.3f ms\n",
                il, pos0, n_tok, (t1 - t0) * 1000.0);
    }
    return ok;
}

static int cuda_prefill_chunk(sf37_session *s,
                              const sf37_tokens *prompt,
                              uint32_t pos0,
                              uint32_t n_tok,
                              bool need_logits,
                              const sf37_image_features *features,
                              char *err,
                              size_t errlen) {
    if (!s || !prompt || n_tok == 0 || !s->cuda_prefill_ready) return 0;
    sf37_engine *e = s->engine;
    sf37_cuda_prefill_state *p = &s->cuda_prefill;
    int32_t *tok = xmalloc((size_t)n_tok * sizeof(tok[0]));
    for (uint32_t i = 0; i < n_tok; i++) tok[i] = (int32_t)prompt->v[pos0 + i];
    int ok = sf37_cuda_tensor_write(p->tokens, 0, tok, (uint64_t)n_tok * sizeof(tok[0]));
    free(tok);
    if (!ok) {
        session_set_err(err, errlen, "CUDA prefill token upload failed");
        return 0;
    }
    if (!sf37_cuda_embed_tokens_bf16_mapped(p->hidden,
                                            e->gguf.map,
                                            e->gguf.size,
                                            e->embed_tokens->abs_offset,
                                            SF37_EMBD,
                                            SF37_VOCAB,
                                            p->tokens,
                                            n_tok)) {
        session_set_err(err, errlen, "CUDA prefill embedding gather failed");
        return 0;
    }
    if (!cuda_prefill_apply_image_features(s, prompt, pos0, n_tok, features, err, errlen)) {
        return 0;
    }
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (!cuda_prefill_layer(e, &s->cuda_state, p, il, pos0, n_tok,
                                s->imatrix_collector)) {
            session_set_err(err, errlen, "CUDA batch prefill failed at layer %u", il);
            return 0;
        }
    }
    const uint64_t last_off = (uint64_t)(n_tok - 1u) * SF37_EMBD * sizeof(float);
    if (!sf37_cuda_tensor_copy(s->cuda_state.hidden, 0, p->hidden, last_off,
                               (uint64_t)SF37_EMBD * sizeof(float)) ||
        !sf37_cuda_tensor_read(s->cuda_state.hidden, 0, s->hidden,
                               (uint64_t)SF37_EMBD * sizeof(float))) {
        session_set_err(err, errlen, "CUDA prefill final hidden read failed");
        return 0;
    }
    if (need_logits) {
        if (!cuda_output_logits(e, &s->cuda_state) ||
            !sf37_cuda_tensor_read(s->cuda_state.logits, 0, s->logits,
                                   (uint64_t)SF37_VOCAB * sizeof(float))) {
            session_set_err(err, errlen, "CUDA prefill logits read failed");
            return 0;
        }
    }
    return 1;
}

static int cuda_prefill_chunked_range(sf37_session *s,
                                      const sf37_tokens *prompt,
                                      uint32_t start,
                                      uint32_t n_tokens,
                                      const sf37_image_features *features,
                                      char *err,
                                      size_t errlen) {
    if (!s || !prompt || n_tokens == 0) return 0;
    const uint32_t chunk_cap = cuda_prefill_select_chunk_cap(s, n_tokens);
    if (chunk_cap == 0) return 0;
    if (s->imatrix_collector && !cuda_prefill_ensure_routed_mid_pairs(s, chunk_cap)) {
        session_set_err(err, errlen, "CUDA imatrix routed mid scratch allocation failed");
        return -1;
    }
    if (!cuda_prefill_prepare_image_projection(s, features, err, errlen)) return -1;

    const bool profile = cuda_prefill_profile_enabled();
    const double t0 = profile ? sf37_now_sec() : 0.0;
    const uint32_t end = start + n_tokens;
    sf37_session_report_progress(s, "prefill_chunk", (int)start, prompt->len);
    for (uint32_t pos0 = start; pos0 < end; ) {
        if (sf37_session_cancel_requested(s)) {
            session_set_err(err, errlen, "session cancelled");
            s->checkpoint_valid = false;
            return -1;
        }
        const uint32_t remaining = end - pos0;
        uint32_t chunk = remaining < chunk_cap ? remaining : chunk_cap;
        if (start != 0 && chunk_cap != 0) {
            const uint32_t mod = pos0 % chunk_cap;
            if (mod != 0) {
                const uint32_t to_boundary = chunk_cap - mod;
                if (to_boundary < chunk) chunk = to_boundary;
            }
        }
        const bool need_logits = (pos0 + chunk == end);
        const double ct0 = profile ? sf37_now_sec() : 0.0;
        if (!cuda_prefill_chunk(s, prompt, pos0, chunk, need_logits, features, err, errlen)) {
            s->checkpoint_valid = false;
            return -1;
        }
        for (uint32_t i = 0; i < chunk; i++) sf37_tokens_push(&s->checkpoint, prompt->v[pos0 + i]);
        s->checkpoint_valid = true;
        sf37_session_report_progress(s, "prefill_chunk", (int)(pos0 + chunk), prompt->len);
        if (sf37_session_cancel_requested(s)) {
            session_set_err(err, errlen, "session cancelled");
            s->checkpoint_valid = false;
            return -1;
        }
        if (profile) {
            const double ct1 = sf37_now_sec();
            fprintf(stderr, "sf37: CUDA prefill chunk pos=%u tokens=%u cap=%u total=%.3f ms\n",
                    pos0, chunk, chunk_cap, (ct1 - ct0) * 1000.0);
        }
        pos0 += chunk;
    }
    if (profile) {
        const double t1 = sf37_now_sec();
        fprintf(stderr,
                "sf37: CUDA chunked prefill start=%u tokens=%u chunk=%u total=%.3f ms\n",
                start, n_tokens, chunk_cap, (t1 - t0) * 1000.0);
    }
    return 1;
}

static void imatrix_write_i32(FILE *fp, int32_t v) {
    if (fwrite(&v, sizeof(v), 1, fp) != 1) {
        fprintf(stderr, "sf37: failed to write imatrix\n");
        exit(1);
    }
}

static void imatrix_write_entry(FILE *fp,
                                const char *name,
                                const float *sum2,
                                const uint32_t *counts,
                                uint32_t n_expert,
                                uint32_t n_col) {
    const int32_t len = (int32_t)strlen(name);
    const int32_t ncall = 1;
    const int32_t nval = (int32_t)((uint64_t)n_expert * n_col);
    imatrix_write_i32(fp, len);
    if (fwrite(name, 1, (size_t)len, fp) != (size_t)len) {
        fprintf(stderr, "sf37: failed to write imatrix name\n");
        exit(1);
    }
    imatrix_write_i32(fp, ncall);
    imatrix_write_i32(fp, nval);

    float *tmp = xmalloc((size_t)n_col * sizeof(tmp[0]));
    for (uint32_t e = 0; e < n_expert; e++) {
        const uint32_t count = counts[e];
        const float *src = sum2 + (size_t)e * n_col;
        if (count == 0) {
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = 1.0f;
        } else {
            const float inv = 1.0f / (float)count;
            for (uint32_t i = 0; i < n_col; i++) tmp[i] = src[i] * inv;
        }
        if (fwrite(tmp, sizeof(tmp[0]), n_col, fp) != n_col) {
            fprintf(stderr, "sf37: failed to write imatrix values for %s\n", name);
            free(tmp);
            exit(1);
        }
    }
    free(tmp);
}

static bool imatrix_collector_save(const sf37_imatrix_collector *c,
                                   const sf37_engine *e,
                                   const char *path) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "sf37: failed to open imatrix output %s: %s\n",
                path, strerror(errno));
        return false;
    }
    const int32_t entries = (int32_t)((SF37_MAIN_LAYERS - 3u) * 3u);
    imatrix_write_i32(fp, entries);
    for (uint32_t il = 3; il < SF37_MAIN_LAYERS; il++) {
        const sf37_layer_weights *l = &e->layer[il];
        const size_t layer_gate_off = (size_t)il * SF37_EXPERTS * SF37_EMBD;
        const size_t layer_down_off = (size_t)il * SF37_EXPERTS * SF37_EXPERT_FF;
        const size_t count_off = (size_t)il * SF37_EXPERTS;
        imatrix_write_entry(fp, l->moe_gate->name,
                            c->gate_up_sum2 + layer_gate_off,
                            c->gate_up_count + count_off,
                            SF37_EXPERTS, SF37_EMBD);
        imatrix_write_entry(fp, l->moe_up->name,
                            c->gate_up_sum2 + layer_gate_off,
                            c->gate_up_count + count_off,
                            SF37_EXPERTS, SF37_EMBD);
        imatrix_write_entry(fp, l->moe_down->name,
                            c->down_sum2 + layer_down_off,
                            c->down_count + count_off,
                            SF37_EXPERTS, SF37_EXPERT_FF);
    }
    imatrix_write_i32(fp, (int32_t)c->chunks);
    const char *dataset = c->dataset_path ? c->dataset_path : "";
    const int32_t dataset_len = (int32_t)strlen(dataset);
    imatrix_write_i32(fp, dataset_len);
    if (dataset_len > 0 &&
        fwrite(dataset, 1, (size_t)dataset_len, fp) != (size_t)dataset_len) {
        fprintf(stderr, "sf37: failed to write imatrix dataset name\n");
        fclose(fp);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "sf37: failed to close imatrix output %s: %s\n",
                path, strerror(errno));
        return false;
    }
    return true;
}

static bool imatrix_read_text_file(const char *path, char **out, size_t *len_out) {
    *out = NULL;
    *len_out = 0;
    struct stat st;
    if (stat(path, &st) != 0) {
        fprintf(stderr, "sf37: failed to stat imatrix dataset %s: %s\n",
                path, strerror(errno));
        return false;
    }
    if (st.st_size < 0 || (uint64_t)st.st_size > (uint64_t)SIZE_MAX - 1u) {
        fprintf(stderr, "sf37: imatrix dataset is too large: %s\n", path);
        return false;
    }
    FILE *fp = fopen(path, "rb");
    if (!fp) {
        fprintf(stderr, "sf37: failed to open imatrix dataset %s: %s\n",
                path, strerror(errno));
        return false;
    }
    char *buf = xmalloc((size_t)st.st_size + 1u);
    if (st.st_size > 0 &&
        fread(buf, 1, (size_t)st.st_size, fp) != (size_t)st.st_size) {
        fprintf(stderr, "sf37: failed to read imatrix dataset %s\n", path);
        free(buf);
        fclose(fp);
        return false;
    }
    if (fclose(fp) != 0) {
        fprintf(stderr, "sf37: failed to close imatrix dataset %s: %s\n",
                path, strerror(errno));
        free(buf);
        return false;
    }
    buf[st.st_size] = '\0';
    *out = buf;
    *len_out = (size_t)st.st_size;
    return true;
}

static char *imatrix_trim_block(char *p, char *end) {
    while (p < end && isspace((unsigned char)*p)) p++;
    while (end > p && isspace((unsigned char)end[-1])) end--;
    *end = '\0';
    return p;
}

int sf37_engine_collect_imatrix(sf37_engine *e,
                                const char *dataset_path,
                                const char *output_path,
                                int ctx_size,
                                int max_prompts,
                                int max_tokens) {
    if (!e || !dataset_path || !output_path) return 1;
    if (e->backend != SF37_BACKEND_CUDA || !e->cuda_ready) {
        fprintf(stderr, "sf37: imatrix collection requires --backend cuda\n");
        return 1;
    }
    if (ctx_size <= 0) ctx_size = 8192;
    if (ctx_size > SF37_CTX) ctx_size = SF37_CTX;

    char *dataset = NULL;
    size_t dataset_len = 0;
    if (!imatrix_read_text_file(dataset_path, &dataset, &dataset_len)) return 1;

    sf37_session *session = NULL;
    if (sf37_session_create(&session, e, ctx_size) != 0) {
        fprintf(stderr, "sf37: failed to create imatrix CUDA session\n");
        free(dataset);
        return 1;
    }

    sf37_imatrix_collector collector;
    if (!imatrix_collector_init(&collector, dataset_path)) {
        fprintf(stderr, "sf37: failed to allocate imatrix collector\n");
        sf37_session_free(session);
        free(dataset);
        return 1;
    }

    fprintf(stderr,
            "sf37: collecting routed-MoE imatrix from %s (layers=%u, experts=%u, ctx=%d)\n",
            dataset_path, SF37_MAIN_LAYERS - 3u, SF37_EXPERTS, ctx_size);

    int prompts_done = 0;
    int tokens_done = 0;
    bool ok = true;
    char *cursor = dataset;
    const char *marker_lit = "===== SF37_IMATRIX_PROMPT";
    while (*cursor) {
        char *start = cursor;
        char *marker = strstr(cursor, marker_lit);
        if (marker) {
            char *nl = strchr(marker, '\n');
            if (!nl) break;
            start = nl + 1;
        } else if (prompts_done != 0) {
            break;
        }
        char *next = strstr(start, marker_lit);
        char *end = next ? next : dataset + dataset_len;
        char saved = *end;
        char *prompt_text = imatrix_trim_block(start, end);
        if (prompt_text[0] != '\0') {
            sf37_tokens prompt = {0};
            sf37_tokenize_rendered_chat(e, prompt_text, &prompt);
            if (prompt.len > ctx_size) prompt.len = ctx_size;
            if (max_tokens > 0 && prompt.len > max_tokens - tokens_done) {
                prompt.len = max_tokens - tokens_done;
            }
            if (prompt.len > 0) {
                char err[256] = {0};
                session_reset(session);
                session->imatrix_collector = &collector;
                const int rc = cuda_prefill_chunked_range(session, &prompt, 0,
                                                          (uint32_t)prompt.len,
                                                          NULL, err, sizeof(err));
                session->imatrix_collector = NULL;
                if (rc <= 0) {
                    fprintf(stderr, "sf37: imatrix prefill failed at prompt %d: %s\n",
                            prompts_done + 1, err[0] ? err : "unknown CUDA prefill error");
                    ok = false;
                    sf37_tokens_free(&prompt);
                    *end = saved;
                    break;
                }
                prompts_done++;
                tokens_done += prompt.len;
                if (prompts_done % 10 == 0) {
                    fprintf(stderr,
                            "sf37: imatrix prompts=%d tokens=%d routes=%llu\r",
                            prompts_done,
                            tokens_done,
                            (unsigned long long)collector.observed_routes);
                    fflush(stderr);
                }
            }
            sf37_tokens_free(&prompt);
        }
        *end = saved;
        if (!next) break;
        cursor = next;
        if (max_prompts > 0 && prompts_done >= max_prompts) break;
        if (max_tokens > 0 && tokens_done >= max_tokens) break;
    }
    fputc('\n', stderr);

    if (ok) {
        ok = imatrix_collector_save(&collector, e, output_path);
        if (ok) {
            fprintf(stderr,
                    "sf37: wrote imatrix %s from %d prompts, %d tokens, %llu routed expert observations\n",
                    output_path,
                    prompts_done,
                    tokens_done,
                    (unsigned long long)collector.observed_routes);
        }
    }
    imatrix_collector_free(&collector);
    sf37_session_free(session);
    free(dataset);
    return ok ? 0 : 1;
}
#endif

#ifndef SF37_USE_CUDA
int sf37_engine_collect_imatrix(sf37_engine *e,
                                const char *dataset_path,
                                const char *output_path,
                                int ctx_size,
                                int max_prompts,
                                int max_tokens) {
    (void)e;
    (void)dataset_path;
    (void)output_path;
    (void)ctx_size;
    (void)max_prompts;
    (void)max_tokens;
    fprintf(stderr, "sf37: imatrix collection requires a CUDA build\n");
    return 1;
}
#endif

static void session_reset(sf37_session *s) {
    if (!s) return;
    s->checkpoint.len = 0;
    s->checkpoint_valid = true;
    if (!session_uses_cuda(s)) {
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) s->cpu_cache.layer[il].n = 0;
    }
}

static uint32_t session_live_kv_rows(const sf37_session *s) {
    if (!s || s->checkpoint.len <= 0 || s->ctx_size <= 0) return 0;
    uint32_t n = (uint32_t)s->checkpoint.len;
    if (n > (uint32_t)s->ctx_size) n = (uint32_t)s->ctx_size;
    return n;
}

static uint32_t session_layer_kv_cap(const sf37_session *s, uint32_t il) {
    if (!s || il >= SF37_MAIN_LAYERS || s->ctx_size <= 0) return 0;
#ifdef SF37_USE_CUDA
    if (session_uses_cuda(s) && s->cuda_state_ready && s->cuda_state.cache_cap[il] != 0) {
        return s->cuda_state.cache_cap[il];
    }
#endif
    return (uint32_t)s->ctx_size;
}

static uint32_t session_layer_live_kv_rows(const sf37_session *s, uint32_t il) {
    uint32_t rows = session_live_kv_rows(s);
    const uint32_t cap = session_layer_kv_cap(s, il);
    if (cap != 0 && rows > cap) rows = cap;
    if (s && !session_uses_cuda(s) && il < SF37_MAIN_LAYERS && !layer_is_full(il) &&
        s->cpu_cache.layer[il].n != 0 && rows > s->cpu_cache.layer[il].n) {
        rows = s->cpu_cache.layer[il].n;
    }
    return rows;
}

static void session_kv_row_counts(const sf37_session *s,
                                  uint32_t rows[SF37_MAIN_LAYERS]) {
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        rows[il] = session_layer_live_kv_rows(s, il);
    }
}

static uint64_t session_live_kv_row_sum(const sf37_session *s) {
    uint64_t rows = 0;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        rows += session_layer_live_kv_rows(s, il);
    }
    return rows;
}

void sf37_session_rewind(sf37_session *s, int pos) {
    if (!s) return;
    if (pos < 0) pos = 0;
    if (pos > s->checkpoint.len) pos = s->checkpoint.len;
    s->checkpoint.len = pos;
    s->checkpoint_valid = true;
    if (!session_uses_cuda(s)) {
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            s->cpu_cache.layer[il].n = (uint32_t)pos;
        }
    }
}

typedef struct {
    uint8_t *p;
    uint64_t cap;
    uint64_t pos;
} sf37_snapshot_writer;

typedef struct {
    const uint8_t *p;
    uint64_t len;
    uint64_t pos;
} sf37_snapshot_reader;

static void snapshot_set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!errlen) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static int snapshot_write_bytes(sf37_snapshot_writer *w, const void *src, uint64_t bytes) {
    if (!w || !src || bytes > w->cap || w->pos > w->cap - bytes) return 1;
    memcpy(w->p + w->pos, src, (size_t)bytes);
    w->pos += bytes;
    return 0;
}

static int snapshot_write_u32(sf37_snapshot_writer *w, uint32_t v) {
    return snapshot_write_bytes(w, &v, sizeof(v));
}

static int snapshot_read_bytes(sf37_snapshot_reader *r, void *dst, uint64_t bytes) {
    if (!r || !dst || bytes > r->len || r->pos > r->len - bytes) return 1;
    memcpy(dst, r->p + r->pos, (size_t)bytes);
    r->pos += bytes;
    return 0;
}

static int snapshot_read_u32(sf37_snapshot_reader *r, uint32_t *v) {
    return snapshot_read_bytes(r, v, sizeof(*v));
}

static uint32_t payload_kv_groups_per_row(uint32_t elems, uint32_t group) {
    if (group == 0) return 0;
    return (elems + group - 1u) / group;
}

static uint64_t payload_encoded_kv_row_bytes(uint32_t elems, uint32_t group) {
    const uint32_t groups = payload_kv_groups_per_row(elems, group);
    return (uint64_t)groups * sizeof(float) + (uint64_t)elems * sizeof(int8_t);
}

static void payload_encode_kv_row_i8_grouped(const float *src, uint32_t elems,
                                             uint32_t group, uint8_t *dst) {
    const uint32_t groups = payload_kv_groups_per_row(elems, group);
    float *scales = (float *)dst;
    int8_t *q = (int8_t *)(dst + (uint64_t)groups * sizeof(float));
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t begin = g * group;
        uint32_t end = begin + group;
        if (end > elems) end = elems;
        float max_abs = 0.0f;
        for (uint32_t i = begin; i < end; i++) {
            const float a = fabsf(src[i]);
            if (isfinite(a) && a > max_abs) max_abs = a;
        }
        const float scale = max_abs > 0.0f ? max_abs / 127.0f : 1.0f;
        scales[g] = scale;
        for (uint32_t i = begin; i < end; i++) {
            float v = src[i];
            if (!isfinite(v)) v = 0.0f;
            int qi = (int)lrintf(v / scale);
            if (qi > 127) qi = 127;
            if (qi < -127) qi = -127;
            q[i] = (int8_t)qi;
        }
    }
}

static void payload_decode_kv_row_i8_grouped(const uint8_t *src, uint32_t elems,
                                             uint32_t group, float *dst) {
    const uint32_t groups = payload_kv_groups_per_row(elems, group);
    const float *scales = (const float *)src;
    const int8_t *q = (const int8_t *)(src + (uint64_t)groups * sizeof(float));
    for (uint32_t g = 0; g < groups; g++) {
        const uint32_t begin = g * group;
        uint32_t end = begin + group;
        if (end > elems) end = elems;
        const float scale = scales[g];
        for (uint32_t i = begin; i < end; i++) {
            dst[i] = (float)q[i] * scale;
        }
    }
}

#ifdef SF37_PAYLOAD_TEST
int main(void) {
    const uint32_t elems = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t group = SF37_SESSION_PAYLOAD_KV_GROUP;
    const uint64_t encoded_bytes = payload_encoded_kv_row_bytes(elems, group);
    if (encoded_bytes != 16u * sizeof(float) + elems) {
        fprintf(stderr, "unexpected sf37 payload row size\n");
        return 1;
    }

    float *src = xmalloc((size_t)elems * sizeof(float));
    float *dst = xmalloc((size_t)elems * sizeof(float));
    uint8_t *enc = xmalloc((size_t)encoded_bytes);
    for (uint32_t i = 0; i < elems; i++) {
        int m = (int)(i % 97u) - 48;
        src[i] = (float)m / 13.0f;
    }
    payload_encode_kv_row_i8_grouped(src, elems, group, enc);
    payload_decode_kv_row_i8_grouped(enc, elems, group, dst);
    float max_err = 0.0f;
    for (uint32_t i = 0; i < elems; i++) {
        const float err = fabsf(src[i] - dst[i]);
        if (err > max_err) max_err = err;
    }
    if (!(max_err < 0.02f)) {
        fprintf(stderr, "sf37 payload row roundtrip error too high: %.8f\n", max_err);
        free(enc);
        free(dst);
        free(src);
        return 1;
    }

    memset(src, 0, (size_t)elems * sizeof(float));
    payload_encode_kv_row_i8_grouped(src, elems, group, enc);
    payload_decode_kv_row_i8_grouped(enc, elems, group, dst);
    for (uint32_t i = 0; i < elems; i++) {
        if (dst[i] != 0.0f) {
            fprintf(stderr, "sf37 payload zero row did not roundtrip\n");
            free(enc);
            free(dst);
            free(src);
            return 1;
        }
    }

    free(enc);
    free(dst);
    free(src);
    fprintf(stderr, "sf37 payload tests passed\n");
    return 0;
}
#endif

static int payload_write_bytes(FILE *fp, const void *src, uint64_t bytes,
                               char *err, size_t errlen) {
    if (!fp || (!src && bytes)) {
        snapshot_set_err(err, errlen, "invalid session payload write");
        return 1;
    }
    const uint8_t *p = (const uint8_t *)src;
    uint64_t left = bytes;
    while (left > 0) {
        size_t chunk = left > SF37_SESSION_IO_CHUNK ?
            (size_t)SF37_SESSION_IO_CHUNK : (size_t)left;
        if (fwrite(p, 1, chunk, fp) != chunk) {
            snapshot_set_err(err, errlen, "failed to write session payload");
            return 1;
        }
        p += chunk;
        left -= chunk;
    }
    return 0;
}

static int payload_write_u32(FILE *fp, uint32_t v, char *err, size_t errlen) {
    return payload_write_bytes(fp, &v, sizeof(v), err, errlen);
}

static int payload_read_bytes(FILE *fp, void *dst, uint64_t bytes,
                              uint64_t *remaining, char *err, size_t errlen) {
    if (!fp || (!dst && bytes) || !remaining || bytes > *remaining) {
        snapshot_set_err(err, errlen, "truncated session payload");
        return 1;
    }
    uint8_t *p = (uint8_t *)dst;
    uint64_t left = bytes;
    while (left > 0) {
        size_t chunk = left > SF37_SESSION_IO_CHUNK ?
            (size_t)SF37_SESSION_IO_CHUNK : (size_t)left;
        if (fread(p, 1, chunk, fp) != chunk) {
            snapshot_set_err(err, errlen, "truncated session payload");
            return 1;
        }
        p += chunk;
        left -= chunk;
        *remaining -= chunk;
    }
    return 0;
}

static int payload_read_u32(FILE *fp, uint32_t *v, uint64_t *remaining,
                            char *err, size_t errlen) {
    return payload_read_bytes(fp, v, sizeof(*v), remaining, err, errlen);
}

static int payload_copy_file_bytes(FILE *src, FILE *dst, uint64_t bytes,
                                   char *err, size_t errlen) {
    if (!src || !dst) {
        snapshot_set_err(err, errlen, "invalid staged session payload copy");
        return 1;
    }
    uint8_t *buf = xmalloc(SF37_SESSION_IO_CHUNK);
    uint64_t left = bytes;
    int rc = 0;
    while (left > 0) {
        size_t chunk = left > SF37_SESSION_IO_CHUNK ?
            (size_t)SF37_SESSION_IO_CHUNK : (size_t)left;
        if (fread(buf, 1, chunk, src) != chunk) {
            snapshot_set_err(err, errlen, "failed to read staged session payload");
            rc = 1;
            break;
        }
        if (fwrite(buf, 1, chunk, dst) != chunk) {
            snapshot_set_err(err, errlen, "failed to write staged session payload");
            rc = 1;
            break;
        }
        left -= chunk;
    }
    free(buf);
    return rc;
}

uint64_t sf37_session_snapshot_bytes(sf37_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    const uint64_t live_rows = session_live_kv_row_sum(s);
    const uint64_t kv_row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    uint64_t bytes = (uint64_t)SF37_SESSION_SNAPSHOT_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)SF37_MAIN_LAYERS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)SF37_VOCAB * sizeof(float);
    bytes += (uint64_t)SF37_EMBD * sizeof(float);
    bytes += live_rows * kv_row * 2u * sizeof(float);
    return bytes;
}

uint64_t sf37_session_payload_bytes(sf37_session *s) {
    if (!s || !s->checkpoint_valid) return 0;
    const uint64_t live_rows = session_live_kv_row_sum(s);
    const uint32_t kv_row = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint64_t encoded_row = payload_encoded_kv_row_bytes(kv_row,
                                                              SF37_SESSION_PAYLOAD_KV_GROUP);
    uint64_t bytes = (uint64_t)SF37_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t);
    bytes += (uint64_t)SF37_MAIN_LAYERS * sizeof(uint32_t);
    bytes += (uint64_t)s->checkpoint.len * sizeof(uint32_t);
    bytes += (uint64_t)SF37_VOCAB * sizeof(float);
    bytes += (uint64_t)SF37_EMBD * sizeof(float);
    bytes += live_rows * 2u * encoded_row;
    return bytes;
}

int sf37_session_write_staged_payload(const sf37_session_payload_file *payload,
                                      FILE *fp, char *err, size_t errlen) {
    if (!payload || !payload->path || !fp) {
        snapshot_set_err(err, errlen, "invalid staged session payload");
        return 1;
    }
    FILE *src = fopen(payload->path, "rb");
    if (!src) {
        snapshot_set_err(err, errlen, "failed to open staged session payload");
        return 1;
    }
    int rc = payload_copy_file_bytes(src, fp, payload->bytes, err, errlen);
    if (fclose(src) != 0 && rc == 0) {
        snapshot_set_err(err, errlen, "failed to close staged session payload");
        return 1;
    }
    return rc;
}

void sf37_session_payload_file_free(sf37_session_payload_file *payload) {
    if (!payload) return;
    if (payload->path) {
        unlink(payload->path);
        free(payload->path);
    }
    memset(payload, 0, sizeof(*payload));
}

int sf37_session_stage_payload(sf37_session *s, sf37_session_payload_file *out,
                               char *err, size_t errlen) {
    if (!out) {
        snapshot_set_err(err, errlen, "invalid session payload staging request");
        return 1;
    }
    memset(out, 0, sizeof(*out));
    if (!s || !s->checkpoint_valid) {
        snapshot_set_err(err, errlen, "session has no valid checkpoint to stage");
        return 1;
    }

    char tmpl[] = "/tmp/sf37-session-payload.XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        snapshot_set_err(err, errlen, "failed to create staged session payload");
        return 1;
    }
    FILE *fp = fdopen(fd, "wb");
    if (!fp) {
        int saved = errno;
        close(fd);
        unlink(tmpl);
        snapshot_set_err(err, errlen, "failed to open staged session payload: %s",
                         strerror(saved));
        return 1;
    }

    int rc = sf37_session_save_payload(s, fp, err, errlen);
    if (rc == 0 && fflush(fp) != 0) {
        snapshot_set_err(err, errlen, "failed to flush staged session payload");
        rc = 1;
    }
    off_t pos = -1;
    if (rc == 0) {
        pos = ftello(fp);
        if (pos < 0) {
            snapshot_set_err(err, errlen, "failed to measure staged session payload");
            rc = 1;
        } else if ((uint64_t)pos != sf37_session_payload_bytes(s)) {
            snapshot_set_err(err, errlen, "staged session payload size mismatch");
            rc = 1;
        }
    }
    if (fclose(fp) != 0 && rc == 0) {
        snapshot_set_err(err, errlen, "failed to close staged session payload");
        rc = 1;
    }
    if (rc != 0) {
        unlink(tmpl);
        return 1;
    }
    out->path = xstrdup(tmpl);
    out->bytes = (uint64_t)pos;
    return 0;
}

int sf37_session_save_payload(sf37_session *s, FILE *fp, char *err, size_t errlen) {
    if (!s || !fp || !s->checkpoint_valid) {
        snapshot_set_err(err, errlen, "session has no valid checkpoint to save");
        return 1;
    }
#ifdef SF37_USE_CUDA
    if (session_uses_cuda(s) && !sf37_cuda_synchronize()) {
        snapshot_set_err(err, errlen, "failed to synchronize CUDA before payload save");
        return 1;
    }
#endif

    const uint32_t token_count = (uint32_t)s->checkpoint.len;
    const uint32_t live_rows = session_live_kv_rows(s);
    uint32_t row_counts[SF37_MAIN_LAYERS];
    session_kv_row_counts(s, row_counts);
    const uint32_t kv_row = SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t group = SF37_SESSION_PAYLOAD_KV_GROUP;
    const uint32_t groups = payload_kv_groups_per_row(kv_row, group);
    const uint32_t encoded_row = (uint32_t)payload_encoded_kv_row_bytes(kv_row, group);
    const uint32_t backend_kind = session_uses_cuda(s) ? 1u : 0u;
    const uint32_t header[SF37_SESSION_PAYLOAD_U32_FIELDS] = {
        SF37_SESSION_PAYLOAD_MAGIC,
        SF37_SESSION_PAYLOAD_VERSION,
        (uint32_t)s->ctx_size,
        token_count,
        live_rows,
        SF37_MAIN_LAYERS,
        SF37_KV_HEADS,
        SF37_HEAD_DIM,
        SF37_VOCAB,
        SF37_EMBD,
        backend_kind,
        SF37_SESSION_PAYLOAD_KV_I8_GROUPED,
        group,
        groups,
        encoded_row,
    };
    for (uint32_t i = 0; i < SF37_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_write_u32(fp, header[i], err, errlen) != 0) return 1;
    }
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (payload_write_u32(fp, row_counts[il], err, errlen) != 0) return 1;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (payload_write_u32(fp, (uint32_t)s->checkpoint.v[i], err, errlen) != 0) return 1;
    }
    if (payload_write_bytes(fp, s->logits, (uint64_t)SF37_VOCAB * sizeof(float),
                            err, errlen) != 0 ||
        payload_write_bytes(fp, s->hidden, (uint64_t)SF37_EMBD * sizeof(float),
                            err, errlen) != 0) {
        return 1;
    }

    float *row_buf = xmalloc((size_t)kv_row * sizeof(float));
    uint8_t *enc_buf = xmalloc(encoded_row);
    int rc = 0;
    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        for (uint32_t il = 0; rc == 0 && il < SF37_MAIN_LAYERS; il++) {
            const uint32_t rows = row_counts[il];
            const uint32_t first_pos = token_count - rows;
            const uint32_t cache_cap = s->cuda_state.cache_cap[il];
            for (uint32_t r = 0; rc == 0 && r < rows; r++) {
                const uint32_t phys = (first_pos + r) % cache_cap;
                const uint64_t off = (uint64_t)phys * kv_row * sizeof(float);
                if (!sf37_cuda_tensor_read(s->cuda_state.k_cache[il], off, row_buf,
                                           (uint64_t)kv_row * sizeof(float))) {
                    snapshot_set_err(err, errlen, "failed to read CUDA K cache for payload");
                    rc = 1;
                    break;
                }
                payload_encode_kv_row_i8_grouped(row_buf, kv_row, group, enc_buf);
                if (payload_write_bytes(fp, enc_buf, encoded_row, err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                if (!sf37_cuda_tensor_read(s->cuda_state.v_cache[il], off, row_buf,
                                           (uint64_t)kv_row * sizeof(float))) {
                    snapshot_set_err(err, errlen, "failed to read CUDA V cache for payload");
                    rc = 1;
                    break;
                }
                payload_encode_kv_row_i8_grouped(row_buf, kv_row, group, enc_buf);
                if (payload_write_bytes(fp, enc_buf, encoded_row, err, errlen) != 0) rc = 1;
            }
        }
#endif
    } else {
        for (uint32_t il = 0; rc == 0 && il < SF37_MAIN_LAYERS; il++) {
            const sf37_layer_kv_cache *layer = &s->cpu_cache.layer[il];
            const uint32_t rows = row_counts[il];
            if (layer->n < rows) {
                snapshot_set_err(err, errlen, "CPU KV cache has fewer live rows than checkpoint");
                rc = 1;
                break;
            }
            const uint32_t start = layer->n - rows;
            for (uint32_t r = 0; rc == 0 && r < rows; r++) {
                const float *k = layer->k + ((uint64_t)start + r) * kv_row;
                const float *v = layer->v + ((uint64_t)start + r) * kv_row;
                payload_encode_kv_row_i8_grouped(k, kv_row, group, enc_buf);
                if (payload_write_bytes(fp, enc_buf, encoded_row, err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                payload_encode_kv_row_i8_grouped(v, kv_row, group, enc_buf);
                if (payload_write_bytes(fp, enc_buf, encoded_row, err, errlen) != 0) rc = 1;
            }
        }
    }
    free(enc_buf);
    free(row_buf);
    return rc;
}

int sf37_session_load_payload(sf37_session *s, FILE *fp, uint64_t payload_bytes,
                              char *err, size_t errlen) {
    if (!s || !fp || payload_bytes == 0) {
        snapshot_set_err(err, errlen, "invalid session payload load");
        return 1;
    }
    uint64_t remaining = payload_bytes;
    uint32_t h[SF37_SESSION_PAYLOAD_U32_FIELDS];
    for (uint32_t i = 0; i < SF37_SESSION_PAYLOAD_U32_FIELDS; i++) {
        if (payload_read_u32(fp, &h[i], &remaining, err, errlen) != 0) return 1;
    }
    if (h[0] != SF37_SESSION_PAYLOAD_MAGIC ||
        h[1] != SF37_SESSION_PAYLOAD_VERSION) {
        snapshot_set_err(err, errlen, "unsupported session payload version");
        return 1;
    }
    uint32_t row_counts[SF37_MAIN_LAYERS];
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (payload_read_u32(fp, &row_counts[il], &remaining, err, errlen) != 0) return 1;
    }

    const uint32_t saved_ctx = h[2];
    const uint32_t token_count = h[3];
    const uint32_t live_rows = h[4];
    const uint32_t kv_row = h[6] * h[7];
    const uint32_t group = h[12];
    const uint32_t groups = h[13];
    const uint32_t encoded_row = h[14];
    if (saved_ctx > (uint32_t)s->ctx_size || token_count > (uint32_t)s->ctx_size) {
        snapshot_set_err(err, errlen, "KV checkpoint does not fit current context");
        return 1;
    }
    if (live_rows > token_count ||
        live_rows != (token_count < saved_ctx ? token_count : saved_ctx)) {
        snapshot_set_err(err, errlen, "session payload has invalid token/cache row counts");
        return 1;
    }
    uint64_t row_sum = 0;
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        const uint32_t rows = row_counts[il];
        if (rows > live_rows) {
            snapshot_set_err(err, errlen, "session payload has too many KV rows for layer %u", il);
            return 1;
        }
        if (layer_is_full(il)) {
            if (rows != live_rows) {
                snapshot_set_err(err, errlen, "session payload is missing full-attention KV rows");
                return 1;
            }
        } else {
            const uint32_t needed = live_rows < SF37_SLIDING_WINDOW ?
                                    live_rows : SF37_SLIDING_WINDOW;
            if (rows < needed) {
                snapshot_set_err(err, errlen, "session payload is missing sliding-window KV rows");
                return 1;
            }
        }
        const uint32_t cap = session_layer_kv_cap(s, il);
        if (cap != 0 && rows > cap) {
            snapshot_set_err(err, errlen, "session payload KV rows do not fit current cache");
            return 1;
        }
        row_sum += rows;
    }
    if (h[5] != SF37_MAIN_LAYERS ||
        h[6] != SF37_KV_HEADS ||
        h[7] != SF37_HEAD_DIM ||
        h[8] != SF37_VOCAB ||
        h[9] != SF37_EMBD ||
        h[11] != SF37_SESSION_PAYLOAD_KV_I8_GROUPED ||
        group != SF37_SESSION_PAYLOAD_KV_GROUP ||
        groups != payload_kv_groups_per_row(kv_row, group) ||
        encoded_row != payload_encoded_kv_row_bytes(kv_row, group)) {
        snapshot_set_err(err, errlen, "session payload was written for a different SF37 layout");
        return 1;
    }
    const uint64_t expected_bytes =
        (uint64_t)SF37_SESSION_PAYLOAD_U32_FIELDS * sizeof(uint32_t) +
        (uint64_t)SF37_MAIN_LAYERS * sizeof(uint32_t) +
        (uint64_t)token_count * sizeof(uint32_t) +
        (uint64_t)SF37_VOCAB * sizeof(float) +
        (uint64_t)SF37_EMBD * sizeof(float) +
        row_sum * 2u * encoded_row;
    if (payload_bytes != expected_bytes) {
        snapshot_set_err(err, errlen, "session payload byte count mismatch");
        return 1;
    }

    sf37_tokens new_checkpoint = {0};
    for (uint32_t i = 0; i < token_count; i++) {
        uint32_t tok = 0;
        if (payload_read_u32(fp, &tok, &remaining, err, errlen) != 0 ||
            tok >= SF37_VOCAB) {
            sf37_tokens_free(&new_checkpoint);
            snapshot_set_err(err, errlen, "session payload has invalid token data");
            return 1;
        }
        sf37_tokens_push(&new_checkpoint, (int)tok);
    }
    if (payload_read_bytes(fp, s->logits, (uint64_t)SF37_VOCAB * sizeof(float),
                           &remaining, err, errlen) != 0 ||
        payload_read_bytes(fp, s->hidden, (uint64_t)SF37_EMBD * sizeof(float),
                           &remaining, err, errlen) != 0) {
        sf37_tokens_free(&new_checkpoint);
        return 1;
    }

    float *row_buf = xmalloc((size_t)kv_row * sizeof(float));
    uint8_t *enc_buf = xmalloc(encoded_row);
    s->checkpoint_valid = false;
    int rc = 0;
    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        for (uint32_t il = 0; rc == 0 && il < SF37_MAIN_LAYERS; il++) {
            const uint32_t rows = row_counts[il];
            const uint32_t first_pos = token_count - rows;
            const uint32_t cache_cap = s->cuda_state.cache_cap[il];
            for (uint32_t rr = 0; rc == 0 && rr < rows; rr++) {
                const uint32_t phys = (first_pos + rr) % cache_cap;
                const uint64_t off = (uint64_t)phys * kv_row * sizeof(float);
                if (payload_read_bytes(fp, enc_buf, encoded_row, &remaining,
                                       err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                payload_decode_kv_row_i8_grouped(enc_buf, kv_row, group, row_buf);
                if (!sf37_cuda_tensor_write(s->cuda_state.k_cache[il], off, row_buf,
                                            (uint64_t)kv_row * sizeof(float))) {
                    snapshot_set_err(err, errlen, "failed to restore CUDA K cache from payload");
                    rc = 1;
                    break;
                }
                if (payload_read_bytes(fp, enc_buf, encoded_row, &remaining,
                                       err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                payload_decode_kv_row_i8_grouped(enc_buf, kv_row, group, row_buf);
                if (!sf37_cuda_tensor_write(s->cuda_state.v_cache[il], off, row_buf,
                                            (uint64_t)kv_row * sizeof(float))) {
                    snapshot_set_err(err, errlen, "failed to restore CUDA V cache from payload");
                    rc = 1;
                    break;
                }
            }
        }
        if (rc == 0 &&
            (!sf37_cuda_tensor_write(s->cuda_state.hidden, 0, s->hidden,
                                     (uint64_t)SF37_EMBD * sizeof(float)) ||
             !sf37_cuda_tensor_write(s->cuda_state.logits, 0, s->logits,
                                     (uint64_t)SF37_VOCAB * sizeof(float)))) {
            snapshot_set_err(err, errlen, "failed to restore CUDA session scratch from payload");
            rc = 1;
        }
#endif
    } else {
        for (uint32_t il = 0; rc == 0 && il < SF37_MAIN_LAYERS; il++) {
            sf37_layer_kv_cache *layer = &s->cpu_cache.layer[il];
            const uint32_t rows = row_counts[il];
            for (uint32_t rr = 0; rc == 0 && rr < rows; rr++) {
                if (payload_read_bytes(fp, enc_buf, encoded_row, &remaining,
                                       err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                payload_decode_kv_row_i8_grouped(enc_buf, kv_row, group,
                                                 layer->k + (uint64_t)rr * kv_row);
                if (payload_read_bytes(fp, enc_buf, encoded_row, &remaining,
                                       err, errlen) != 0) {
                    rc = 1;
                    break;
                }
                payload_decode_kv_row_i8_grouped(enc_buf, kv_row, group,
                                                 layer->v + (uint64_t)rr * kv_row);
            }
            if (rc == 0) layer->n = rows;
        }
    }
    free(enc_buf);
    free(row_buf);
    if (rc != 0) {
        sf37_tokens_free(&new_checkpoint);
        return 1;
    }
    if (remaining != 0) {
        sf37_tokens_free(&new_checkpoint);
        snapshot_set_err(err, errlen, "session payload has trailing bytes");
        return 1;
    }

    sf37_tokens_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    s->checkpoint_valid = true;
    return 0;
}

int sf37_session_save_snapshot(sf37_session *s, sf37_session_snapshot *snap,
                               char *err, size_t errlen) {
    if (!s || !snap || !s->checkpoint_valid) {
        snapshot_set_err(err, errlen, "session has no valid checkpoint to snapshot");
        return 1;
    }
    const uint64_t bytes = sf37_session_snapshot_bytes(s);
    if (bytes == 0 || bytes > (uint64_t)SIZE_MAX) {
        snapshot_set_err(err, errlen, "session snapshot is too large for this platform");
        return 1;
    }
    if (snap->cap < bytes) {
        uint8_t *p = realloc(snap->ptr, (size_t)bytes);
        if (!p) {
            snapshot_set_err(err, errlen, "out of memory while allocating session snapshot");
            return 1;
        }
        snap->ptr = p;
        snap->cap = bytes;
    }
    snap->len = 0;

#ifdef SF37_USE_CUDA
    if (session_uses_cuda(s) && !sf37_cuda_synchronize()) {
        snapshot_set_err(err, errlen, "failed to synchronize CUDA before snapshot");
        return 1;
    }
#endif

    sf37_snapshot_writer w = { .p = snap->ptr, .cap = snap->cap, .pos = 0 };
    const uint32_t token_count = (uint32_t)s->checkpoint.len;
    const uint32_t live_rows = session_live_kv_rows(s);
    uint32_t row_counts[SF37_MAIN_LAYERS];
    session_kv_row_counts(s, row_counts);
    const uint64_t kv_row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    const uint32_t backend_kind = session_uses_cuda(s) ? 1u : 0u;
    const uint32_t header[SF37_SESSION_SNAPSHOT_U32_FIELDS] = {
        SF37_SESSION_SNAPSHOT_MAGIC,
        SF37_SESSION_SNAPSHOT_VERSION,
        (uint32_t)s->ctx_size,
        token_count,
        live_rows,
        SF37_MAIN_LAYERS,
        SF37_KV_HEADS,
        SF37_HEAD_DIM,
        SF37_VOCAB,
        SF37_EMBD,
        backend_kind,
    };
    for (uint32_t i = 0; i < SF37_SESSION_SNAPSHOT_U32_FIELDS; i++) {
        if (snapshot_write_u32(&w, header[i]) != 0) goto overflow;
    }
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (snapshot_write_u32(&w, row_counts[il]) != 0) goto overflow;
    }
    for (int i = 0; i < s->checkpoint.len; i++) {
        if (snapshot_write_u32(&w, (uint32_t)s->checkpoint.v[i]) != 0) goto overflow;
    }
    if (snapshot_write_bytes(&w, s->logits, (uint64_t)SF37_VOCAB * sizeof(float)) != 0 ||
        snapshot_write_bytes(&w, s->hidden, (uint64_t)SF37_EMBD * sizeof(float)) != 0) {
        goto overflow;
    }

    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        uint8_t *row_buf = xmalloc((size_t)kv_row * sizeof(float));
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            const uint32_t rows = row_counts[il];
            const uint32_t first_pos = token_count - rows;
            const uint32_t cache_cap = s->cuda_state.cache_cap[il];
            for (uint32_t r = 0; r < rows; r++) {
                const uint32_t phys = (first_pos + r) % cache_cap;
                const uint64_t off = (uint64_t)phys * kv_row * sizeof(float);
                if (!sf37_cuda_tensor_read(s->cuda_state.k_cache[il], off, row_buf,
                                           kv_row * sizeof(float)) ||
                    snapshot_write_bytes(&w, row_buf, kv_row * sizeof(float)) != 0 ||
                    !sf37_cuda_tensor_read(s->cuda_state.v_cache[il], off, row_buf,
                                           kv_row * sizeof(float)) ||
                    snapshot_write_bytes(&w, row_buf, kv_row * sizeof(float)) != 0) {
                    free(row_buf);
                    snapshot_set_err(err, errlen, "failed to read CUDA KV cache for snapshot");
                    return 1;
                }
            }
        }
        free(row_buf);
#endif
    } else {
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            const sf37_layer_kv_cache *layer = &s->cpu_cache.layer[il];
            const uint32_t rows = row_counts[il];
            if (layer->n < rows) {
                snapshot_set_err(err, errlen, "CPU KV cache has fewer live rows than checkpoint");
                return 1;
            }
            const uint32_t start = layer->n - rows;
            if (snapshot_write_bytes(&w, layer->k + (uint64_t)start * kv_row,
                                     (uint64_t)rows * kv_row * sizeof(float)) != 0 ||
                snapshot_write_bytes(&w, layer->v + (uint64_t)start * kv_row,
                                     (uint64_t)rows * kv_row * sizeof(float)) != 0) {
                goto overflow;
            }
        }
    }
    snap->len = w.pos;
    return 0;

overflow:
    snapshot_set_err(err, errlen, "session snapshot writer overflow");
    return 1;
}

int sf37_session_load_snapshot(sf37_session *s, const sf37_session_snapshot *snap,
                               char *err, size_t errlen) {
    if (!s || !snap || !snap->ptr || snap->len == 0) {
        snapshot_set_err(err, errlen, "invalid session snapshot load");
        return 1;
    }
    sf37_snapshot_reader r = { .p = snap->ptr, .len = snap->len, .pos = 0 };
    uint32_t h[SF37_SESSION_SNAPSHOT_U32_FIELDS];
    for (uint32_t i = 0; i < SF37_SESSION_SNAPSHOT_U32_FIELDS; i++) {
        if (snapshot_read_u32(&r, &h[i]) != 0) {
            snapshot_set_err(err, errlen, "truncated session snapshot header");
            return 1;
        }
    }
    if (h[0] != SF37_SESSION_SNAPSHOT_MAGIC ||
        h[1] != SF37_SESSION_SNAPSHOT_VERSION) {
        snapshot_set_err(err, errlen, "unsupported session snapshot version");
        return 1;
    }
    uint32_t row_counts[SF37_MAIN_LAYERS];
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        if (snapshot_read_u32(&r, &row_counts[il]) != 0) {
            snapshot_set_err(err, errlen, "truncated session snapshot KV row counts");
            return 1;
        }
    }
    const uint32_t saved_ctx = h[2];
    const uint32_t token_count = h[3];
    const uint32_t live_rows = h[4];
    if (saved_ctx != (uint32_t)s->ctx_size) {
        snapshot_set_err(err, errlen, "session snapshot ctx %u does not match current ctx %d",
                         saved_ctx, s->ctx_size);
        return 1;
    }
    if (token_count > (uint32_t)s->ctx_size ||
        live_rows > token_count ||
        live_rows != (token_count < saved_ctx ? token_count : saved_ctx)) {
        snapshot_set_err(err, errlen, "session snapshot has invalid token/cache row counts");
        return 1;
    }
    for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
        const uint32_t rows = row_counts[il];
        if (rows > live_rows) {
            snapshot_set_err(err, errlen, "session snapshot has too many KV rows for layer %u", il);
            return 1;
        }
        if (layer_is_full(il)) {
            if (rows != live_rows) {
                snapshot_set_err(err, errlen, "session snapshot is missing full-attention KV rows");
                return 1;
            }
        } else {
            const uint32_t needed = live_rows < SF37_SLIDING_WINDOW ?
                                    live_rows : SF37_SLIDING_WINDOW;
            if (rows < needed) {
                snapshot_set_err(err, errlen, "session snapshot is missing sliding-window KV rows");
                return 1;
            }
        }
        const uint32_t cap = session_layer_kv_cap(s, il);
        if (cap != 0 && rows > cap) {
            snapshot_set_err(err, errlen, "session snapshot KV rows do not fit current cache");
            return 1;
        }
    }
    if (h[5] != SF37_MAIN_LAYERS ||
        h[6] != SF37_KV_HEADS ||
        h[7] != SF37_HEAD_DIM ||
        h[8] != SF37_VOCAB ||
        h[9] != SF37_EMBD) {
        snapshot_set_err(err, errlen, "session snapshot was written for a different SF37 layout");
        return 1;
    }

    sf37_tokens new_checkpoint = {0};
    for (uint32_t i = 0; i < token_count; i++) {
        uint32_t tok = 0;
        if (snapshot_read_u32(&r, &tok) != 0 || tok >= SF37_VOCAB) {
            sf37_tokens_free(&new_checkpoint);
            snapshot_set_err(err, errlen, "session snapshot has invalid token data");
            return 1;
        }
        sf37_tokens_push(&new_checkpoint, (int)tok);
    }
    if (snapshot_read_bytes(&r, s->logits, (uint64_t)SF37_VOCAB * sizeof(float)) != 0 ||
        snapshot_read_bytes(&r, s->hidden, (uint64_t)SF37_EMBD * sizeof(float)) != 0) {
        sf37_tokens_free(&new_checkpoint);
        snapshot_set_err(err, errlen, "truncated session snapshot logits/hidden");
        return 1;
    }

    const uint64_t kv_row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    s->checkpoint_valid = false;

    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        uint8_t *row_buf = xmalloc((size_t)kv_row * sizeof(float));
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            const uint32_t rows = row_counts[il];
            const uint32_t first_pos = token_count - rows;
            const uint32_t cache_cap = s->cuda_state.cache_cap[il];
            for (uint32_t rr = 0; rr < rows; rr++) {
                const uint32_t phys = (first_pos + rr) % cache_cap;
                const uint64_t off = (uint64_t)phys * kv_row * sizeof(float);
                if (snapshot_read_bytes(&r, row_buf, kv_row * sizeof(float)) != 0 ||
                    !sf37_cuda_tensor_write(s->cuda_state.k_cache[il], off, row_buf,
                                            kv_row * sizeof(float)) ||
                    snapshot_read_bytes(&r, row_buf, kv_row * sizeof(float)) != 0 ||
                    !sf37_cuda_tensor_write(s->cuda_state.v_cache[il], off, row_buf,
                                            kv_row * sizeof(float))) {
                    free(row_buf);
                    sf37_tokens_free(&new_checkpoint);
                    snapshot_set_err(err, errlen, "failed to restore CUDA KV cache from snapshot");
                    return 1;
                }
            }
        }
        free(row_buf);
        if (!sf37_cuda_tensor_write(s->cuda_state.hidden, 0, s->hidden,
                                    (uint64_t)SF37_EMBD * sizeof(float)) ||
            !sf37_cuda_tensor_write(s->cuda_state.logits, 0, s->logits,
                                    (uint64_t)SF37_VOCAB * sizeof(float))) {
            sf37_tokens_free(&new_checkpoint);
            snapshot_set_err(err, errlen, "failed to restore CUDA session scratch from snapshot");
            return 1;
        }
#endif
    } else {
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            sf37_layer_kv_cache *layer = &s->cpu_cache.layer[il];
            const uint32_t rows = row_counts[il];
            if (snapshot_read_bytes(&r, layer->k,
                                    (uint64_t)rows * kv_row * sizeof(float)) != 0 ||
                snapshot_read_bytes(&r, layer->v,
                                    (uint64_t)rows * kv_row * sizeof(float)) != 0) {
                sf37_tokens_free(&new_checkpoint);
                snapshot_set_err(err, errlen, "truncated CPU KV cache in session snapshot");
                return 1;
            }
            layer->n = rows;
        }
    }
    if (r.pos != r.len) {
        sf37_tokens_free(&new_checkpoint);
        snapshot_set_err(err, errlen, "session snapshot has trailing bytes");
        return 1;
    }

    sf37_tokens_free(&s->checkpoint);
    s->checkpoint = new_checkpoint;
    s->checkpoint_valid = true;
    return 0;
}

void sf37_session_snapshot_free(sf37_session_snapshot *snap) {
    if (!snap) return;
    free(snap->ptr);
    memset(snap, 0, sizeof(*snap));
}

static int sf37_session_eval_internal(sf37_session *s, int token, bool need_logits,
                                      char *err, size_t errlen) {
    if (!s || !s->engine) {
        session_set_err(err, errlen, "missing session");
        return 1;
    }
    if (token < 0 || token >= SF37_VOCAB) {
        session_set_err(err, errlen, "token id %d is outside vocabulary", token);
        return 1;
    }
    if (s->checkpoint.len >= s->ctx_size) {
        session_set_err(err, errlen, "context is full (%d tokens)", s->ctx_size);
        return 1;
    }
    sf37_engine *e = s->engine;
    const uint32_t pos = (uint32_t)s->checkpoint.len;

    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        if (!sf37_cuda_embed_token_bf16_mapped(s->cuda_state.hidden,
                                               e->gguf.map,
                                               e->gguf.size,
                                               e->embed_tokens->abs_offset,
                                               SF37_EMBD,
                                               SF37_VOCAB,
                                               (int32_t)token)) {
            session_set_err(err, errlen, "CUDA token embedding failed");
            s->checkpoint_valid = false;
            return 1;
        }
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            if (!cuda_decode_layer(e, &s->cuda_state, il, pos)) {
                session_set_err(err, errlen, "CUDA decode failed at layer %u", il);
                s->checkpoint_valid = false;
                return 1;
            }
        }
        if (need_logits) {
            if (!cuda_output_logits(e, &s->cuda_state) ||
                !sf37_cuda_tensor_read(s->cuda_state.logits, 0, s->logits,
                                       (uint64_t)SF37_VOCAB * sizeof(float))) {
                session_set_err(err, errlen, "CUDA output logits read failed");
                s->checkpoint_valid = false;
                return 1;
            }
        }
#endif
    } else {
        embed_token_bf16(e, token, s->hidden);
        for (uint32_t il = 0; il < SF37_MAIN_LAYERS; il++) {
            run_layer_decode(e, &s->cpu_cache, il, pos, s->hidden);
        }
        if (need_logits) output_logits_bf16(s->logits, e, s->hidden, NULL);
    }
    sf37_tokens_push(&s->checkpoint, token);
    s->checkpoint_valid = true;
    return 0;
}

int sf37_session_eval(sf37_session *s, int token, char *err, size_t errlen) {
    return sf37_session_eval_internal(s, token, true, err, errlen);
}

int sf37_session_eval_no_logits(sf37_session *s, int token, char *err, size_t errlen) {
    return sf37_session_eval_internal(s, token, false, err, errlen);
}

int sf37_session_eval_argmax(sf37_session *s, int token, char *err, size_t errlen) {
    if (sf37_session_eval_internal(s, token, false, err, errlen) != 0) return -1;
    const int next = sf37_session_output_argmax(s, err, errlen);
    if (next < 0) s->checkpoint_valid = false;
    return next;
}

int sf37_session_output_logits(sf37_session *s, char *err, size_t errlen) {
    if (!s || !s->engine) {
        session_set_err(err, errlen, "missing session");
        return 1;
    }
    sf37_engine *e = s->engine;
    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        if (!cuda_output_logits(e, &s->cuda_state) ||
            !sf37_cuda_tensor_read(s->cuda_state.logits, 0, s->logits,
                                   (uint64_t)SF37_VOCAB * sizeof(float))) {
            session_set_err(err, errlen, "CUDA output logits read failed");
            s->checkpoint_valid = false;
            return 1;
        }
#endif
    } else {
        output_logits_bf16(s->logits, e, s->hidden, NULL);
    }
    return 0;
}

int sf37_session_output_argmax(sf37_session *s, char *err, size_t errlen) {
    if (!s || !s->engine) {
        session_set_err(err, errlen, "missing session");
        return -1;
    }
    sf37_engine *e = s->engine;
    if (session_uses_cuda(s)) {
#ifdef SF37_USE_CUDA
        int tok = -1;
        if (cuda_output_argmax(e, &s->cuda_state, -1, &tok)) return tok;
        if (!cuda_output_logits(e, &s->cuda_state) ||
            !sf37_cuda_tensor_read(s->cuda_state.logits, 0, s->logits,
                                   (uint64_t)SF37_VOCAB * sizeof(float))) {
            session_set_err(err, errlen, "CUDA output argmax failed");
            s->checkpoint_valid = false;
            return -1;
        }
        return sample_argmax(s->logits, SF37_VOCAB);
#else
        (void)e;
#endif
    } else {
        output_logits_bf16(s->logits, e, s->hidden, NULL);
    }
    return sample_argmax(s->logits, SF37_VOCAB);
}

int sf37_session_common_prefix(sf37_session *s, const sf37_tokens *prompt) {
    if (!s || !prompt || !s->checkpoint_valid) return 0;
    int n = s->checkpoint.len < prompt->len ? s->checkpoint.len : prompt->len;
    int i = 0;
    while (i < n && s->checkpoint.v[i] == prompt->v[i]) i++;
    return i;
}

int sf37_session_sync_multimodal(sf37_session *s, const sf37_tokens *prompt,
                                 const sf37_image_features *image_features,
                                 char *err, size_t errlen) {
    if (!s || !prompt) {
        session_set_err(err, errlen, "invalid session sync");
        return 1;
    }
    if (prompt->len > s->ctx_size) {
        session_set_err(err, errlen, "prompt length %d exceeds context %d", prompt->len, s->ctx_size);
        return 1;
    }
    uint32_t image_rows = 0;
    if (image_features) {
        const bool has_pixels =
            (image_features->pixel_values && image_features->images > 0) ||
            (image_features->patch_pixel_values && image_features->patch_images > 0);
        if (!image_features_effective_rows(image_features, &image_rows, err, errlen)) return 1;
        if (!has_pixels && image_features->rows > 0 && !image_features->data) {
            session_set_err(err, errlen, "image features have rows but no data");
            return 1;
        }
        if (!has_pixels && image_features->rows > 0 &&
            image_features->dim != SF37_EMBD &&
            image_features->dim != SF37_VISION_PROJECTOR_IN) {
            session_set_err(err, errlen, "unsupported image feature dim %u", image_features->dim);
            return 1;
        }
        const uint32_t patches = prompt_count_im_patch(s->engine, prompt, 0, (uint32_t)prompt->len);
        if (patches != image_rows) {
            session_set_err(err, errlen,
                            "<im_patch> count %u does not match image feature rows %u",
                            patches, image_rows);
            return 1;
        }
    }
    int common = sf37_session_common_prefix(s, prompt);
    if (image_rows > 0) {
        session_reset(s);
        common = 0;
    } else if (!s->checkpoint_valid || common == 0 || common < s->checkpoint.len) {
        session_reset(s);
        common = 0;
    } else if (common < prompt->len) {
        sf37_session_rewind(s, common);
    }
#ifdef SF37_USE_CUDA
    if (session_uses_cuda(s)) {
        const int suffix = prompt->len - common;
        const uint32_t resume_min = cuda_resume_prefill_min_tokens();
        if (!cuda_batch_prefill_disabled() && suffix > 0 && (uint32_t)suffix >= resume_min) {
            const int rc = cuda_prefill_chunked_range(s, prompt, (uint32_t)common,
                                                      (uint32_t)suffix, image_features,
                                                      err, errlen);
            if (rc > 0) return 0;
            if (rc < 0 || image_features) return 1;
        }
        cuda_prefill_prepare_q8_cache(s->engine, s->progress, s->progress_ud,
                                      common, prompt->len);
    }
#endif
    if (image_features && image_rows > 0) {
        session_set_err(err, errlen, "multimodal session sync requires CUDA batch prefill");
        return 1;
    }
    for (int i = common; i < prompt->len; i++) {
        if (sf37_session_cancel_requested(s)) {
            session_set_err(err, errlen, "session cancelled");
            return 1;
        }
        const bool need_logits = (i + 1 == prompt->len);
        if (sf37_session_eval_internal(s, prompt->v[i], need_logits, err, errlen) != 0) return 1;
        sf37_session_report_progress(s, "prefill_chunk", i + 1, prompt->len);
        if (sf37_session_cancel_requested(s)) {
            session_set_err(err, errlen, "session cancelled");
            return 1;
        }
    }
    return 0;
}

int sf37_session_sync(sf37_session *s, const sf37_tokens *prompt, char *err, size_t errlen) {
    return sf37_session_sync_multimodal(s, prompt, NULL, err, errlen);
}

int sf37_session_argmax(sf37_session *s) {
    return s ? sample_argmax(s->logits, SF37_VOCAB) : -1;
}

int sf37_session_argmax_excluding(sf37_session *s, int excluded_id) {
    if (!s || !s->logits) return -1;
    int best = -1;
    float best_v = -INFINITY;
    for (uint32_t i = 0; i < SF37_VOCAB; i++) {
        if ((int)i == excluded_id) continue;
        const float v = s->logits[i];
        if (best < 0 || v > best_v) {
            best = (int)i;
            best_v = v;
        }
    }
    return best;
}

int sf37_sample_logits(const float *logits, int n_vocab, float temperature,
                       int top_k, float top_p, float min_p, uint64_t *rng) {
    if (!logits || n_vocab <= 0) return -1;
    return sample_top_p_min_p(logits, (uint32_t)n_vocab, temperature,
                              top_k, top_p, min_p, rng);
}

int sf37_session_sample(sf37_session *s, float temperature, int top_k,
                        float top_p, float min_p, uint64_t *rng) {
    if (!s || !s->logits) return -1;
    return sample_top_p_min_p(s->logits, SF37_VOCAB, temperature,
                              top_k, top_p, min_p, rng);
}

int sf37_session_top_logprobs(sf37_session *s, sf37_token_score *out, int k) {
    if (!s || !out || k <= 0) return 0;
    if (k > SF37_VOCAB) k = SF37_VOCAB;
    for (int i = 0; i < k; i++) {
        out[i].id = -1;
        out[i].logit = -INFINITY;
        out[i].logprob = -INFINITY;
    }

    float max_logit = -INFINITY;
    for (uint32_t i = 0; i < SF37_VOCAB; i++) {
        const float v = s->logits[i];
        if (!isfinite(v)) continue;
        if (v > max_logit) max_logit = v;
        for (int j = 0; j < k; j++) {
            if (out[j].id < 0 || v > out[j].logit) {
                for (int l = k - 1; l > j; l--) out[l] = out[l - 1];
                out[j].id = (int)i;
                out[j].logit = v;
                break;
            }
        }
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < SF37_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    for (int i = 0; i < k && out[i].id >= 0; i++) {
        out[i].logprob = isfinite(out[i].logit) ?
            (float)((double)out[i].logit - logsum) : -INFINITY;
    }
    return k;
}

int sf37_session_token_logprob(sf37_session *s, int token, sf37_token_score *out) {
    if (!s || !out || token < 0 || token >= SF37_VOCAB) return 0;
    float max_logit = -INFINITY;
    for (uint32_t i = 0; i < SF37_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v) && v > max_logit) max_logit = v;
    }
    if (!isfinite(max_logit)) return 0;

    double sum = 0.0;
    for (uint32_t i = 0; i < SF37_VOCAB; i++) {
        const float v = s->logits[i];
        if (isfinite(v)) sum += exp((double)v - (double)max_logit);
    }
    const double logsum = (double)max_logit + log(sum);
    out->id = token;
    out->logit = s->logits[token];
    out->logprob = isfinite(out->logit) ?
        (float)((double)out->logit - logsum) : -INFINITY;
    return 1;
}

int sf37_session_copy_logits(sf37_session *s, float *out, int cap) {
    if (!s || !out || cap < SF37_VOCAB) return 0;
    memcpy(out, s->logits, (size_t)SF37_VOCAB * sizeof(out[0]));
    return SF37_VOCAB;
}

int sf37_session_pos(sf37_session *s) {
    return s ? s->checkpoint.len : 0;
}

int sf37_session_ctx(sf37_session *s) {
    return s ? s->ctx_size : 0;
}

uint64_t sf37_session_kv_bytes(sf37_session *s) {
    if (!s) return 0;
#ifdef SF37_USE_CUDA
    if (session_uses_cuda(s) && s->cuda_state_ready && s->cuda_state.kv_cache_bytes != 0) {
        return s->cuda_state.kv_cache_bytes;
    }
#endif
    const uint64_t row = (uint64_t)SF37_KV_HEADS * SF37_HEAD_DIM;
    return (uint64_t)SF37_MAIN_LAYERS * (uint64_t)s->ctx_size * row * 2u * sizeof(float);
}

const sf37_tokens *sf37_session_tokens(sf37_session *s) {
    return s ? &s->checkpoint : NULL;
}

const char *sf37_backend_name(sf37_backend backend) {
    switch (backend) {
    case SF37_BACKEND_CPU: return "cpu";
    case SF37_BACKEND_CUDA: return "cuda";
    default: return "unknown";
    }
}

void sf37_log(FILE *fp, sf37_log_type type, const char *fmt, ...) {
    const char *prefix = "sf37";
    switch (type) {
    case SF37_LOG_OK: prefix = "sf37: ok"; break;
    case SF37_LOG_WARNING: prefix = "sf37: warning"; break;
    case SF37_LOG_ERROR: prefix = "sf37: error"; break;
    default: prefix = "sf37"; break;
    }
    fprintf(fp ? fp : stderr, "%s: ", prefix);
    va_list ap;
    va_start(ap, fmt);
    vfprintf(fp ? fp : stderr, fmt, ap);
    va_end(ap);
    fprintf(fp ? fp : stderr, "\n");
}

int sf37_engine_open(sf37_engine **out, const sf37_engine_options *opt) {
    if (!out || !opt || !opt->model_path) return -1;
    *out = NULL;
    sf37_engine *e = xcalloc(1, sizeof(*e));
    e->gguf.fd = -1;
    e->backend = opt->backend;
    e->model_path = xstrdup(opt->model_path);
    e->tokenizer_path = opt->tokenizer_path && opt->tokenizer_path[0] ?
        xstrdup(opt->tokenizer_path) : path_dirname_dup(opt->model_path);

#ifdef SF37_USE_CUDA
    if (e->backend == SF37_BACKEND_CUDA) {
        if (!sf37_cuda_init()) {
            sf37_engine_close(e);
            return -1;
        }
        e->cuda_ready = true;
    }
#else
    if (e->backend == SF37_BACKEND_CUDA) {
        sf37_log(stderr, SF37_LOG_WARNING,
                 "CUDA backend requested from a CPU-only build; using CPU reference code");
    }
#endif

    char err[512] = {0};
    const double map_t0 = sf37_now_sec();
    if (gguf_open(&e->gguf, opt->model_path, err, sizeof(err)) != 0) {
        sf37_log(stderr, SF37_LOG_ERROR, "%s", err);
        sf37_engine_close(e);
        return -1;
    }
    if (bind_model(e, err, sizeof(err)) != 0) {
        sf37_log(stderr, SF37_LOG_ERROR, "%s", err);
        sf37_engine_close(e);
        return -1;
    }
    const double map_t1 = sf37_now_sec();
    e->timing_model_map_sec = map_t1 - map_t0;
#ifdef SF37_USE_CUDA
    if (e->backend == SF37_BACKEND_CUDA) {
        const double preload_t0 = sf37_now_sec();
        if (!sf37_cuda_prepare_model(e)) {
            e->timing_preload_sec = sf37_now_sec() - preload_t0;
            sf37_log(stderr, SF37_LOG_ERROR,
                     "CUDA failed to prepare model mapping/cache; aborting startup");
            sf37_engine_close(e);
            return -1;
        }
        e->timing_preload_sec = sf37_now_sec() - preload_t0;
    }
#endif

    if (e->tokenizer_path) {
        char tok[4096];
        snprintf(tok, sizeof(tok), "%s/tokenizer.json", e->tokenizer_path);
        if (path_exists(tok)) {
            char tok_err[512] = {0};
            if (vocab_load_from_tokenizer_json(&e->vocab, tok, tok_err, sizeof(tok_err)) != 0) {
                sf37_log(stderr, SF37_LOG_ERROR, "failed to load tokenizer.json: %s", tok_err);
                sf37_engine_close(e);
                return -1;
            }
        } else {
            sf37_log(stderr, SF37_LOG_WARNING,
                     "tokenizer.json not found under tokenizer path: %s",
                     e->tokenizer_path);
        }
        snprintf(tok, sizeof(tok), "%s/tokenizer_config.json", e->tokenizer_path);
        if (!path_exists(tok)) {
            sf37_log(stderr, SF37_LOG_WARNING,
                     "tokenizer_config.json not found under tokenizer path: %s",
                     e->tokenizer_path);
        }
    } else {
        sf37_log(stderr, SF37_LOG_WARNING,
                 "tokenizer path is unavailable; tokenizer/session prompt helpers are disabled");
    }

    *out = e;
    return 0;
}

void sf37_engine_close(sf37_engine *e) {
    if (!e) return;
#ifdef SF37_USE_CUDA
    if (e->cuda_ready) sf37_cuda_cleanup();
#endif
    vocab_free(&e->vocab);
    gguf_close(&e->gguf);
    free(e->model_path);
    free(e->tokenizer_path);
    free(e);
}

void sf37_engine_timing(sf37_engine *e, sf37_engine_timing_info *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!e) return;
    out->model_map_sec = e->timing_model_map_sec;
    out->preload_sec = e->timing_preload_sec;
}

static void print_kv_string(FILE *fp, const sf37_gguf *g, const char *key) {
    sf37_str s;
    if (kv_string(g, key, &s)) {
        fprintf(fp, "%s: %.*s\n", key, (int)s.len, s.ptr);
    }
}

void sf37_engine_summary(sf37_engine *e, FILE *fp) {
    if (!fp) fp = stdout;
    const sf37_gguf *g = &e->gguf;
    fprintf(fp, "sf37 inspect\n");
    fprintf(fp, "model: %s\n", e->model_path);
    if (e->tokenizer_path) fprintf(fp, "tokenizer path: %s\n", e->tokenizer_path);
    fprintf(fp, "backend: %s\n", sf37_backend_name(e->backend));
    fprintf(fp, "gguf: v%u, %" PRIu64 " metadata keys, %" PRIu64 " tensors, alignment=%u\n",
            g->version, g->n_kv, g->n_tensors, g->alignment);
    print_kv_string(fp, g, "general.architecture");
    print_kv_string(fp, g, "general.name");
    print_kv_string(fp, g, "sf37.q3_layout");
    print_kv_string(fp, g, "sf37.q2_layout");
    print_kv_string(fp, g, "sf37.quant_policy");
    print_kv_string(fp, g, "sf37.keep_bf16");

    fprintf(fp, "text: layers=%u, hidden=%u, vocab=%u, ctx=%u, moe_layers=3..44, experts=%u, topk=%u\n",
            SF37_MAIN_LAYERS, SF37_EMBD, SF37_VOCAB, SF37_CTX, SF37_EXPERTS, SF37_EXPERT_USED);
    fprintf(fp, "tokenizer: %s%s\n",
            e->vocab.ready ? "tokenizer.json loaded" : "not loaded",
            e->vocab.ready && e->vocab.n_vocab != SF37_VOCAB ? " (vocab size differs from model head)" : "");
    fprintf(fp, "attention: full_heads=%u, sliding_heads=%u, kv_heads=%u, head_dim=%u, sliding_window=512\n",
            SF37_FULL_Q_HEADS, SF37_SLIDING_Q_HEADS, SF37_KV_HEADS, SF37_HEAD_DIM);
    fprintf(fp, "vision: image=%u, patch=%u, grid=%ux%u, width=%u, layers=%u, features=%u, patch_features=%u\n",
            SF37_VISION_IMAGE, SF37_VISION_PATCH, SF37_VISION_GRID, SF37_VISION_GRID,
            SF37_VISION_WIDTH, SF37_VISION_LAYERS, SF37_VISION_FEATURES,
            SF37_VISION_PATCH_FEATURES);
    fprintf(fp, "bound: text_tensors=%u, vision_tensors=%u, mtp_tensors_inactive=%u\n",
            e->bound_text_tensors, e->bound_vision_tensors, e->inactive_mtp_tensors);

    fprintf(fp, "tensor types:\n");
    for (uint32_t type = 0; type < 42; type++) {
        if (e->type_count[type]) {
            fprintf(fp, "  %-6s type=%2u tensors=%4" PRIu64 " bytes=%10.3f GiB\n",
                    type_name(type), type, e->type_count[type],
                    (double)e->type_bytes[type] / 1073741824.0);
        }
    }
}
