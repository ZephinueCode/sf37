/*
 * SF37 Step-3.7-Flash HF safetensors -> GGUF quantizer.
 *
 * This file reuses the small JSON, safetensors, GGUF, and conversion helpers
 * from the ds4 quantizer in the same directory, but writes SF37 metadata and
 * applies a Step-3.7-specific tensor policy.
 */

#define main ds4_deepseek4_quantize_main_disabled
#include "deepseek4-quantize.c"
#undef main

typedef struct {
    tensor_meta meta;
    char *hf_name;
    char *dtype;
    int src_n_dims;
    int64_t src_shape[MAX_DIMS];
} sf37_tensor;

typedef struct {
    sf37_tensor *v;
    uint64_t len;
    uint64_t cap;
    size_t tensor_bytes;
    size_t meta_size;
    size_t data_offset;
    uint32_t alignment;
} sf37_plan;

typedef struct {
    const char *hf_dir;
    const char *out_path;
    const char *imatrix_file;
    bool dry_run;
    bool overwrite;
    bool imatrix_strict;
    int64_t limit;
    int n_threads;
} sf37_args;

typedef struct {
    ds4q_type type;
    const float *src;
    void *dst;
    int64_t ncols;
    int64_t row0;
    int64_t nrows;
    int64_t expert_rows;
    int64_t n_experts;
    const float *imatrix;
} sf37_quant_job;

static bool sf37_contains(const char *s, const char *needle) {
    return strstr(s, needle) != NULL;
}

static bool sf37_is_routed_gate(const char *name) {
    return sf37_contains(name, ".moe.gate_proj.weight");
}

static bool sf37_is_routed_up(const char *name) {
    return sf37_contains(name, ".moe.up_proj.weight");
}

static bool sf37_is_routed_down(const char *name) {
    return sf37_contains(name, ".moe.down_proj.weight");
}

static bool sf37_is_router(const char *name) {
    return sf37_contains(name, ".moe.gate.weight") ||
           sf37_contains(name, ".moe.router_bias");
}

static bool sf37_keep_bf16(const char *name) {
    return strcmp(name, "lm_head.weight") == 0 ||
           strcmp(name, "model.embed_tokens.weight") == 0 ||
           strcmp(name, "vit_large_projector.weight") == 0;
}

static ds4q_type sf37_source_type(const char *dtype) {
    if (strcmp(dtype, "BF16") == 0) return DS4Q_TYPE_BF16;
    if (strcmp(dtype, "F32") == 0) return DS4Q_TYPE_F32;
    if (strcmp(dtype, "F16") == 0) return DS4Q_TYPE_F16;
    fprintf(stderr, "error: unsupported source dtype: %s\n", dtype);
    exit(1);
}

static bool sf37_is_quant_type(ds4q_type type) {
    return type == DS4Q_TYPE_Q8_0 ||
           type == DS4Q_TYPE_Q2_K ||
           type == DS4Q_TYPE_Q3_K ||
           type == DS4Q_TYPE_Q4_K ||
           type == DS4Q_TYPE_IQ2_XXS;
}

static ds4q_type sf37_policy_type(const char *name, const st_info *info) {
    ds4q_type src = sf37_source_type(info->dtype);
    if (src == DS4Q_TYPE_F32) return DS4Q_TYPE_F32;
    if (sf37_keep_bf16(name)) return DS4Q_TYPE_BF16;
    if (info->n_dims < 2) return src;

    /*
     * Keep true convolution kernels and other rank-4 tensors BF16 for the first
     * multimodal quant. They are small enough, and preserving their shape keeps
     * the future vision runtime straightforward.
     */
    if (info->n_dims > 3) return src;

    if (sf37_is_router(name)) return src;

    int64_t ncols = info->shape[info->n_dims - 1];
    if (sf37_is_routed_gate(name) || sf37_is_routed_up(name)) {
        return (ncols % ds4q_block_size(DS4Q_TYPE_Q3_K)) == 0 ? DS4Q_TYPE_Q3_K : src;
    }
    if (sf37_is_routed_down(name)) {
        return (ncols % ds4q_block_size(DS4Q_TYPE_Q2_K)) == 0 ? DS4Q_TYPE_Q2_K : src;
    }

    return (ncols % ds4q_block_size(DS4Q_TYPE_Q8_0)) == 0 ? DS4Q_TYPE_Q8_0 : src;
}

static void sf37_plan_push(sf37_plan *p, sf37_tensor t) {
    if (p->len == p->cap) {
        p->cap = p->cap ? p->cap * 2 : 2048;
        p->v = xrealloc(p->v, (size_t)p->cap * sizeof(p->v[0]));
    }
    p->v[p->len++] = t;
}

static size_t sf37_tensor_info_size(const tensor_meta *t) {
    return gguf_string_size(t->name) + 4 + (size_t)t->n_dims * 8 + 4 + 8;
}

static size_t sf37_kv_size_string(const char *key, const char *value) {
    return gguf_string_size(key) + 4 + gguf_string_size(value);
}

static size_t sf37_kv_size_u32(const char *key) {
    return gguf_string_size(key) + 4 + 4;
}

static size_t sf37_kv_size_bool(const char *key) {
    return gguf_string_size(key) + 4 + 1;
}

static void sf37_write_kv_string(FILE *fp, const char *key, const char *value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_STRING);
    write_gguf_string(fp, value);
}

static void sf37_write_kv_u32(FILE *fp, const char *key, uint32_t value) {
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_UINT32);
    write_u32(fp, value);
}

static void sf37_write_kv_bool(FILE *fp, const char *key, bool value) {
    uint8_t byte = value ? 1 : 0;
    write_gguf_string(fp, key);
    write_u32(fp, GGUF_TYPE_BOOL);
    if (fwrite(&byte, 1, 1, fp) != 1) die("write bool failed");
}

static uint64_t sf37_kv_count(const imatrix_store *imatrix) {
    return 20 + extra_imatrix_kv_count(imatrix);
}

static size_t sf37_kv_bytes(const char *hf_dir, const imatrix_store *imatrix) {
    size_t n = 0;
    n += sf37_kv_size_string("general.architecture", "sf37");
    n += sf37_kv_size_string("general.name", "Super Fast Step 3.7 Flash");
    n += sf37_kv_size_u32("general.alignment");
    n += sf37_kv_size_u32("general.file_type");
    n += sf37_kv_size_u32("sf37.version");
    n += sf37_kv_size_string("sf37.source", hf_dir);
    n += sf37_kv_size_u32("sf37.block_count");
    n += sf37_kv_size_u32("sf37.mtp_layer_count");
    n += sf37_kv_size_u32("sf37.embedding_length");
    n += sf37_kv_size_u32("sf37.feed_forward_length");
    n += sf37_kv_size_u32("sf37.expert_feed_forward_length");
    n += sf37_kv_size_u32("sf37.expert_count");
    n += sf37_kv_size_u32("sf37.expert_used_count");
    n += sf37_kv_size_u32("sf37.context_length");
    n += sf37_kv_size_bool("sf37.vision_included");
    n += sf37_kv_size_string("sf37.q3_layout", "q3_asym_g64_f16scale_u8zp_payload3_pad110");
    n += sf37_kv_size_string("sf37.q2_layout", "q2_asym_g64_f16scale_u8zp_payload2_pad84");
    n += sf37_kv_size_string("sf37.quant_policy", "routed_gate=q3_asym,routed_up=q3_asym,routed_down=q2_asym,rank2_rest=q8_0,router_norm_misc=source");
    n += sf37_kv_size_string("sf37.keep_bf16", "lm_head.weight,model.embed_tokens.weight,vit_large_projector.weight");
    n += sf37_kv_size_string("sf37.tensor_name_format", "hf_safetensors_names_with_reversed_gguf_dimensions");
    n += extra_imatrix_kv_size(imatrix);
    return n;
}

static void sf37_write_kvs(FILE *fp, const char *hf_dir, const imatrix_store *imatrix) {
    sf37_write_kv_string(fp, "general.architecture", "sf37");
    sf37_write_kv_string(fp, "general.name", "Super Fast Step 3.7 Flash");
    sf37_write_kv_u32(fp, "general.alignment", DS4_GGUF_DEFAULT_ALIGNMENT);
    sf37_write_kv_u32(fp, "general.file_type", 0);
    sf37_write_kv_u32(fp, "sf37.version", 1);
    sf37_write_kv_string(fp, "sf37.source", hf_dir);
    sf37_write_kv_u32(fp, "sf37.block_count", 45);
    sf37_write_kv_u32(fp, "sf37.mtp_layer_count", 3);
    sf37_write_kv_u32(fp, "sf37.embedding_length", 4096);
    sf37_write_kv_u32(fp, "sf37.feed_forward_length", 11264);
    sf37_write_kv_u32(fp, "sf37.expert_feed_forward_length", 1280);
    sf37_write_kv_u32(fp, "sf37.expert_count", 288);
    sf37_write_kv_u32(fp, "sf37.expert_used_count", 8);
    sf37_write_kv_u32(fp, "sf37.context_length", 262144);
    sf37_write_kv_bool(fp, "sf37.vision_included", true);
    sf37_write_kv_string(fp, "sf37.q3_layout", "q3_asym_g64_f16scale_u8zp_payload3_pad110");
    sf37_write_kv_string(fp, "sf37.q2_layout", "q2_asym_g64_f16scale_u8zp_payload2_pad84");
    sf37_write_kv_string(fp, "sf37.quant_policy", "routed_gate=q3_asym,routed_up=q3_asym,routed_down=q2_asym,rank2_rest=q8_0,router_norm_misc=source");
    sf37_write_kv_string(fp, "sf37.keep_bf16", "lm_head.weight,model.embed_tokens.weight,vit_large_projector.weight");
    sf37_write_kv_string(fp, "sf37.tensor_name_format", "hf_safetensors_names_with_reversed_gguf_dimensions");
    write_imatrix_kvs(fp, imatrix);
}

static void sf37_build_plan(st_db *db, sf37_plan *plan, int64_t limit,
                            const char *hf_dir, const imatrix_store *imatrix) {
    memset(plan, 0, sizeof(*plan));
    plan->alignment = DS4_GGUF_DEFAULT_ALIGNMENT;

    const int64_t n = limit > 0 && limit < db->n_weights ? limit : db->n_weights;
    size_t off = 0;
    size_t tensor_info = 0;
    for (int64_t i = 0; i < n; i++) {
        const char *name = db->weights[i].name;
        tensor_entry *te = db_tensor(db, name, NULL);
        ds4q_type type = sf37_policy_type(name, &te->info);

        sf37_tensor st = {0};
        st.hf_name = xstrdup(name);
        st.dtype = xstrdup(te->info.dtype);
        st.src_n_dims = te->info.n_dims;
        memcpy(st.src_shape, te->info.shape, sizeof(st.src_shape));
        st.meta.name = xstrdup(name);
        st.meta.n_dims = te->info.n_dims;
        for (int d = 0; d < te->info.n_dims; d++) {
            st.meta.ne[d] = te->info.shape[te->info.n_dims - 1 - d];
        }
        st.meta.type = type;

        if (sf37_is_quant_type(type) && st.meta.ne[0] % ds4q_block_size(type) != 0) {
            fprintf(stderr, "warning: %s ncols=%" PRId64 " is not divisible by %ld, keeping source dtype\n",
                    name, st.meta.ne[0], (long)ds4q_block_size(type));
            type = sf37_source_type(te->info.dtype);
            st.meta.type = type;
        }

        st.meta.size = tensor_nbytes(type, st.meta.ne, st.meta.n_dims);
        st.meta.new_offset = off;
        off += ds4q_pad(st.meta.size, plan->alignment);
        tensor_info += sf37_tensor_info_size(&st.meta);
        sf37_plan_push(plan, st);
    }

    plan->tensor_bytes = off;
    plan->meta_size = 4 + 4 + 8 + 8 + sf37_kv_bytes(hf_dir, imatrix) + tensor_info;
    plan->data_offset = ds4q_pad(plan->meta_size, plan->alignment);
}

static void sf37_free_plan(sf37_plan *plan) {
    for (uint64_t i = 0; i < plan->len; i++) {
        free(plan->v[i].meta.name);
        free(plan->v[i].hf_name);
        free(plan->v[i].dtype);
    }
    free(plan->v);
    memset(plan, 0, sizeof(*plan));
}

static void *sf37_quant_worker(void *arg) {
    sf37_quant_job *job = arg;
    if (job->imatrix && job->expert_rows > 0 && job->n_experts > 0) {
        int64_t row = job->row0;
        const int64_t end = job->row0 + job->nrows;
        while (row < end) {
            const int64_t expert = row / job->expert_rows;
            if (expert < 0 || expert >= job->n_experts) break;
            const int64_t expert_end = (expert + 1) * job->expert_rows;
            const int64_t rows = (end < expert_end ? end : expert_end) - row;
            const float *imat = job->imatrix + (size_t)expert * (size_t)job->ncols;
            ds4q_quantize_chunk(job->type, job->src, job->dst,
                                row * job->ncols, rows, job->ncols, imat);
            row += rows;
        }
    } else {
        ds4q_quantize_chunk(job->type, job->src, job->dst,
                            job->row0 * job->ncols, job->nrows, job->ncols,
                            job->imatrix);
    }
    return NULL;
}

static byte_buf sf37_f32_to_type_parallel(const float *src, int64_t n, ds4q_type type,
                                          int64_t ncols, int n_threads,
                                          const float *imatrix,
                                          int64_t expert_rows,
                                          int64_t n_experts) {
    if (!sf37_is_quant_type(type)) return f32_to_type(src, n, type, ncols, NULL);
    if (n_threads <= 1 && (!imatrix || expert_rows <= 0)) return f32_to_type(src, n, type, ncols, imatrix);
    if (ncols <= 0 || n % ncols != 0) die("bad ncols for tensor conversion");
    if (ncols % ds4q_block_size(type) != 0) die("ncols is not divisible by quant block size");

    const int64_t nrows = n / ncols;
    if (nrows <= 1 && (!imatrix || expert_rows <= 0)) return f32_to_type(src, n, type, ncols, imatrix);
    if (expert_rows > 0 && n_experts > 0 && nrows != expert_rows * n_experts) {
        die("expert imatrix row layout mismatch");
    }
    int workers = n_threads;
    if (workers < 1) workers = 1;
    if ((int64_t)workers > nrows) workers = (int)nrows;

    byte_buf out = {0};
    out.size = (size_t)nrows * ds4q_row_size(type, ncols);
    out.data = xmalloc(out.size);
    ds4q_quantize_init(type);

    sf37_quant_job *jobs = xcalloc((size_t)workers, sizeof(jobs[0]));
    pthread_t *threads = xcalloc((size_t)workers, sizeof(threads[0]));
    int64_t base = 0;
    for (int i = 0; i < workers; i++) {
        int64_t rows = nrows / workers + (i < (nrows % workers) ? 1 : 0);
        jobs[i] = (sf37_quant_job){
            .type = type,
            .src = src,
            .dst = out.data,
            .ncols = ncols,
            .row0 = base,
            .nrows = rows,
            .expert_rows = expert_rows,
            .n_experts = n_experts,
            .imatrix = imatrix,
        };
        base += rows;
    }
    for (int i = 1; i < workers; i++) pthread_create(&threads[i], NULL, sf37_quant_worker, &jobs[i]);
    sf37_quant_worker(&jobs[0]);
    for (int i = 1; i < workers; i++) pthread_join(threads[i], NULL);
    free(threads);
    free(jobs);
    return out;
}

static bool sf37_is_routed_tensor(const char *name) {
    return sf37_is_routed_gate(name) || sf37_is_routed_up(name) || sf37_is_routed_down(name);
}

static byte_buf sf37_generate_tensor(st_db *db, const sf37_tensor *t, int n_threads,
                                     const imatrix_store *imatrix) {
    st_value sv = db_read(db, t->hf_name);
    ds4q_type src_type = sf37_source_type(sv.dtype);
    ds4q_type dst_type = t->meta.type;

    if (dst_type == src_type) {
        byte_buf out = { .data = sv.data, .size = sv.nbytes };
        sv.data = NULL;
        st_value_free(&sv);
        return out;
    }

    int64_t n = 0;
    float *f32 = tensor_to_f32(&sv, &n);
    st_value_free(&sv);
    const float *imat = NULL;
    int64_t expert_rows = 0;
    int64_t n_experts = 0;
    if (imatrix_enabled(imatrix) && sf37_is_routed_tensor(t->hf_name) &&
        t->meta.n_dims == 3 && sf37_is_quant_type(dst_type)) {
        const char *names[1] = { t->hf_name };
        n_experts = t->meta.ne[2];
        expert_rows = t->meta.ne[1];
        imat = imatrix_find(imatrix, names, 1,
                            t->meta.ne[0] * n_experts,
                            -1, 0);
        if (imat) {
            fprintf(stderr, "  imatrix: %s experts=%" PRId64 " cols=%" PRId64 "\n",
                    t->hf_name, n_experts, t->meta.ne[0]);
        }
    }
    byte_buf out = sf37_f32_to_type_parallel(f32, n, dst_type, t->meta.ne[0],
                                             n_threads, imat,
                                             expert_rows, n_experts);
    free(f32);
    return out;
}

static void sf37_print_plan(const sf37_plan *plan) {
    size_t by_type[DS4Q_TYPE_COUNT] = {0};
    uint64_t count_by_type[DS4Q_TYPE_COUNT] = {0};
    for (uint64_t i = 0; i < plan->len; i++) {
        ds4q_type type = plan->v[i].meta.type;
        if (type >= 0 && type < DS4Q_TYPE_COUNT) {
            by_type[type] += plan->v[i].meta.size;
            count_by_type[type]++;
        }
    }
    fprintf(stderr, "SF37 plan: tensors=%" PRIu64 " data=%.3f GiB meta=%.3f MiB total=%.3f GiB\n",
            plan->len,
            (double)plan->tensor_bytes / 1073741824.0,
            (double)plan->data_offset / 1048576.0,
            (double)(plan->data_offset + plan->tensor_bytes) / 1073741824.0);
    for (int i = 0; i < DS4Q_TYPE_COUNT; i++) {
        if (count_by_type[i]) {
            fprintf(stderr, "  %-10s tensors=%4" PRIu64 " size=%9.3f GiB\n",
                    ds4q_type_name((ds4q_type)i), count_by_type[i],
                    (double)by_type[i] / 1073741824.0);
        }
    }
}

static void sf37_write_gguf(st_db *db, const sf37_plan *plan,
                            const sf37_args *args,
                            const imatrix_store *imatrix) {
    FILE *fp = fopen(args->out_path, "wb");
    if (!fp) die_errno("open output", args->out_path);
    if (fwrite("GGUF", 1, 4, fp) != 4) die("write GGUF magic failed");
    write_u32(fp, 3);
    write_u64(fp, plan->len);
    write_u64(fp, sf37_kv_count(imatrix));
    sf37_write_kvs(fp, args->hf_dir, imatrix);

    for (uint64_t i = 0; i < plan->len; i++) {
        const tensor_meta *t = &plan->v[i].meta;
        write_gguf_string(fp, t->name);
        write_u32(fp, (uint32_t)t->n_dims);
        for (int d = 0; d < t->n_dims; d++) write_u64(fp, (uint64_t)t->ne[d]);
        write_u32(fp, (uint32_t)t->type);
        write_u64(fp, t->new_offset);
    }

    long pos = ftell(fp);
    if (pos < 0) die("ftell failed");
    if ((size_t)pos > plan->data_offset) die("GGUF metadata larger than planned");
    write_padding(fp, plan->data_offset - (size_t)pos);

    for (uint64_t i = 0; i < plan->len; i++) {
        const sf37_tensor *t = &plan->v[i];
        fprintf(stderr, "[%4" PRIu64 "/%4" PRIu64 "] %s -> %s\n",
                i + 1, plan->len, t->hf_name, ds4q_type_name(t->meta.type));
        byte_buf data = sf37_generate_tensor(db, t, args->n_threads, imatrix);
        if (data.size != t->meta.size) {
            fprintf(stderr, "error: generated size mismatch for %s: got %zu expected %zu\n",
                    t->hf_name, data.size, t->meta.size);
            exit(1);
        }
        if (fwrite(data.data, 1, data.size, fp) != data.size) die_errno("write tensor", args->out_path);
        size_t padded = ds4q_pad(data.size, plan->alignment);
        write_padding(fp, padded - data.size);
        free(data.data);
    }
    fclose(fp);
}

static const char *sf37_need_value(int argc, char **argv, int *i, const char *arg) {
    if (*i + 1 >= argc) {
        fprintf(stderr, "error: %s requires a value\n", arg);
        exit(1);
    }
    return argv[++*i];
}

static void sf37_usage(const char *argv0) {
    printf("usage: %s --hf DIR --out OUT.gguf [--imatrix FILE] [--imatrix-strict] [--dry-run] [--overwrite] [--limit N] [--threads N]\n", argv0);
}

static sf37_args sf37_parse_args(int argc, char **argv) {
    sf37_args a = { .n_threads = 8 };
    for (int i = 1; i < argc; i++) {
        const char *arg = argv[i];
        if (strcmp(arg, "--hf") == 0) {
            a.hf_dir = sf37_need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--out") == 0) {
            a.out_path = sf37_need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix") == 0) {
            a.imatrix_file = sf37_need_value(argc, argv, &i, arg);
        } else if (strcmp(arg, "--imatrix-strict") == 0) {
            a.imatrix_strict = true;
        } else if (strcmp(arg, "--dry-run") == 0) {
            a.dry_run = true;
        } else if (strcmp(arg, "--overwrite") == 0) {
            a.overwrite = true;
        } else if (strcmp(arg, "--limit") == 0) {
            a.limit = atoll(sf37_need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "--threads") == 0) {
            a.n_threads = atoi(sf37_need_value(argc, argv, &i, arg));
        } else if (strcmp(arg, "-h") == 0 || strcmp(arg, "--help") == 0) {
            sf37_usage(argv[0]);
            exit(0);
        } else {
            fprintf(stderr, "error: unknown argument: %s\n", arg);
            exit(1);
        }
    }
    if (!a.hf_dir) die("--hf is required");
    if (!a.out_path && !a.dry_run) die("--out is required unless --dry-run is used");
    if (a.out_path && file_exists(a.out_path) && !a.overwrite) die("output exists; use --overwrite");
    if (a.n_threads < 1) a.n_threads = 1;
    return a;
}

int main(int argc, char **argv) {
    sf37_args args = sf37_parse_args(argc, argv);
    imatrix_store imatrix = {0};
    if (args.imatrix_file) imatrix_load(&imatrix, args.imatrix_file, args.imatrix_strict);
    st_db db = {0};
    db_open(&db, args.hf_dir);
    sf37_plan plan = {0};
    sf37_build_plan(&db, &plan, args.limit, args.hf_dir, &imatrix);
    sf37_print_plan(&plan);
    if (!args.dry_run) {
        sf37_write_gguf(&db, &plan, &args, &imatrix);
        fprintf(stderr, "wrote %s\n", args.out_path);
    }
    sf37_free_plan(&plan);
    db_close(&db);
    imatrix_free(&imatrix);
    return 0;
}
