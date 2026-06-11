#ifndef SF37_H
#define SF37_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef enum {
    SF37_BACKEND_CPU = 0,
    SF37_BACKEND_CUDA = 1,
} sf37_backend;

typedef enum {
    SF37_LOG_DEFAULT,
    SF37_LOG_OK,
    SF37_LOG_WARNING,
    SF37_LOG_ERROR,
} sf37_log_type;

typedef enum {
    SF37_THINK_NONE = 0,
    SF37_THINK_ENABLED = 1,
} sf37_think_mode;

#define SF37_DEFAULT_TEMPERATURE 0.0f
#define SF37_DEFAULT_TOP_P 1.0f
#define SF37_DEFAULT_MIN_P 0.0f

typedef struct {
    int *v;
    int len;
    int cap;
} sf37_tokens;

typedef struct {
    int id;
    float logit;
    float logprob;
} sf37_token_score;

typedef struct {
    uint8_t *ptr;
    uint64_t len;
    uint64_t cap;
} sf37_session_snapshot;

typedef struct {
    char *path;
    uint64_t bytes;
} sf37_session_payload_file;

typedef struct {
    const float *data;
    uint32_t rows;
    uint32_t dim;
    const float *pixel_values;
    uint32_t images;
    uint32_t pixel_channels;
    uint32_t pixel_height;
    uint32_t pixel_width;
    const float *patch_pixel_values;
    uint32_t patch_images;
    const uint32_t *patches_per_image;
} sf37_image_features;

typedef struct sf37_engine sf37_engine;
typedef struct sf37_session sf37_session;
typedef void (*sf37_session_progress_fn)(void *ud, const char *event, int current, int total);
typedef bool (*sf37_session_cancel_fn)(void *ud);

typedef struct {
    double model_map_sec;
    double preload_sec;
} sf37_engine_timing_info;

typedef struct {
    const char *model_path;
    const char *tokenizer_path;
    sf37_backend backend;
    bool inspect_only;
} sf37_engine_options;

int sf37_engine_open(sf37_engine **out, const sf37_engine_options *opt);
void sf37_engine_close(sf37_engine *e);
void sf37_engine_summary(sf37_engine *e, FILE *fp);
void sf37_engine_timing(sf37_engine *e, sf37_engine_timing_info *out);
int sf37_engine_vocab_size(sf37_engine *e);
int sf37_engine_smoke_token(sf37_engine *e, int token, FILE *fp);
int sf37_engine_smoke_decode(sf37_engine *e, int token, int layers, int topk, FILE *fp);
int sf37_engine_cuda_smoke_decode(sf37_engine *e, int token, int layers, int topk, FILE *fp);
int sf37_engine_cuda_bench_decode(sf37_engine *e, int token, int layers,
                                  int repeat, int cache_cap,
                                  bool include_logits, FILE *fp);
int sf37_engine_cuda_layer0_smoke(sf37_engine *e, int token, FILE *fp);
int sf37_engine_cuda_layer0_seq_smoke(sf37_engine *e, int token0, int token1, FILE *fp);
int sf37_engine_cuda_layer3_moe_smoke(sf37_engine *e, int token, FILE *fp);
int sf37_engine_cuda_layer_replay_smoke(sf37_engine *e, int token, int layer, FILE *fp);
int sf37_engine_collect_imatrix(sf37_engine *e,
                                const char *dataset_path,
                                const char *output_path,
                                int ctx_size,
                                int max_prompts,
                                int max_tokens);
const char *sf37_backend_name(sf37_backend backend);
void sf37_log(FILE *fp, sf37_log_type type, const char *fmt, ...);

void sf37_tokens_push(sf37_tokens *tv, int token);
void sf37_tokens_free(sf37_tokens *tv);
void sf37_tokens_copy(sf37_tokens *dst, const sf37_tokens *src);
bool sf37_tokens_starts_with(const sf37_tokens *tokens, const sf37_tokens *prefix);

int sf37_tokenizer_ready(sf37_engine *e);
void sf37_tokenize_text(sf37_engine *e, const char *text, sf37_tokens *out);
void sf37_tokenize_rendered_chat(sf37_engine *e, const char *text, sf37_tokens *out);
void sf37_chat_begin(sf37_engine *e, sf37_tokens *tokens);
void sf37_chat_append_message(sf37_engine *e, sf37_tokens *tokens,
                              const char *role, const char *content);
void sf37_chat_append_assistant_prefix(sf37_engine *e, sf37_tokens *tokens,
                                       sf37_think_mode think_mode);
void sf37_encode_chat_prompt(sf37_engine *e, const char *system, const char *prompt,
                             sf37_think_mode think_mode, sf37_tokens *out);
char *sf37_token_text(sf37_engine *e, int token, size_t *len);
int sf37_token_bos(sf37_engine *e);
int sf37_token_eos(sf37_engine *e);
int sf37_token_im_start(sf37_engine *e);
int sf37_token_im_end(sf37_engine *e);
int sf37_token_im_patch(sf37_engine *e);

int sf37_session_create(sf37_session **out, sf37_engine *e, int ctx_size);
void sf37_session_free(sf37_session *s);
void sf37_session_set_progress(sf37_session *s, sf37_session_progress_fn fn, void *ud);
void sf37_session_set_cancel(sf37_session *s, sf37_session_cancel_fn fn, void *ud);
void sf37_session_report_progress(sf37_session *s, const char *event, int current, int total);
int sf37_session_eval(sf37_session *s, int token, char *err, size_t errlen);
int sf37_session_eval_no_logits(sf37_session *s, int token, char *err, size_t errlen);
int sf37_session_eval_argmax(sf37_session *s, int token, char *err, size_t errlen);
int sf37_session_output_logits(sf37_session *s, char *err, size_t errlen);
int sf37_session_output_argmax(sf37_session *s, char *err, size_t errlen);
int sf37_session_sync(sf37_session *s, const sf37_tokens *prompt, char *err, size_t errlen);
int sf37_session_sync_multimodal(sf37_session *s, const sf37_tokens *prompt,
                                 const sf37_image_features *image_features,
                                 char *err, size_t errlen);
void sf37_session_rewind(sf37_session *s, int pos);
int sf37_session_common_prefix(sf37_session *s, const sf37_tokens *prompt);
uint64_t sf37_session_snapshot_bytes(sf37_session *s);
uint64_t sf37_session_payload_bytes(sf37_session *s);
int sf37_session_stage_payload(sf37_session *s, sf37_session_payload_file *out, char *err, size_t errlen);
int sf37_session_write_staged_payload(const sf37_session_payload_file *payload,
                                      FILE *fp, char *err, size_t errlen);
void sf37_session_payload_file_free(sf37_session_payload_file *payload);
int sf37_session_save_payload(sf37_session *s, FILE *fp, char *err, size_t errlen);
int sf37_session_load_payload(sf37_session *s, FILE *fp, uint64_t payload_bytes,
                              char *err, size_t errlen);
int sf37_session_save_snapshot(sf37_session *s, sf37_session_snapshot *snap, char *err, size_t errlen);
int sf37_session_load_snapshot(sf37_session *s, const sf37_session_snapshot *snap, char *err, size_t errlen);
void sf37_session_snapshot_free(sf37_session_snapshot *snap);
int sf37_session_argmax(sf37_session *s);
int sf37_session_argmax_excluding(sf37_session *s, int excluded_id);
int sf37_sample_logits(const float *logits, int n_vocab, float temperature,
                       int top_k, float top_p, float min_p, uint64_t *rng);
int sf37_session_sample(sf37_session *s, float temperature, int top_k,
                        float top_p, float min_p, uint64_t *rng);
int sf37_session_top_logprobs(sf37_session *s, sf37_token_score *out, int k);
int sf37_session_token_logprob(sf37_session *s, int token, sf37_token_score *out);
int sf37_session_copy_logits(sf37_session *s, float *out, int cap);
int sf37_session_pos(sf37_session *s);
int sf37_session_ctx(sf37_session *s);
uint64_t sf37_session_kv_bytes(sf37_session *s);
const sf37_tokens *sf37_session_tokens(sf37_session *s);

#endif
