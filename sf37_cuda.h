#ifndef SF37_CUDA_H
#define SF37_CUDA_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct sf37_cuda_tensor sf37_cuda_tensor;

int sf37_cuda_device_count(void);
void sf37_cuda_print_devices(void);
int sf37_cuda_init(void);
void sf37_cuda_cleanup(void);
int sf37_cuda_synchronize(void);
int sf37_cuda_memory_info(uint64_t *free_bytes, uint64_t *total_bytes);
int sf37_cuda_begin_layer(uint32_t layer);
void sf37_cuda_end_layer(void);

int sf37_cuda_set_model_fd(int fd);
int sf37_cuda_set_model_map_range(const void *model_map, uint64_t model_size,
                                  uint64_t map_offset, uint64_t map_size,
                                  uint64_t max_tensor_bytes);
int sf37_cuda_set_model_map_spans(const void *model_map, uint64_t model_size,
                                  const uint64_t *offsets,
                                  const uint64_t *sizes,
                                  uint32_t count,
                                  uint64_t max_tensor_bytes);
int sf37_cuda_cache_model_range(const void *model_map, uint64_t model_size,
                                uint64_t offset, uint64_t bytes,
                                const char *label);
void sf37_cuda_evict_model_cache(const char *reason);
int sf37_cuda_cache_q8_f16_range(const void *model_map, uint64_t model_size,
                                 uint64_t offset, uint64_t bytes,
                                 uint64_t in_dim, uint64_t out_dim,
                                 const char *label);

sf37_cuda_tensor *sf37_cuda_tensor_alloc(uint64_t bytes);
sf37_cuda_tensor *sf37_cuda_tensor_alloc_managed(uint64_t bytes);
void sf37_cuda_tensor_free(sf37_cuda_tensor *tensor);
uint64_t sf37_cuda_tensor_bytes(const sf37_cuda_tensor *tensor);
int sf37_cuda_tensor_write(sf37_cuda_tensor *tensor, uint64_t offset,
                           const void *data, uint64_t bytes);
int sf37_cuda_tensor_read(const sf37_cuda_tensor *tensor, uint64_t offset,
                          void *data, uint64_t bytes);
int sf37_cuda_tensor_copy(sf37_cuda_tensor *dst, uint64_t dst_offset,
                          const sf37_cuda_tensor *src, uint64_t src_offset,
                          uint64_t bytes);
int sf37_cuda_tensor_fill_f32(sf37_cuda_tensor *tensor, float value, uint64_t count);
int sf37_cuda_embed_tokens_bf16_mapped(sf37_cuda_tensor *out,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint64_t dim,
                                        uint64_t vocab,
                                        const sf37_cuda_tensor *tokens,
                                        uint32_t n_tok);

int sf37_cuda_rms_norm_weight1_f32(sf37_cuda_tensor *out,
                                   const sf37_cuda_tensor *x,
                                   const sf37_cuda_tensor *weight,
                                   uint32_t n, float eps);
int sf37_cuda_rms_norm_weight1_bf16(sf37_cuda_tensor *out,
                                    const sf37_cuda_tensor *x,
                                    const sf37_cuda_tensor *weight,
                                    uint32_t n, float eps);
int sf37_cuda_add_inplace_f32(sf37_cuda_tensor *dst,
                              const sf37_cuda_tensor *src,
                              uint64_t n);
int sf37_cuda_swiglu_f32(sf37_cuda_tensor *out,
                         const sf37_cuda_tensor *gate,
                         const sf37_cuda_tensor *up,
                         uint64_t n, float clamp);
int sf37_cuda_head_rms_norm_weight1_bf16(sf37_cuda_tensor *x,
                                         const sf37_cuda_tensor *weight,
                                         uint32_t n_head,
                                         uint32_t head_dim,
                                         float eps);
int sf37_cuda_rope_split_half(sf37_cuda_tensor *x,
                              uint32_t n_head,
                              uint32_t head_dim,
                              uint32_t rotary_dim,
                              double theta,
                              int llama3,
                              uint32_t pos);
int sf37_cuda_gqa_single_token_heads(sf37_cuda_tensor *out_heads,
                                     const sf37_cuda_tensor *v,
                                     const sf37_cuda_tensor *head_gate,
                                     uint32_t q_heads,
                                     uint32_t kv_heads,
                                     uint32_t head_dim);
int sf37_cuda_attention_decode_heads(sf37_cuda_tensor *out_heads,
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
                                     uint32_t window);
int sf37_cuda_attention_decode_heads_at(sf37_cuda_tensor *out_heads,
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
                                        uint32_t window);
int sf37_cuda_should_use_managed_kv_cache(uint64_t kv_cache_bytes,
                                          uint64_t context_bytes);
int sf37_cuda_matvec_q8_0(sf37_cuda_tensor *out,
                          const sf37_cuda_tensor *weights,
                          uint64_t in_dim, uint64_t out_dim,
                          const sf37_cuda_tensor *x);
int sf37_cuda_matvec_q8_0_pair(sf37_cuda_tensor *out0,
                               sf37_cuda_tensor *out1,
                               const sf37_cuda_tensor *weights0,
                               const sf37_cuda_tensor *weights1,
                               uint64_t in_dim, uint64_t out_dim,
                               const sf37_cuda_tensor *x);
int sf37_cuda_matvec_bf16(sf37_cuda_tensor *out,
                          const sf37_cuda_tensor *weights,
                          uint64_t in_dim, uint64_t out_dim,
                          const sf37_cuda_tensor *x);
int sf37_cuda_argmax_f32(int32_t *out_token,
                         const sf37_cuda_tensor *logits,
                         uint32_t n_vocab,
                         int32_t excluded_id);
int sf37_cuda_matvec_f32(sf37_cuda_tensor *out,
                         const sf37_cuda_tensor *weights,
                         uint64_t in_dim, uint64_t out_dim,
                         const sf37_cuda_tensor *x);
int sf37_cuda_matvec_q3_asym(sf37_cuda_tensor *out,
                             const sf37_cuda_tensor *weights,
                             uint64_t in_dim, uint64_t out_dim,
                             const sf37_cuda_tensor *x);
int sf37_cuda_matvec_q2_asym(sf37_cuda_tensor *out,
                             const sf37_cuda_tensor *weights,
                             uint64_t in_dim, uint64_t out_dim,
                             const sf37_cuda_tensor *x);
int sf37_cuda_router_select(sf37_cuda_tensor *selected,
                            sf37_cuda_tensor *weights,
                            sf37_cuda_tensor *probs,
                            const sf37_cuda_tensor *logits,
                            const sf37_cuda_tensor *bias,
                            uint32_t n_experts,
                            uint32_t topk,
                            float scale);
int sf37_cuda_routed_moe_one(sf37_cuda_tensor *out,
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
                             const sf37_cuda_tensor *x);

int sf37_cuda_rms_norm_weight1_bf16_mapped(sf37_cuda_tensor *out,
                                           const sf37_cuda_tensor *x,
                                           const void *model_map,
                                           uint64_t model_size,
                                           uint64_t weight_offset,
                                           uint32_t n, float eps);
int sf37_cuda_rms_norm_weight1_bf16_batch_mapped(sf37_cuda_tensor *out,
                                                 const sf37_cuda_tensor *x,
                                                 const void *model_map,
                                                 uint64_t model_size,
                                                 uint64_t weight_offset,
                                                 uint32_t n,
                                                 uint32_t n_tok,
                                                 float eps);
int sf37_cuda_head_rms_norm_weight1_bf16_mapped(sf37_cuda_tensor *x,
                                                const void *model_map,
                                                uint64_t model_size,
                                                uint64_t weight_offset,
                                                uint32_t n_head,
                                                uint32_t head_dim,
                                                float eps);
int sf37_cuda_head_rms_norm_weight1_bf16_batch_mapped(sf37_cuda_tensor *x,
                                                      const void *model_map,
                                                      uint64_t model_size,
                                                      uint64_t weight_offset,
                                                      uint32_t n_tok,
                                                      uint32_t n_head,
                                                      uint32_t head_dim,
                                                      float eps);
int sf37_cuda_rope_split_half_batch(sf37_cuda_tensor *x,
                                    uint32_t n_tok,
                                    uint32_t n_head,
                                    uint32_t head_dim,
                                    uint32_t rotary_dim,
                                    double theta,
                                    int llama3,
                                    uint32_t pos0);
int sf37_cuda_store_kv_cache_batch(sf37_cuda_tensor *k_cache,
                                   sf37_cuda_tensor *v_cache,
                                   const sf37_cuda_tensor *k,
                                   const sf37_cuda_tensor *v,
                                   uint32_t pos0,
                                   uint32_t n_tok,
                                   uint32_t cache_cap,
                                   uint32_t kv_dim);
int sf37_cuda_attention_prefill_heads(sf37_cuda_tensor *out_heads,
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
                                      uint32_t window);
int sf37_cuda_matvec_q8_0_mapped(sf37_cuda_tensor *out,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t weight_offset,
                                 uint64_t in_dim, uint64_t out_dim,
                                 const sf37_cuda_tensor *x);
int sf37_cuda_matmul_q8_0_mapped(sf37_cuda_tensor *out,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t weight_offset,
                                 uint64_t in_dim, uint64_t out_dim,
                                 const sf37_cuda_tensor *x,
                                 uint64_t n_tok);
int sf37_cuda_matmul_q8_0_pair_mapped(sf37_cuda_tensor *out0,
                                      sf37_cuda_tensor *out1,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight0_offset,
                                      uint64_t weight1_offset,
                                      uint64_t in_dim,
                                      uint64_t out_dim,
                                      const sf37_cuda_tensor *x,
                                      uint64_t n_tok);
int sf37_cuda_matvec_q8_0_pair_mapped(sf37_cuda_tensor *out0,
                                      sf37_cuda_tensor *out1,
                                      const void *model_map,
                                      uint64_t model_size,
                                      uint64_t weight0_offset,
                                      uint64_t weight1_offset,
                                      uint64_t in_dim, uint64_t out_dim,
                                      const sf37_cuda_tensor *x);
int sf37_cuda_matvec_bf16_mapped(sf37_cuda_tensor *out,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t weight_offset,
                                 uint64_t in_dim, uint64_t out_dim,
                                 const sf37_cuda_tensor *x);
int sf37_cuda_matvec_bf16_argmax_mapped(int32_t *out_token,
                                        const void *model_map,
                                        uint64_t model_size,
                                        uint64_t weight_offset,
                                        uint64_t in_dim,
                                        uint64_t out_dim,
                                        const sf37_cuda_tensor *x,
                                        int32_t excluded_id);
int sf37_cuda_matmul_bf16_mapped(sf37_cuda_tensor *out,
                                 const void *model_map,
                                 uint64_t model_size,
                                 uint64_t weight_offset,
                                 uint64_t in_dim,
                                 uint64_t out_dim,
                                 const sf37_cuda_tensor *x,
                                 uint64_t n_tok);
int sf37_cuda_matvec_f32_mapped(sf37_cuda_tensor *out,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint64_t in_dim, uint64_t out_dim,
                                const sf37_cuda_tensor *x);
int sf37_cuda_matmul_f32_mapped(sf37_cuda_tensor *out,
                                const void *model_map,
                                uint64_t model_size,
                                uint64_t weight_offset,
                                uint64_t in_dim,
                                uint64_t out_dim,
                                const sf37_cuda_tensor *x,
                                uint64_t n_tok);
int sf37_cuda_matvec_q3_asym_mapped(sf37_cuda_tensor *out,
                                    const void *model_map,
                                    uint64_t model_size,
                                    uint64_t weight_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const sf37_cuda_tensor *x);
int sf37_cuda_matvec_q2_asym_mapped(sf37_cuda_tensor *out,
                                    const void *model_map,
                                    uint64_t model_size,
                                    uint64_t weight_offset,
                                    uint64_t in_dim, uint64_t out_dim,
                                    const sf37_cuda_tensor *x);
int sf37_cuda_router_select_mapped(sf37_cuda_tensor *selected,
                                   sf37_cuda_tensor *weights,
                                   sf37_cuda_tensor *probs,
                                   const sf37_cuda_tensor *logits,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t bias_offset,
                                   uint32_t n_experts,
                                   uint32_t topk,
                                   float scale);
int sf37_cuda_router_select_batch_mapped(sf37_cuda_tensor *selected,
                                         sf37_cuda_tensor *weights,
                                         sf37_cuda_tensor *probs,
                                         const sf37_cuda_tensor *logits,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t bias_offset,
                                         uint32_t n_experts,
                                         uint32_t topk,
                                         float scale,
                                         uint32_t n_tok);
int sf37_cuda_routed_moe_one_mapped(sf37_cuda_tensor *out,
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
                                    const sf37_cuda_tensor *x);
int sf37_cuda_routed_moe_batch_mapped(sf37_cuda_tensor *out,
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
                                      uint32_t n_tok);

int sf37_cuda_layer_norm_bf16_mapped(sf37_cuda_tensor *out,
                                     const sf37_cuda_tensor *x,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t weight_offset,
                                     uint64_t bias_offset,
                                     uint32_t n,
                                     uint32_t n_rows,
                                     float eps);
int sf37_cuda_add_bias_bf16_mapped(sf37_cuda_tensor *x,
                                   const void *model_map,
                                   uint64_t model_size,
                                   uint64_t bias_offset,
                                   uint32_t n,
                                   uint32_t n_rows);
int sf37_cuda_add_scaled_bf16_mapped(sf37_cuda_tensor *dst,
                                     const sf37_cuda_tensor *src,
                                     const void *model_map,
                                     uint64_t model_size,
                                     uint64_t gamma_offset,
                                     uint32_t n,
                                     uint32_t n_rows);
int sf37_cuda_quick_gelu_f32(sf37_cuda_tensor *x, uint64_t n);
int sf37_cuda_vision_conv2d_bf16_mapped(sf37_cuda_tensor *out_nlc,
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
                                        int input_nlc);
int sf37_cuda_vision_add_pos_q8_0_mapped(sf37_cuda_tensor *hidden,
                                         const void *model_map,
                                         uint64_t model_size,
                                         uint64_t pos_offset,
                                         uint32_t width,
                                         uint32_t grid_h,
                                         uint32_t grid_w,
                                         uint32_t base_grid);
int sf37_cuda_vision_qkv_split_rope_bf16_mapped(sf37_cuda_tensor *q,
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
                                                double theta);
int sf37_cuda_vision_attention(sf37_cuda_tensor *out,
                               const sf37_cuda_tensor *q,
                               const sf37_cuda_tensor *k,
                               const sf37_cuda_tensor *v,
                               uint32_t n_tok,
                               uint32_t n_heads,
                               uint32_t head_dim);

#ifdef __cplusplus
}
#endif

#endif
