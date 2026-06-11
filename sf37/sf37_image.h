#ifndef SF37_IMAGE_H
#define SF37_IMAGE_H

#include "sf37.h"

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float *pixel_values;
    uint32_t images;
    uint32_t cap_images;
    float *patch_pixel_values;
    uint32_t patch_images;
    uint32_t cap_patch_images;
    uint32_t *patches_per_image;
    uint32_t cap_patches_per_image;
    uint8_t *patch_newline_mask;
    uint32_t cap_patch_newline_mask;
} sf37_image_batch;

void sf37_image_batch_free(sf37_image_batch *b);
int sf37_image_batch_add_source(sf37_image_batch *b, const char *source,
                                char *err, size_t errlen);
char *sf37_image_batch_placeholder_for_image(const sf37_image_batch *b,
                                             uint32_t image_index);
char *sf37_image_batch_placeholder_text(const sf37_image_batch *b);
char *sf37_image_batch_apply_placeholders(const sf37_image_batch *b,
                                          const char *text,
                                          char *err,
                                          size_t errlen);
void sf37_image_features_from_batch(const sf37_image_batch *b,
                                    sf37_image_features *out);

#endif
