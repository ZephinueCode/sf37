#include "sf37_image.h"

#include <ctype.h>
#include <math.h>
#include <stdbool.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define STBI_ONLY_JPEG
#define STBI_ONLY_PNG
#define STB_IMAGE_IMPLEMENTATION
#include "third_party/stb_image.h"

#define SF37_IMAGE_MAIN_SIZE 728u
#define SF37_IMAGE_PATCH_SIZE 504u
#define SF37_IMAGE_MAIN_ROWS 169u
#define SF37_IMAGE_PATCH_ROWS 81u
#define SF37_IMAGE_MAX_SIZE 3024u

typedef struct {
    uint8_t *p;
    uint32_t w;
    uint32_t h;
} image_u8;

typedef struct {
    char *ptr;
    size_t len;
    size_t cap;
} sbuf;

static void set_err(char *err, size_t errlen, const char *fmt, ...) {
    if (!err || errlen == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(err, errlen, fmt, ap);
    va_end(ap);
}

static void *xmalloc_local(size_t n) {
    void *p = malloc(n ? n : 1u);
    return p;
}

static void *xrealloc_local(void *p, size_t n) {
    return realloc(p, n ? n : 1u);
}

static char *xstrdup_local(const char *s) {
    size_t n = strlen(s ? s : "");
    char *p = xmalloc_local(n + 1u);
    if (!p) return NULL;
    memcpy(p, s ? s : "", n);
    p[n] = '\0';
    return p;
}

static bool sbuf_reserve(sbuf *b, size_t add) {
    size_t need = b->len + add + 1u;
    if (need <= b->cap) return true;
    size_t cap = b->cap ? b->cap : 256u;
    while (cap < need) {
        if (cap > SIZE_MAX / 2u) return false;
        cap *= 2u;
    }
    char *p = xrealloc_local(b->ptr, cap);
    if (!p) return false;
    b->ptr = p;
    b->cap = cap;
    return true;
}

static bool sbuf_putn(sbuf *b, const char *s, size_t n) {
    if (!sbuf_reserve(b, n)) return false;
    if (n) memcpy(b->ptr + b->len, s, n);
    b->len += n;
    b->ptr[b->len] = '\0';
    return true;
}

static bool sbuf_puts(sbuf *b, const char *s) {
    return sbuf_putn(b, s ? s : "", strlen(s ? s : ""));
}

static bool sbuf_putc(sbuf *b, char c) {
    return sbuf_putn(b, &c, 1u);
}

static char *sbuf_take(sbuf *b) {
    if (!b->ptr) return xstrdup_local("");
    char *p = b->ptr;
    memset(b, 0, sizeof(*b));
    return p;
}

static void sbuf_free(sbuf *b) {
    free(b->ptr);
    memset(b, 0, sizeof(*b));
}

static void image_free(image_u8 *img) {
    if (!img) return;
    free(img->p);
    memset(img, 0, sizeof(*img));
}

static bool image_alloc(image_u8 *img, uint32_t w, uint32_t h) {
    if (!img || w == 0 || h == 0) return false;
    uint64_t bytes = (uint64_t)w * h * 3u;
    if (bytes > SIZE_MAX) return false;
    img->p = xmalloc_local((size_t)bytes);
    if (!img->p) return false;
    img->w = w;
    img->h = h;
    return true;
}

static uint8_t bilerp_u8(const image_u8 *src, float sx, float sy, int c) {
    if (sx < 0.0f) sx = 0.0f;
    if (sy < 0.0f) sy = 0.0f;
    if (sx > (float)(src->w - 1u)) sx = (float)(src->w - 1u);
    if (sy > (float)(src->h - 1u)) sy = (float)(src->h - 1u);
    uint32_t x0 = (uint32_t)floorf(sx);
    uint32_t y0 = (uint32_t)floorf(sy);
    uint32_t x1 = x0 + 1u < src->w ? x0 + 1u : x0;
    uint32_t y1 = y0 + 1u < src->h ? y0 + 1u : y0;
    float tx = sx - (float)x0;
    float ty = sy - (float)y0;
    const uint8_t *p00 = src->p + ((uint64_t)y0 * src->w + x0) * 3u;
    const uint8_t *p01 = src->p + ((uint64_t)y0 * src->w + x1) * 3u;
    const uint8_t *p10 = src->p + ((uint64_t)y1 * src->w + x0) * 3u;
    const uint8_t *p11 = src->p + ((uint64_t)y1 * src->w + x1) * 3u;
    float a = (float)p00[c] * (1.0f - tx) + (float)p01[c] * tx;
    float b = (float)p10[c] * (1.0f - tx) + (float)p11[c] * tx;
    float v = a * (1.0f - ty) + b * ty;
    if (v < 0.0f) v = 0.0f;
    if (v > 255.0f) v = 255.0f;
    return (uint8_t)(v + 0.5f);
}

static bool image_resize_bilinear(const image_u8 *src, uint32_t w, uint32_t h,
                                  image_u8 *dst) {
    if (!src || !src->p || !image_alloc(dst, w, h)) return false;
    const float scale_x = (float)src->w / (float)w;
    const float scale_y = (float)src->h / (float)h;
    for (uint32_t y = 0; y < h; y++) {
        float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        for (uint32_t x = 0; x < w; x++) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            uint8_t *d = dst->p + ((uint64_t)y * w + x) * 3u;
            d[0] = bilerp_u8(src, sx, sy, 0);
            d[1] = bilerp_u8(src, sx, sy, 1);
            d[2] = bilerp_u8(src, sx, sy, 2);
        }
    }
    return true;
}

static bool image_crop(const image_u8 *src, uint32_t x, uint32_t y,
                       uint32_t w, uint32_t h, image_u8 *dst) {
    if (!src || !src->p || x > src->w || y > src->h ||
        w > src->w - x || h > src->h - y) return false;
    if (!image_alloc(dst, w, h)) return false;
    for (uint32_t yy = 0; yy < h; yy++) {
        const uint8_t *sp = src->p + ((uint64_t)(y + yy) * src->w + x) * 3u;
        uint8_t *dp = dst->p + (uint64_t)yy * w * 3u;
        memcpy(dp, sp, (size_t)w * 3u);
    }
    return true;
}

static bool image_square_pad_top_left(const image_u8 *src, image_u8 *dst) {
    if (!src || !src->p) return false;
    uint32_t size = src->w > src->h ? src->w : src->h;
    if (!image_alloc(dst, size, size)) return false;
    memset(dst->p, 0, (size_t)((uint64_t)size * size * 3u));
    for (uint32_t y = 0; y < src->h; y++) {
        memcpy(dst->p + (uint64_t)y * size * 3u,
               src->p + (uint64_t)y * src->w * 3u,
               (size_t)src->w * 3u);
    }
    return true;
}

static float *image_to_normalized_nchw(const image_u8 *src, uint32_t out_size) {
    if (!src || !src->p || out_size == 0) return NULL;
    uint64_t count = (uint64_t)3u * out_size * out_size;
    if (count > SIZE_MAX / sizeof(float)) return NULL;
    float *out = xmalloc_local((size_t)count * sizeof(float));
    if (!out) return NULL;
    static const float mean[3] = {0.48145466f, 0.4578275f, 0.40821073f};
    static const float stdv[3] = {0.26862954f, 0.26130258f, 0.27577711f};
    const float scale_x = (float)src->w / (float)out_size;
    const float scale_y = (float)src->h / (float)out_size;
    const uint64_t plane = (uint64_t)out_size * out_size;
    for (uint32_t y = 0; y < out_size; y++) {
        float sy = ((float)y + 0.5f) * scale_y - 0.5f;
        for (uint32_t x = 0; x < out_size; x++) {
            float sx = ((float)x + 0.5f) * scale_x - 0.5f;
            for (int c = 0; c < 3; c++) {
                float v = (float)bilerp_u8(src, sx, sy, c) / 255.0f;
                out[(uint64_t)c * plane + (uint64_t)y * out_size + x] =
                    (v - mean[c]) / stdv[c];
            }
        }
    }
    return out;
}

static int b64_value(unsigned char c) {
    if (c >= 'A' && c <= 'Z') return (int)(c - 'A');
    if (c >= 'a' && c <= 'z') return (int)(c - 'a' + 26);
    if (c >= '0' && c <= '9') return (int)(c - '0' + 52);
    if (c == '+') return 62;
    if (c == '/') return 63;
    return -1;
}

static bool base64_decode_alloc(const char *s, uint8_t **out, size_t *out_len) {
    *out = NULL;
    *out_len = 0;
    if (!s) return false;
    const char *comma = strchr(s, ',');
    if (comma && !strncmp(s, "data:", 5)) s = comma + 1;
    size_t clean = 0;
    for (const char *p = s; *p; p++) {
        if (!isspace((unsigned char)*p)) clean++;
    }
    if (clean == 0 || (clean & 3u) != 0) return false;
    size_t cap = clean / 4u * 3u;
    uint8_t *buf = xmalloc_local(cap ? cap : 1u);
    if (!buf) return false;
    size_t n = 0;
    int vals[4];
    int vi = 0;
    for (const char *p = s; *p; p++) {
        unsigned char c = (unsigned char)*p;
        if (isspace(c)) continue;
        if (c == '=') vals[vi++] = -2;
        else {
            int v = b64_value(c);
            if (v < 0) {
                free(buf);
                return false;
            }
            vals[vi++] = v;
        }
        if (vi == 4) {
            if (vals[0] < 0 || vals[1] < 0 ||
                vals[2] == -1 || vals[3] == -1) {
                free(buf);
                return false;
            }
            uint32_t tri = ((uint32_t)vals[0] << 18) |
                           ((uint32_t)vals[1] << 12) |
                           ((uint32_t)(vals[2] < 0 ? 0 : vals[2]) << 6) |
                           (uint32_t)(vals[3] < 0 ? 0 : vals[3]);
            buf[n++] = (uint8_t)(tri >> 16);
            if (vals[2] != -2) buf[n++] = (uint8_t)(tri >> 8);
            if (vals[3] != -2) buf[n++] = (uint8_t)tri;
            vi = 0;
        }
    }
    *out = buf;
    *out_len = n;
    return true;
}

static int hex_digit(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
}

static char *percent_decode(const char *s) {
    size_t n = strlen(s ? s : "");
    char *out = xmalloc_local(n + 1u);
    if (!out) return NULL;
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (s[i] == '%' && i + 2u < n) {
            int hi = hex_digit(s[i + 1u]);
            int lo = hex_digit(s[i + 2u]);
            if (hi >= 0 && lo >= 0) {
                out[j++] = (char)((hi << 4) | lo);
                i += 2u;
                continue;
            }
        }
        out[j++] = s[i];
    }
    out[j] = '\0';
    return out;
}

static char *source_to_local_path(const char *source, char *err, size_t errlen) {
    if (!source || !source[0]) {
        set_err(err, errlen, "empty image source");
        return NULL;
    }
    if (!strncmp(source, "http://", 7) || !strncmp(source, "https://", 8)) {
        set_err(err, errlen,
                "network image URLs are not supported; use a local path or file:// URL");
        return NULL;
    }
    if (strncmp(source, "file://", 7) != 0) return xstrdup_local(source);
    const char *p = source + 7;
    if (!strncmp(p, "localhost/", 10)) p += 9;
    if (p[0] != '/') {
        set_err(err, errlen, "file:// image URL must use an absolute local path");
        return NULL;
    }
    return percent_decode(p);
}

static bool load_image_source_rgb(const char *source, image_u8 *out,
                                  char *err, size_t errlen) {
    memset(out, 0, sizeof(*out));
    int w = 0, h = 0, comp = 0;
    unsigned char *pixels = NULL;
    if (source && !strncmp(source, "data:", 5)) {
        const char *comma = strchr(source, ',');
        if (!comma || !strstr(source, ";base64")) {
            set_err(err, errlen, "only base64 data:image URLs are supported");
            return false;
        }
        uint8_t *bytes = NULL;
        size_t nbytes = 0;
        if (!base64_decode_alloc(source, &bytes, &nbytes)) {
            set_err(err, errlen, "invalid base64 image data URL");
            return false;
        }
        pixels = stbi_load_from_memory(bytes, (int)nbytes, &w, &h, &comp, 3);
        free(bytes);
    } else {
        char *path = source_to_local_path(source, err, errlen);
        if (!path) return false;
        pixels = stbi_load(path, &w, &h, &comp, 3);
        if (!pixels) set_err(err, errlen, "failed to decode image %s: %s",
                             path, stbi_failure_reason());
        free(path);
    }
    if (!pixels) return false;
    if (w <= 0 || h <= 0) {
        stbi_image_free(pixels);
        set_err(err, errlen, "decoded image has invalid dimensions");
        return false;
    }
    uint64_t bytes = (uint64_t)(uint32_t)w * (uint32_t)h * 3u;
    if (bytes > SIZE_MAX) {
        stbi_image_free(pixels);
        set_err(err, errlen, "decoded image is too large");
        return false;
    }
    out->p = xmalloc_local((size_t)bytes);
    if (!out->p) {
        stbi_image_free(pixels);
        set_err(err, errlen, "out of memory decoding image");
        return false;
    }
    memcpy(out->p, pixels, (size_t)bytes);
    out->w = (uint32_t)w;
    out->h = (uint32_t)h;
    stbi_image_free(pixels);
    return true;
}

static uint32_t determine_window_size(uint32_t long_side, uint32_t short_side) {
    if (short_side == 0) return 0;
    double ratio = (double)long_side / (double)short_side;
    if (long_side <= SF37_IMAGE_MAIN_SIZE) {
        return ratio > 1.5 ? short_side : 0u;
    }
    return ratio > 4.0 ? (short_side < SF37_IMAGE_PATCH_SIZE ? short_side : SF37_IMAGE_PATCH_SIZE)
                       : SF37_IMAGE_PATCH_SIZE;
}

static void get_size_for_padding(uint32_t w, uint32_t h,
                                 uint32_t *out_w, uint32_t *out_h) {
    double ratio = h ? (double)w / (double)h : 1.0;
    if ((w < 32u || h < 32u) && (ratio > 4.0 || ratio < 0.25)) {
        uint32_t size = w > h ? w : h;
        *out_w = size;
        *out_h = size;
    } else {
        *out_w = w;
        *out_h = h;
    }
}

static void get_size_for_preprocess(uint32_t w, uint32_t h,
                                    uint32_t *out_w, uint32_t *out_h) {
    uint32_t max_side = w > h ? w : h;
    if (max_side > SF37_IMAGE_MAX_SIZE) {
        double scale = (double)SF37_IMAGE_MAX_SIZE / (double)max_side;
        w = (uint32_t)((double)w * scale);
        h = (uint32_t)((double)h * scale);
        if (w == 0) w = 1;
        if (h == 0) h = 1;
    }
    *out_w = w;
    *out_h = h;
}

static void get_size_for_crop(uint32_t w, uint32_t h, uint32_t window,
                              uint32_t *out_w, uint32_t *out_h) {
    if (window == 0) {
        *out_w = w;
        *out_h = h;
        return;
    }
    double wr = (double)w / (double)window;
    double hr = (double)h / (double)window;
    if (wr < 1.0) {
        *out_w = w;
    } else {
        double dec = wr - (double)(w / window);
        uint32_t n = (uint32_t)wr + (dec > 0.2 ? 1u : 0u);
        *out_w = window * n;
    }
    if (hr < 1.0) {
        *out_h = h;
    } else {
        double dec = hr - (double)(h / window);
        uint32_t n = (uint32_t)hr + (dec > 0.2 ? 1u : 0u);
        *out_h = window * n;
    }
}

static bool ensure_main_cap(sf37_image_batch *b, uint32_t need) {
    if (need <= b->cap_images) return true;
    uint32_t cap = b->cap_images ? b->cap_images * 2u : 1u;
    while (cap < need) cap *= 2u;
    uint64_t floats = (uint64_t)cap * 3u * SF37_IMAGE_MAIN_SIZE * SF37_IMAGE_MAIN_SIZE;
    if (floats > SIZE_MAX / sizeof(float)) return false;
    float *p = xrealloc_local(b->pixel_values, (size_t)floats * sizeof(float));
    if (!p) return false;
    b->pixel_values = p;
    uint32_t *pp = xrealloc_local(b->patches_per_image, (size_t)cap * sizeof(uint32_t));
    if (!pp) return false;
    b->patches_per_image = pp;
    b->cap_images = cap;
    b->cap_patches_per_image = cap;
    return true;
}

static bool ensure_patch_cap(sf37_image_batch *b, uint32_t need) {
    if (need <= b->cap_patch_images) return true;
    uint32_t cap = b->cap_patch_images ? b->cap_patch_images * 2u : 1u;
    while (cap < need) cap *= 2u;
    uint64_t floats = (uint64_t)cap * 3u * SF37_IMAGE_PATCH_SIZE * SF37_IMAGE_PATCH_SIZE;
    if (floats > SIZE_MAX / sizeof(float)) return false;
    float *p = xrealloc_local(b->patch_pixel_values, (size_t)floats * sizeof(float));
    if (!p) return false;
    b->patch_pixel_values = p;
    uint8_t *m = xrealloc_local(b->patch_newline_mask, (size_t)cap * sizeof(uint8_t));
    if (!m) return false;
    b->patch_newline_mask = m;
    b->cap_patch_images = cap;
    b->cap_patch_newline_mask = cap;
    return true;
}

static bool append_main_pixels(sf37_image_batch *b, const float *pixels) {
    if (!ensure_main_cap(b, b->images + 1u)) return false;
    uint64_t floats = (uint64_t)3u * SF37_IMAGE_MAIN_SIZE * SF37_IMAGE_MAIN_SIZE;
    memcpy(b->pixel_values + (uint64_t)b->images * floats, pixels,
           (size_t)floats * sizeof(float));
    b->images++;
    return true;
}

static bool append_patch_pixels(sf37_image_batch *b, const float *pixels, bool newline) {
    if (!ensure_patch_cap(b, b->patch_images + 1u)) return false;
    uint64_t floats = (uint64_t)3u * SF37_IMAGE_PATCH_SIZE * SF37_IMAGE_PATCH_SIZE;
    memcpy(b->patch_pixel_values + (uint64_t)b->patch_images * floats, pixels,
           (size_t)floats * sizeof(float));
    b->patch_newline_mask[b->patch_images] = newline ? 1u : 0u;
    b->patch_images++;
    return true;
}

void sf37_image_batch_free(sf37_image_batch *b) {
    if (!b) return;
    free(b->pixel_values);
    free(b->patch_pixel_values);
    free(b->patches_per_image);
    free(b->patch_newline_mask);
    memset(b, 0, sizeof(*b));
}

static bool append_placeholder_rows(sbuf *out, const char *start,
                                    const char *end, uint32_t rows) {
    if (!sbuf_puts(out, start)) return false;
    for (uint32_t i = 0; i < rows; i++) {
        if (!sbuf_puts(out, "<im_patch>")) return false;
    }
    return sbuf_puts(out, end);
}

char *sf37_image_batch_placeholder_for_image(const sf37_image_batch *b,
                                             uint32_t image_index) {
    if (!b || image_index >= b->images || !b->patches_per_image) return NULL;
    uint32_t patch_base = 0;
    for (uint32_t i = 0; i < image_index; i++) patch_base += b->patches_per_image[i];
    uint32_t np = b->patches_per_image[image_index];
    if (patch_base > b->patch_images || np > b->patch_images - patch_base) return NULL;
    sbuf out = {0};
    for (uint32_t i = 0; i < np; i++) {
        if (!append_placeholder_rows(&out, "<patch_start>", "<patch_end>",
                                     SF37_IMAGE_PATCH_ROWS)) goto oom;
        if (b->patch_newline_mask && b->patch_newline_mask[patch_base + i]) {
            if (!sbuf_puts(&out, "<patch_newline>")) goto oom;
        }
    }
    if (!append_placeholder_rows(&out, "<im_start>", "<im_end>",
                                 SF37_IMAGE_MAIN_ROWS)) goto oom;
    return sbuf_take(&out);
oom:
    sbuf_free(&out);
    return NULL;
}

char *sf37_image_batch_placeholder_text(const sf37_image_batch *b) {
    if (!b || b->images == 0) return xstrdup_local("");
    sbuf out = {0};
    for (uint32_t i = 0; i < b->images; i++) {
        char *one = sf37_image_batch_placeholder_for_image(b, i);
        if (!one || !sbuf_puts(&out, one)) {
            free(one);
            sbuf_free(&out);
            return NULL;
        }
        free(one);
    }
    return sbuf_take(&out);
}

char *sf37_image_batch_apply_placeholders(const sf37_image_batch *b,
                                          const char *text,
                                          char *err,
                                          size_t errlen) {
    if (!b || b->images == 0) return xstrdup_local(text ? text : "");
    const char *needle = "<im_patch>";
    const size_t needle_len = strlen(needle);
    uint32_t count = 0;
    for (const char *p = text ? text : ""; (p = strstr(p, needle)) != NULL; p += needle_len) {
        count++;
    }
    if (count != 0 && count != b->images) {
        set_err(err, errlen,
                "prompt has %u <im_patch> placeholders but %u images were supplied",
                count, b->images);
        return NULL;
    }
    sbuf out = {0};
    if (count == 0) {
        if (text && text[0]) {
            if (!sbuf_puts(&out, text) || !sbuf_putc(&out, '\n')) goto oom;
        }
        char *all = sf37_image_batch_placeholder_text(b);
        if (!all || !sbuf_puts(&out, all)) {
            free(all);
            goto oom;
        }
        free(all);
        return sbuf_take(&out);
    }
    const char *cur = text ? text : "";
    for (uint32_t i = 0; i < b->images; i++) {
        const char *hit = strstr(cur, needle);
        if (!hit) goto oom;
        if (!sbuf_putn(&out, cur, (size_t)(hit - cur))) goto oom;
        char *one = sf37_image_batch_placeholder_for_image(b, i);
        if (!one || !sbuf_puts(&out, one)) {
            free(one);
            goto oom;
        }
        free(one);
        cur = hit + needle_len;
    }
    if (!sbuf_puts(&out, cur)) goto oom;
    return sbuf_take(&out);
oom:
    set_err(err, errlen, "out of memory building image placeholders");
    sbuf_free(&out);
    return NULL;
}

static bool process_source_image(sf37_image_batch *b, image_u8 *decoded,
                                 char *err, size_t errlen) {
    image_u8 img = *decoded;
    memset(decoded, 0, sizeof(*decoded));
    uint32_t pad_w = 0, pad_h = 0;
    get_size_for_padding(img.w, img.h, &pad_w, &pad_h);
    if (pad_w != img.w || pad_h != img.h) {
        image_u8 padded = {0};
        if (!image_square_pad_top_left(&img, &padded)) {
            image_free(&img);
            set_err(err, errlen, "out of memory padding image");
            return false;
        }
        image_free(&img);
        img = padded;
    }
    const uint32_t base_w = img.w;
    const uint32_t base_h = img.h;
    uint32_t prep_w = 0, prep_h = 0;
    get_size_for_preprocess(base_w, base_h, &prep_w, &prep_h);
    if (prep_w != img.w || prep_h != img.h) {
        image_u8 resized = {0};
        if (!image_resize_bilinear(&img, prep_w, prep_h, &resized)) {
            image_free(&img);
            set_err(err, errlen, "out of memory resizing image");
            return false;
        }
        image_free(&img);
        img = resized;
    }
    uint32_t long_side = prep_w > prep_h ? prep_w : prep_h;
    uint32_t short_side = prep_w < prep_h ? prep_w : prep_h;
    uint32_t window = determine_window_size(long_side, short_side);
    const uint32_t image_index = b->images;
    if (!ensure_main_cap(b, image_index + 1u)) {
        image_free(&img);
        set_err(err, errlen, "out of memory growing image batch");
        return false;
    }
    uint32_t patch_count = 0;
    if (window != 0) {
        uint32_t crop_w = 0, crop_h = 0;
        get_size_for_crop(prep_w, prep_h, window, &crop_w, &crop_h);
        image_u8 crop_src = {0};
        bool owns_crop_src = false;
        if (crop_w != base_w || crop_h != base_h) {
            if (!image_resize_bilinear(&img, crop_w, crop_h, &crop_src)) {
                image_free(&img);
                set_err(err, errlen, "out of memory resizing image for patches");
                return false;
            }
            owns_crop_src = true;
        } else {
            crop_src = img;
        }
        uint32_t x_num = crop_w <= window ? 1u :
            (uint32_t)ceil(((double)crop_w - (double)window) / (double)window + 1.0);
        uint32_t y_num = crop_h <= window ? 1u :
            (uint32_t)ceil(((double)crop_h - (double)window) / (double)window + 1.0);
        for (uint32_t yy = 0; yy < y_num; yy++) {
            uint32_t y = window * yy;
            if (y_num > 1u && y + window > crop_h) y = crop_h - window;
            for (uint32_t xx = 0; xx < x_num; xx++) {
                uint32_t x = window * xx;
                if (x_num > 1u && x + window > crop_w) x = crop_w - window;
                image_u8 patch = {0};
                if (!image_crop(&crop_src, x, y, window, window, &patch)) {
                    if (owns_crop_src) image_free(&crop_src);
                    image_free(&img);
                    set_err(err, errlen, "out of memory cropping image patch");
                    return false;
                }
                float *pf = image_to_normalized_nchw(&patch, SF37_IMAGE_PATCH_SIZE);
                image_free(&patch);
                if (!pf) {
                    if (owns_crop_src) image_free(&crop_src);
                    image_free(&img);
                    set_err(err, errlen, "out of memory preprocessing image patch");
                    return false;
                }
                bool newline = (xx + 1u == x_num) && !(yy + 1u == y_num);
                if (!append_patch_pixels(b, pf, newline)) {
                    free(pf);
                    if (owns_crop_src) image_free(&crop_src);
                    image_free(&img);
                    set_err(err, errlen, "out of memory appending image patch");
                    return false;
                }
                free(pf);
                patch_count++;
            }
        }
        if (owns_crop_src) image_free(&crop_src);
    }
    float *main_pixels = image_to_normalized_nchw(&img, SF37_IMAGE_MAIN_SIZE);
    image_free(&img);
    if (!main_pixels) {
        set_err(err, errlen, "out of memory preprocessing main image");
        return false;
    }
    if (!append_main_pixels(b, main_pixels)) {
        free(main_pixels);
        set_err(err, errlen, "out of memory appending main image");
        return false;
    }
    free(main_pixels);
    b->patches_per_image[image_index] = patch_count;
    return true;
}

int sf37_image_batch_add_source(sf37_image_batch *b, const char *source,
                                char *err, size_t errlen) {
    if (!b || !source) {
        set_err(err, errlen, "invalid image source");
        return 1;
    }
    image_u8 decoded = {0};
    if (!load_image_source_rgb(source, &decoded, err, errlen)) return 1;
    if (!process_source_image(b, &decoded, err, errlen)) {
        image_free(&decoded);
        return 1;
    }
    return 0;
}

void sf37_image_features_from_batch(const sf37_image_batch *b,
                                    sf37_image_features *out) {
    if (!out) return;
    memset(out, 0, sizeof(*out));
    if (!b || b->images == 0) return;
    out->pixel_values = b->pixel_values;
    out->images = b->images;
    out->pixel_channels = 3u;
    out->pixel_height = SF37_IMAGE_MAIN_SIZE;
    out->pixel_width = SF37_IMAGE_MAIN_SIZE;
    out->patch_pixel_values = b->patch_pixel_values;
    out->patch_images = b->patch_images;
    out->patches_per_image = b->patches_per_image;
}
