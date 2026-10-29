/* See plan.h. Mirrors yblod-engine b7c39bf + big-fella-el-scaler.patch:
 * dv_source_renderer_submit_input / present_internal (dv_source_renderer.c)
 * and reconstruct_impl / reconstruct_single_p010 / reconstruct_piecewise_surface
 * (opencl_backend.c), with native-dv.env's settings fixed: SPACE2,
 * SINGLE_PIECEWISE, SINGLE_FULL_HEIGHT, SINGLE_NV12, SINGLE_ORDER2_CHROMA,
 * RECON_ACTIVE_ROWS, PRIVATE_RESIZE, DIRECT_P010, FILTER_FUSED, LUMA_PAIR,
 * BASE_PREP_FUSED, LUMA_GUIDE_FUSED, MMR_64, MMR_96, MMR_UNIVARIATE_LUT,
 * MMR_DUAL, MMR_HORIZONTAL_FUSED (no MMR_PAIR/UNIFORM/MULHI). */
#include "plan.h"
#include "dv_ffmpeg_metadata.h"
#include "mmr_prepare.h"
#include "mmr64_lut.h"
#include "pixel_polynomial.h"
#include "single_reshape.h"
#include "source_geometry.h"
#include "source_overlay.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

int dv_reference_luma(const dv_intel_composer_config *, uint16_t, uint16_t, int32_t *, int32_t *, uint16_t *);

_Static_assert(sizeof(dv_single_reshape) == 4544, "GPU reshape_mapping layout");

void dvvk_plan_layout(size_t out[6])
{
    out[0] = sizeof(dvvk_plan);
    out[1] = offsetof(dvvk_plan, residual);
    out[2] = offsetof(dvvk_plan, route);
    out[3] = offsetof(dvvk_plan, mapping);
    out[4] = offsetof(dvvk_plan, packets);
    out[5] = offsetof(dvvk_plan, packet_count);
}

static uint32_t checksum(const unsigned char *data, size_t bytes)
{
    uint32_t value = UINT32_MAX;
    for (size_t i = 0; i < bytes; ++i) {
        value ^= (uint32_t)data[i] << 24;
        for (unsigned bit = 0; bit < 8; ++bit)
            value = (value << 1) ^ ((value >> 31) ? UINT32_C(0x04c11db7) : 0u);
    }
    return value;
}

/* dv_source_renderer.c packetize(), unchanged. */
static unsigned packetize(const unsigned char *dm, size_t bytes, unsigned id, unsigned char out[640])
{
    unsigned count = bytes <= 119 ? 1u : 1u + (unsigned)((bytes - 119u + 120u) / 121u);
    size_t offset = 0;
    memset(out, 0, 640);
    for (unsigned i = 0; i < count; ++i) {
        unsigned char *p = out + i * 128;
        unsigned type = count == 1 ? 0 : i == 0 ? 1 : i + 1 == count ? 3 : 2;
        p[0] = (unsigned char)(type << 6);
        p[1] = (unsigned char)((id << 4) | id);
        size_t start = i ? 3 : 5, space = i ? 121 : 119, take = bytes - offset;
        if (take > space)
            take = space;
        if (!i) {
            p[3] = (unsigned char)(bytes >> 8);
            p[4] = (unsigned char)bytes;
        }
        memcpy(p + start, dm + offset, take);
        offset += take;
        uint32_t crc = checksum(p, 124);
        for (unsigned b = 0; b < 4; ++b)
            p[124 + b] = (unsigned char)(crc >> (24 - b * 8));
    }
    return count;
}

/* opencl_backend.c take_row_window(). */
static void row_window(unsigned req_first, unsigned req_last, unsigned h, uint32_t out[2])
{
    out[0] = out[1] = 0;
    unsigned first = req_first, last = req_last;
    if (last <= first || first >= h)
        return;
    first &= ~7u;
    last = last > h - (h % 8u) ? h : (last + 7u) & ~7u;
    if (last > h)
        last = h;
    if (!first && last == h)
        return;
    out[0] = first;
    out[1] = last;
}

/* reconstruct_impl's table preparation and kernel choice, direct route. */
static int plan_dual(const dv_frame_settings *s, int contract, dvvk_plan *o)
{
    const dv_intel_composer_config *cfg = &s->composer;
    if (cfg->el_spatial_resampling_flag > 1 || s->el_scaler > 1 || s->chroma_phase > 1)
        return DV_UNSUPPORTED;
    o->equal = !cfg->el_spatial_resampling_flag;
    o->el_scale = o->equal ? 1 : 2;
    o->nearest = !s->el_scaler;
    /* 2: none. CB1 reshapes the decoded chroma as it is and enlarges the EL's 64/192. */
    o->chroma_phase = contract ? 2u : s->chroma_phase;
    int32_t mapped, residual;
    uint16_t unused;
    for (unsigned i = 0; i < 1024; ++i) {
        if (dv_reference_luma(cfg, (uint16_t)i, (uint16_t)cfg->component[0].nlq_offset, &mapped, &residual, &unused))
            return DV_UNSUPPORTED;
        o->mapping[i] = (uint16_t)mapped;
        if (dv_reference_luma(cfg, 0, (uint16_t)i, &mapped, &o->residual[0][i], &unused))
            return DV_UNSUPPORTED;
    }
    for (unsigned c = 1; c < 3; ++c) {
        if (!cfg->component[c].mapping) {
            if (dv_prepare_polynomial_component(cfg, c, o->chroma_mapping[c - 1], o->residual[c]))
                return DV_UNSUPPORTED;
            memset(o->mmr_params[c - 1], 0, sizeof(o->mmr_params[c - 1]));
        } else if (dv_mmr_prepare(cfg, c, o->mmr_params[c - 1], o->residual[c]) != DV_OK)
            return DV_UNSUPPORTED;
    }
    int lut[2] = {0, 0};
    for (unsigned c = 0; c < 2; ++c)
        if (cfg->component[c + 1].mapping && dv_mmr64_eligible(o->mmr_params[c], NULL)) {
            if (!dv_mmr64_build_lut(o->mmr_params[c], o->mmr_lut[c]))
                return DV_UNSUPPORTED;
            lut[c] = 1;
        }
    o->dual = cfg->component[1].mapping && cfg->component[2].mapping && lut[0] && lut[1] &&
              !memcmp(o->mmr_params[0] + 2, o->mmr_params[1] + 2, 6 * sizeof(uint32_t));
    o->horizontal_fused = !o->equal && o->dual;
    for (unsigned c = 0; c < 2; ++c) {
        const uint32_t *p = o->mmr_params[c];
        if (!cfg->component[c + 1].mapping)
            o->chroma_kind[c] = DVVK_CHROMA_POLY;
        else if (lut[c])
            o->chroma_kind[c] = DVVK_CHROMA_MMR64_LUT;
        else {
            /* mmr96_eligible() from opencl_backend.c */
            int e = !(p[0] < 1 || p[0] > 3 || p[1] < 15 || p[1] > 32);
            for (unsigned k = 0; e && k < 3; k++)
                if (p[2 + 2 * k] > p[3 + 2 * k] || p[3 + 2 * k] > 1023)
                    e = 0;
            if (e) {
                uint32_t sign = p[10] & 0x80000u ? UINT32_MAX : 0;
                if ((p[8] & 0xfffffu) || p[11] != sign || (p[10] & 0xfff00000u) != (sign & 0xfff00000u))
                    e = 0;
                for (unsigned k = 0; e && k < 7 * p[0]; k++) {
                    const uint32_t *v = p + 12 + 4 * k;
                    sign = v[1] & 0x80000000u ? UINT32_MAX : 0;
                    if (v[2] != sign || v[3] != sign)
                        e = 0;
                }
            }
            o->chroma_kind[c] = e ? DVVK_CHROMA_MMR96 : DVVK_CHROMA_MMR128;
        }
    }
    return DV_OK;
}

int dvvk_plan_frame(const dvvk_plan_in *in, const void *metadata, size_t metadata_size, dvvk_plan *o)
{
    if (!in || !o)
        return DV_INVALID;
    memset(o, 0, sizeof(*o));
    dv_source_geometry g = {
        .width = in->width, .height = in->height, .canvas_width = 3840, .canvas_height = 2160,
        .x = in->x, .y = in->y,
        .chroma_siting = (in->cb1_contract || !in->has_el) && in->chroma_top_left ? DV_SOURCE_CHROMA_TOP_LEFT
                                                                                  : DV_SOURCE_CHROMA_CENTER_LEFT,
        .destination_width = in->dw, .destination_height = in->dh,
    };
    unsigned w = in->width, h = in->height, depth = in->nv12 ? 8 : 10;
    /* DV_SINGLE_FULL_HEIGHT; CB1 reshapes at 4:2:0 and resamples the result */
    int contract = in->cb1_contract != 0, nv12 = in->nv12 != 0, full = !nv12 && !contract;
    if (!dv_source_geometry_valid(&g))
        return DV_INVALID;
    unsigned dw = dv_source_destination_width(&g), dh = dv_source_destination_height(&g);
    int resized = dw != w || dh != h;
    dv_frame_settings settings;
    dv_single_frame_settings general;
    memset(&general, 0, sizeof(general));
    unsigned char dm[512];
    size_t bytes = 0;
    dv_identity id = {0, in->pts};
    int status = dv_transport_settings_from_ffmpeg(metadata, metadata_size, w, h, id, &settings, dm, &bytes);
    int single = !in->has_el, use_piecewise = 0;
    if (status == DV_OK) {
        settings.el_scaler = in->el_scaler;
        if (!single && !in->fel && settings.composer.residual_enabled) {
            dv_intel_composer_config *cfg = &settings.composer;
            cfg->residual_enabled = 0;
            for (unsigned c = 0; c < 3; ++c) {
                dv_component_config *v = &cfg->component[c];
                memset(v, 0, sizeof(*v));
                v->pivot_count = 2;
                v->pivots[1] = 1023;
                v->polynomial_order[0] = 1;
                v->polynomial[0][1].integer = 1;
            }
            single = 1;
        }
    }
    int polynomial = status == DV_OK;
    if (polynomial)
        for (unsigned c = 0; c < 3; ++c)
            if (settings.composer.component[c].mapping)
                polynomial = 0;
    if (polynomial && !settings.composer.residual_enabled)
        single = 1;
    if ((single || status == DV_UNSUPPORTED) && !polynomial) {
        if (!full && !contract && g.chroma_siting != DV_SOURCE_CHROMA_CENTER_LEFT)
            return DV_UNSUPPORTED;
        if (full && resized)
            return DV_UNSUPPORTED;
        status = dv_single_settings_from_ffmpeg_for_depth(metadata, metadata_size, w, h, depth, id, &general, dm, &bytes);
        if (status)
            return status;
        single = 1;
        uint32_t masks[3];
        if (dv_single_reshape_surface_masks(&general.mapping, depth, masks))
            return DV_UNSUPPORTED;
        memset(&settings, 0, sizeof(settings));
        settings.abi_version = DV_ENGINE_ABI;
        settings.identity = general.identity;
        settings.source = general.source;
        memcpy(settings.active, general.active, sizeof(settings.active));
        use_piecewise = 1;
        o->layout = full ? 1u : 0u;
        o->sampling = full ? g.chroma_siting : 0u;
    }
    if (status)
        return status;
    if (nv12 && !use_piecewise)
        return DV_UNSUPPORTED;
    if (single && settings.composer.residual_enabled)
        return DV_UNSUPPORTED;
    if (!single && !contract && g.chroma_siting != DV_SOURCE_CHROMA_CENTER_LEFT)
        return DV_UNSUPPORTED;
    if (dv_source_place_metadata(&g, dm, &bytes))
        return DV_UNSUPPORTED;
    for (unsigned c = 0; c < 3; ++c)
        if ((settings.source.offset[c] & 65535u) || (settings.source.offset[c] >> 16) > 4095)
            return DV_UNSUPPORTED;

    /* DV_RECON_ACTIVE_ROWS (not for NV12) */
    if (!nv12 && settings.active[1] < settings.active[3]) {
        unsigned margin = 8 + 4 * ((h + dh - 1) / dh);
        unsigned first = settings.active[1] > margin ? settings.active[1] - margin : 0, last = settings.active[3] + margin;
        row_window(first, last < h ? last : h, h, &o->row_first);
    }

    o->width = w;
    o->height = h;
    o->depth = depth;
    if (use_piecewise) {
        o->route = DVVK_ROUTE_PIECEWISE;
        if (o->layout > 1 || (depth == 8 && o->layout))
            return DV_UNSUPPORTED;
        uint32_t mask[3];
        if (dv_single_reshape_surface_masks(&general.mapping, depth, mask) ||
            memcmp(mask, general.narrow_mask, sizeof(mask)))
            return DV_UNSUPPORTED;
        const dv_source_dm *sd = &general.source;
        if (sd->compressed || sd->depth != 12 || (sd->space != 0 && sd->space != 2) || sd->eotf != 65535 ||
            sd->full_range != 1 || general.active[0] >= general.active[2] || general.active[1] >= general.active[3] ||
            general.active[2] > w || general.active[3] > h)
            return DV_UNSUPPORTED;
        memcpy(o->narrow_mask, mask, sizeof(mask));
        o->order2 = o->layout == 1 && depth == 10 && dv_single_reshape_order2_chroma(&general.mapping);
        if (dv_single_reshape_luma_table_for_depth(&general.mapping, depth, o->luma_table, (size_t)1 << depth))
            return DV_UNSUPPORTED;
        memcpy(o->reshape, &general.mapping, sizeof(general.mapping));
        if (nv12)
            o->row_first = o->row_last = 0;
    } else if (single) {
        o->route = DVVK_ROUTE_SINGLE;
        if (settings.chroma_phase || settings.composer.residual_enabled)
            return DV_UNSUPPORTED;
        if (dv_prepare_polynomial_tables(&settings.composer, o->single_table))
            return DV_UNSUPPORTED;
    } else {
        o->route = DVVK_ROUTE_DUAL;
        if ((status = plan_dual(&settings, contract, o)) != DV_OK)
            return status;
    }
    /* source_identity()'s gate for the tunnel output */
    if (!use_piecewise) {
        const dv_source_dm *sd = &settings.source;
        if (sd->compressed || sd->depth != 12 || (sd->space != 0 && sd->space != 2) || sd->eotf != 65535 ||
            settings.active[0] >= settings.active[2] || settings.active[1] >= settings.active[3] ||
            settings.active[2] > w || settings.active[3] > h)
            return DV_UNSUPPORTED;
    }

    /* present_internal() */
    if (in->packet_id > 15)
        return DV_INVALID;
    if (in->refresh)
        dm[1] = 1;
    if (dv_source_overlay_metadata(dm, bytes, in->overlay != 0, in->overlay_visible_before))
        return DV_INVALID;
    /* CB1: the id advances only when the payload changes (one engine per process). */
    static dv_packet_sequence sequence;
    unsigned packet = in->cb1_contract ? dv_packet_sequence_next(&sequence, dm, bytes) : in->packet_id;
    o->packet_count = packetize(dm, bytes, packet, o->packets);
    int full_pack = use_piecewise && o->layout == 1;
    if (full_pack && resized)
        return DV_UNSUPPORTED;
    o->pack = full_pack ? DVVK_PACK_422 : resized ? DVVK_PACK_RESIZED : DVVK_PACK_SITED;
    if (contract && o->pack != DVVK_PACK_SITED)
        return DV_UNSUPPORTED;   /* CB1 doesn't call resized Master output exact */
    o->chroma_filter = contract ? 1u : 0u;
    if (dv_source_map_active(&g, dw, dh, settings.active, o->active))
        return DV_INVALID;
    for (unsigned c = 0; c < 3; ++c)
        o->black[c] = settings.source.offset[c] >> 16;
    o->placement[0] = g.x;
    o->placement[1] = g.y;
    o->placement[2] = dw;
    o->placement[3] = dh;
    o->top_left = g.chroma_siting;
    o->source_w = w;
    o->source_h = h;
    o->overlay_visible = in->overlay != 0;
    if (in->overlay) {
        /* prepare_overlay() in dv_source_renderer.c */
        dv_overlay_colour colour;
        if (dv_overlay_colour_init(&settings.source, in->rgb_to_lms, in->white_nits, &colour))
            return DV_UNSUPPORTED;
        float *coeff = o->gui_coeff;
        for (unsigned i = 0; i < 9; ++i) {
            coeff[i] = (float)colour.source.inverse_ycc[i];
            coeff[i + 9] = (float)colour.source.inverse_lms[i];
            coeff[i + 18] = (float)colour.rgb_to_lms[i];
        }
        for (unsigned i = 0; i < 3; ++i)
            coeff[i + 27] = (float)colour.source.offset[i];
        coeff[30] = (float)colour.white_nits;
        for (unsigned i = 0; i < 31; ++i)
            if (!isfinite(coeff[i]))
                return DV_UNSUPPORTED;
    }
    return DV_OK;
}
