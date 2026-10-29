/* Which route and kernels the engine takes for each frame of a file (yblod's
 * host code via yblod/plan.c), headless:  dvvk-routes FILE [FRAMES] [scaler] */
#include "decode.h"
#include "../yblod/plan.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *route_name(unsigned r)
{
    return r == DVVK_ROUTE_DUAL ? "dual" : r == DVVK_ROUTE_SINGLE ? "single" : r == DVVK_ROUTE_PIECEWISE ? "piecewise" : "?";
}

int main(int argc, char **argv)
{
    if (argc < 2)
        return 2;
    int frames = argc > 2 ? atoi(argv[2]) : 24;
    unsigned scaler = argc > 3 ? (unsigned)atoi(argv[3]) : 0;
    struct dv_decoder d;
    if (dv_decoder_open(&d, argv[1], "/dev/dri/renderD128", 0) < 0)
        return 1;
    static dvvk_plan plan;
    char last[512] = "";
    int n = 0, repeats = 0;
    struct dv_pair p;
    while (n < frames && dv_decoder_next(&d, &p) == 1) {
        unsigned w = (unsigned)p.bl->width, h = (unsigned)p.bl->height;
        /* mpv-yblod's placement: centred, even, scaled to fit 3840x2160 */
        unsigned dw = w, dh = h;
        if (w * 2160 > h * 3840 ? w != 3840 : h != 2160) {
            if (w * 2160 >= h * 3840) { dw = 3840; dh = (h * 3840 / w) & ~3u; }
            else { dh = 2160; dw = (w * 2160 / h) & ~3u; }
        }
        dvvk_plan_in in = {.width = w, .height = h, .x = ((3840 - dw) / 2) & ~1u, .y = ((2160 - dh) / 2) & ~1u,
                           .dw = dw != w || dh != h ? dw : 0, .dh = dw != w || dh != h ? dh : 0,
                           .has_el = p.el != NULL, .el_scaler = scaler, .fel = 1, .pts = p.bl->pts};
        int r = p.dovi ? dvvk_plan_frame(&in, p.dovi->data, p.dovi->size, &plan) : -100;
        char line[512];
        if (r)
            snprintf(line, sizeof(line), "status %d (%ux%u%s)", r, w, h, p.el ? " +EL" : "");
        else
            snprintf(line, sizeof(line),
                     "%s %ux%u el%s scale %u eq %u chroma [%u,%u] dual %u hfused %u layout %u order2 %u pack %u "
                     "rows [%u,%u) active [%u,%u,%u,%u] place [%u,%u,%u,%u] packets %u",
                     route_name(plan.route), w, h, p.el ? "+" : "-", plan.el_scale, plan.equal, plan.chroma_kind[0],
                     plan.chroma_kind[1], plan.dual, plan.horizontal_fused, plan.layout, plan.order2, plan.pack,
                     plan.row_first, plan.row_last, plan.active[0], plan.active[1], plan.active[2], plan.active[3],
                     plan.placement[0], plan.placement[1], plan.placement[2], plan.placement[3], plan.packet_count);
        if (strcmp(line, last)) {
            if (repeats)
                printf("  (x%d)\n", repeats + 1);
            printf("%s: %s\n", argv[1], line);
            strcpy(last, line);
            repeats = 0;
        } else
            repeats++;
        dv_pair_free(&p);
        n++;
    }
    if (repeats)
        printf("  (x%d)\n", repeats + 1);
    dv_decoder_close(&d);
    return 0;
}
