#ifndef DV_PIXEL_POLYNOMIAL_H
#define DV_PIXEL_POLYNOMIAL_H
#include "intel_composer_config.h"
/* Already-aligned component codes; no chroma resampling or colour conversion. */
int dv_reference_polynomial(const dv_intel_composer_config *,unsigned,uint16_t,
                           uint16_t,int32_t *,int32_t *,uint16_t *);
/* Single-layer polynomial mapping. Failure leaves all table entries unchanged. */
int dv_prepare_polynomial_tables(const dv_intel_composer_config *,uint16_t [3][1024]);
#endif
