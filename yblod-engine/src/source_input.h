#ifndef DV_SOURCE_INPUT_H
#define DV_SOURCE_INPUT_H
#include <stdint.h>
enum {DV_SOURCE_INPUT_P010=1,DV_SOURCE_INPUT_NV12=2};
typedef struct {uint32_t size,format,bit_depth;} dv_source_input;
#endif
