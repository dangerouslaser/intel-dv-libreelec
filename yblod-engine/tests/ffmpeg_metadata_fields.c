/* SPDX-License-Identifier: MIT */
#include <libavutil/dovi_meta.h>
#include <stddef.h>

/* The renderer and decoder must agree on the appended raw-block contract.
 * Merely finding the member names does not establish their widths/layout.
 * These compile-time checks remain active in Release builds with NDEBUG. */
_Static_assert(sizeof(((AVDOVIDmData *)0)->dvbridge_raw_magic) == 4,
               "raw metadata magic must be 32 bits");
_Static_assert(sizeof(((AVDOVIDmData *)0)->dvbridge_original_length) == 4,
               "raw metadata length must be 32 bits");
_Static_assert(sizeof(((AVDOVIDmData *)0)->dvbridge_original_bytes) == 32,
               "raw metadata payload must hold 32 bytes");
_Static_assert(offsetof(AVDOVIDmData, dvbridge_original_length) ==
               offsetof(AVDOVIDmData, dvbridge_raw_magic) + 4,
               "unexpected raw metadata trailer layout");
_Static_assert(offsetof(AVDOVIDmData, dvbridge_original_bytes) ==
               offsetof(AVDOVIDmData, dvbridge_raw_magic) + 8,
               "unexpected raw metadata payload offset");
_Static_assert(sizeof(AVDOVIDmData) >=
               offsetof(AVDOVIDmData, dvbridge_original_bytes) + 32,
               "raw metadata payload exceeds block size");
int main(void)
{
    AVDOVIDmData d = {0};
    d.dvbridge_raw_magic = 0x41424456;
    d.dvbridge_original_length = sizeof(d.dvbridge_original_bytes);
    return d.dvbridge_original_bytes[0];
}
