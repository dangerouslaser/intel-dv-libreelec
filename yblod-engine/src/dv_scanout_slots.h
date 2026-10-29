#ifndef DV_SCANOUT_SLOTS_H
#define DV_SCANOUT_SLOTS_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
enum {DV_SCANOUT_BUSY=1};
typedef enum {DV_SCANOUT_REJECTED=0,DV_SCANOUT_ACCEPTED=1,DV_SCANOUT_UNCERTAIN=2} dv_scanout_outcome;
typedef struct {uint64_t frame_id;int64_t pts;} dv_scanout_identity;
typedef struct {
    uint64_t serial;
    dv_scanout_identity identity;
    unsigned slot,packet_id,refresh;
} dv_scanout_lease;
typedef struct {
    uint64_t serial;
    dv_scanout_lease pending;
    int current,writing,ready,quarantined;
    unsigned packet_id,refresh;
} dv_scanout_slots;
/* Single render-thread bookkeeping, no allocation or implicit GPU/KMS waits.
 * Exactly two externally owned BOs. Initialize once before either is used. */
void dv_scanout_slots_init(dv_scanout_slots *);
int dv_scanout_slots_acquire(dv_scanout_slots *,dv_scanout_identity,dv_scanout_lease *);
/* ACCEPTED certifies GPU completion AND release of producer ownership.
 * REJECTED certifies all attempted writes have drained. Otherwise UNCERTAIN. */
int dv_scanout_slots_written(dv_scanout_slots *,const dv_scanout_lease *,dv_scanout_outcome);
/* ACCEPTED means a successful BLOCKING replacement commit, not a queued flip.
 * REJECTED means old scanout is unchanged. Unknown outcomes quarantine both BOs. */
int dv_scanout_slots_committed(dv_scanout_slots *,const dv_scanout_lease *,dv_scanout_outcome);
/* Stream reset retains current display ownership. Pending writes must first
 * finish/drain; ready but uncommitted output may be discarded. */
int dv_scanout_slots_reset_stream(dv_scanout_slots *);
/* Caller certifies external scanout was restored and all GPU work drained.
 * Preserves serial so pre-reset leases cannot authorize a later frame. */
void dv_scanout_slots_restored(dv_scanout_slots *);
int dv_scanout_slots_can_destroy(const dv_scanout_slots *);
#ifdef __cplusplus
}
#endif
#endif
