#include "dv_scanout_slots.h"
#include "dv_engine.h"
#include <string.h>
void dv_scanout_slots_init(dv_scanout_slots *p)
{memset(p,0,sizeof(*p));p->current=-1;p->refresh=1;}
static int matches(const dv_scanout_slots *p,const dv_scanout_lease *l)
{
    return l&&l->serial==p->pending.serial&&l->slot==p->pending.slot&&
        l->identity.frame_id==p->pending.identity.frame_id&&l->identity.pts==p->pending.identity.pts&&
        l->packet_id==p->pending.packet_id&&l->refresh==p->pending.refresh;
}
int dv_scanout_slots_acquire(dv_scanout_slots *p,dv_scanout_identity id,dv_scanout_lease *out)
{
    if(!p||!out)return DV_INVALID;
    if(p->quarantined)return DV_BACKEND;
    if(p->writing||p->ready)return DV_SCANOUT_BUSY;
    if(p->serial==UINT64_MAX)return DV_BACKEND;
    dv_scanout_lease lease={++p->serial,id,p->current==0?1u:0u,p->packet_id,p->refresh};
    p->pending=lease;p->writing=1;*out=lease;return DV_OK;
}
int dv_scanout_slots_written(dv_scanout_slots *p,const dv_scanout_lease *l,dv_scanout_outcome result)
{
    if(!p||result<DV_SCANOUT_REJECTED||result>DV_SCANOUT_UNCERTAIN)return DV_INVALID;
    if(p->quarantined)return DV_BACKEND;
    if(!p->writing||!matches(p,l))return DV_IDENTITY;
    if(result==DV_SCANOUT_UNCERTAIN){p->quarantined=1;return DV_BACKEND;}
    p->writing=0;p->ready=result==DV_SCANOUT_ACCEPTED;
    if(!p->ready)p->refresh=1;
    return DV_OK;
}
int dv_scanout_slots_committed(dv_scanout_slots *p,const dv_scanout_lease *l,dv_scanout_outcome result)
{
    if(!p||result<DV_SCANOUT_REJECTED||result>DV_SCANOUT_UNCERTAIN)return DV_INVALID;
    if(p->quarantined)return DV_BACKEND;
    if(!p->ready||!matches(p,l))return DV_IDENTITY;
    if(result==DV_SCANOUT_UNCERTAIN){p->quarantined=1;return DV_BACKEND;}
    p->ready=0;
    if(result==DV_SCANOUT_ACCEPTED){p->current=(int)l->slot;p->packet_id=(p->packet_id+1)&15u;p->refresh=0;}
    else p->refresh=1;
    return DV_OK;
}
int dv_scanout_slots_reset_stream(dv_scanout_slots *p)
{
    if(!p)return DV_INVALID;
    if(p->quarantined)return DV_BACKEND;
    if(p->writing)return DV_SCANOUT_BUSY;
    p->ready=0;p->packet_id=0;p->refresh=1;return DV_OK;
}
void dv_scanout_slots_restored(dv_scanout_slots *p)
{uint64_t serial=p->serial;dv_scanout_slots_init(p);p->serial=serial;}
int dv_scanout_slots_can_destroy(const dv_scanout_slots *p)
{return p&&p->current<0&&!p->writing&&!p->ready&&!p->quarantined;}
