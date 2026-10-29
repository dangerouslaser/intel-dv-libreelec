#include "dv_scanout_slots.h"
#include "dv_engine.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(void)
{
    dv_scanout_slots p;dv_scanout_slots_init(&p);assert(dv_scanout_slots_can_destroy(&p));
    dv_scanout_lease a,b;dv_scanout_identity id={1,41708};
    assert(dv_scanout_slots_acquire(&p,id,&a)==DV_OK&&a.slot==0&&a.refresh&&a.packet_id==0);
    assert(!dv_scanout_slots_can_destroy(&p));
    assert(dv_scanout_slots_reset_stream(&p)==DV_SCANOUT_BUSY&&p.writing);
    memset(&b,0xab,sizeof(b));dv_scanout_lease saved=b;
    assert(dv_scanout_slots_acquire(&p,id,&b)==DV_SCANOUT_BUSY&&!memcmp(&b,&saved,sizeof(b)));
    assert(dv_scanout_slots_committed(&p,&a,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    b=a;++b.identity.pts;assert(dv_scanout_slots_written(&p,&b,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_ACCEPTED)==DV_OK);
    assert(dv_scanout_slots_acquire(&p,id,&b)==DV_SCANOUT_BUSY);
    assert(dv_scanout_slots_committed(&p,&a,DV_SCANOUT_ACCEPTED)==DV_OK&&p.current==0&&p.packet_id==1);
    assert(dv_scanout_slots_committed(&p,&a,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    for(unsigned frame=0;frame<1000;++frame){int current=p.current;unsigned packet=p.packet_id;
        id.frame_id++;id.pts+=41708;
        assert(dv_scanout_slots_acquire(&p,id,&b)==DV_OK&&(int)b.slot!=current&&b.packet_id==packet);
        assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
        assert(dv_scanout_slots_written(&p,&b,DV_SCANOUT_ACCEPTED)==DV_OK);
        dv_scanout_outcome result=frame%3?DV_SCANOUT_ACCEPTED:DV_SCANOUT_REJECTED;
        assert(dv_scanout_slots_committed(&p,&b,result)==DV_OK);
        if(result==DV_SCANOUT_ACCEPTED)assert(p.current==(int)b.slot&&p.packet_id==((packet+1)&15u)&&!p.refresh);
        else assert(p.current==current&&p.packet_id==packet&&p.refresh);
        a=b;
    }
    assert(dv_scanout_slots_acquire(&p,id,&a)==DV_OK);
    int old=p.current;
    assert(dv_scanout_slots_reset_stream(&p)==DV_SCANOUT_BUSY&&p.current==old);
    assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_ACCEPTED)==DV_OK);
    assert(dv_scanout_slots_reset_stream(&p)==DV_OK&&p.current==old&&p.refresh&&p.packet_id==0);
    assert(!dv_scanout_slots_can_destroy(&p));
    assert(dv_scanout_slots_committed(&p,&a,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    assert(dv_scanout_slots_acquire(&p,id,&b)==DV_OK&&(int)b.slot!=old);
    assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    a=b;
    assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_REJECTED)==DV_OK&&p.refresh);
    assert(dv_scanout_slots_acquire(&p,id,&b)==DV_OK);
    assert(dv_scanout_slots_written(&p,&b,DV_SCANOUT_UNCERTAIN)==DV_BACKEND);
    assert(!dv_scanout_slots_can_destroy(&p)&&dv_scanout_slots_acquire(&p,id,&a)==DV_BACKEND);
    assert(dv_scanout_slots_reset_stream(&p)==DV_BACKEND);
    dv_scanout_slots_restored(&p);assert(dv_scanout_slots_can_destroy(&p));
    assert(dv_scanout_slots_acquire(&p,id,&a)==DV_OK&&a.serial>b.serial&&a.refresh&&a.packet_id==0);
    assert(dv_scanout_slots_written(&p,&b,DV_SCANOUT_ACCEPTED)==DV_IDENTITY);
    assert(dv_scanout_slots_written(&p,&a,DV_SCANOUT_ACCEPTED)==DV_OK);
    assert(dv_scanout_slots_committed(&p,&a,DV_SCANOUT_UNCERTAIN)==DV_BACKEND);
    assert(!dv_scanout_slots_can_destroy(&p)&&dv_scanout_slots_acquire(&p,id,&b)==DV_BACKEND);
    dv_scanout_slots_restored(&p);p.serial=UINT64_MAX;
    assert(dv_scanout_slots_acquire(&p,id,&b)==DV_BACKEND&&dv_scanout_slots_can_destroy(&p));
    puts("{\"blocking_scanout_transitions\":1000,\"failure_quarantine\":true,\"stale_lease_rejected\":true,\"passed\":true}");
    return 0;
}
