#include "single_reshape.h"
#include <assert.h>
#include <stdio.h>
int main(void)
{
    dv_single_reshape m={0};m.base_depth=10;m.denominator=23;
    for(unsigned c=0;c<3;++c){m.curve[c].pivot_count=2;m.curve[c].pivots[1]=1023;
        m.curve[c].piece[0].order=c?2u:1u;m.curve[c].piece[0].method=c?1u:0u;
        m.curve[c].piece[0].coefficient[0][c?c:0]=1<<23;}
    assert(dv_single_reshape_order2_chroma(&m));unsigned checks=1;
#define REJECT(change) do{dv_single_reshape bad=m;change;assert(!dv_single_reshape_order2_chroma(&bad));++checks;}while(0)
    assert(!dv_single_reshape_order2_chroma(NULL));++checks;
    REJECT(bad.base_depth=8);REJECT(bad.denominator=33);
    REJECT(bad.curve[0].piece[0].method=1);
    for(unsigned c=1;c<3;++c){
        REJECT(bad.curve[c].pivot_count=3;bad.curve[c].pivots[1]=512;bad.curve[c].pivots[2]=1023);
        REJECT(bad.curve[c].piece[0].method=0);
        REJECT(bad.curve[c].piece[0].order=1);
        REJECT(bad.curve[c].piece[0].order=3);
        REJECT(bad.curve[c].piece[0].constant=INT64_MIN);
        REJECT(bad.curve[c].piece[0].coefficient[1][6]=INT64_MAX);
        REJECT(bad.curve[c].pivots[1]=0);
    }
    m.curve[0].pivot_count=3;m.curve[0].pivots[1]=512;m.curve[0].pivots[2]=1023;
    m.curve[0].piece[1].order=2;
    assert(dv_single_reshape_order2_chroma(&m));++checks;
    printf("order2 structural gate: %u checks passed\n",checks);
}
