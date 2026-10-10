#include "source_geometry.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
static unsigned read16(const unsigned char *p){return (unsigned)p[0]*256u+p[1];}
static void scaling_tests(void)
{
    dv_source_geometry g={3840,2080,3840,2160,0,42,DV_SOURCE_CHROMA_CENTER_LEFT,0,0};
    unsigned char dm[512]={0},saved[512];size_t n=71;
    uint32_t full[4]={0,0,3840,2080},mapped[4];
    assert(!dv_source_map_active(&g,3840,2076,full,mapped));
    assert(mapped[0]==0&&mapped[1]==42&&mapped[2]==3840&&mapped[3]==2118);
    assert(!dv_source_place_scaled_metadata(&g,3840,2076,dm,&n));
    assert(n==84&&read16(dm+80)==42&&read16(dm+82)==42);
    unsigned char via_geometry[512]={0};size_t geometry_bytes=71;
    g.destination_width=3840;g.destination_height=2076;
    assert(dv_source_geometry_valid(&g));
    assert(!dv_source_place_metadata(&g,via_geometry,&geometry_bytes));
    assert(geometry_bytes==n&&!memcmp(via_geometry,dm,n));
    g.destination_height=0;assert(!dv_source_geometry_valid(&g));
    g.destination_width=0;
    /* L5 is in decoded coordinates, not the previous output's coordinates. */
    dm[81]=100;dm[83]=100;
    assert(!dv_source_place_scaled_metadata(&g,3840,2076,dm,&n));
    assert(read16(dm+80)==141&&read16(dm+82)==141);
    memcpy(saved,dm,sizeof(dm));size_t old_n=n;
    assert(dv_source_place_scaled_metadata(&g,3840,2078,dm,&n));
    assert(n==old_n&&!memcmp(saved,dm,sizeof(dm)));
    assert(dv_source_place_scaled_metadata(&g,3840,2160,dm,&n));
    assert(n==old_n&&!memcmp(saved,dm,sizeof(dm)));
    const unsigned sizes[]={4,8,20,1080,2076,2080,2160,3840,4096};
    unsigned checks=0;
    for(unsigned si=0;si<sizeof(sizes)/sizeof(sizes[0]);++si)
        for(unsigned di=0;di<sizeof(sizes)/sizeof(sizes[0]);++di){
            unsigned source=sizes[si],dest=sizes[di];
            g=(dv_source_geometry){source,source,4096,4096,0,0,DV_SOURCE_CHROMA_TOP_LEFT,0,0};
            for(unsigned boundary=0;boundary<source;++boundary){
                uint32_t in[4]={boundary,boundary,boundary+1,boundary+1};
                assert(!dv_source_map_active(&g,dest,dest,in,mapped));
                /* Independent interval-containment and minimality oracle. */
                assert(mapped[0]*source<=boundary*dest);
                assert((mapped[0]+1)*source>boundary*dest);
                assert(mapped[2]*source>=(boundary+1)*dest);
                assert((mapped[2]-1)*source<(boundary+1)*dest);
                assert(mapped[0]==mapped[1]&&mapped[2]==mapped[3]);
                assert(mapped[0]<mapped[2]&&mapped[2]<=dest);
                ++checks;
            }
        }
    uint32_t invalid[4]={0,0,4097,4096},unchanged[4]={11,22,33,44};
    memcpy(mapped,unchanged,sizeof(mapped));
    assert(dv_source_map_active(&g,4096,4096,invalid,mapped));
    assert(!memcmp(mapped,unchanged,sizeof(mapped)));
    uint32_t alias[4]={1,2,3,4};
    assert(!dv_source_map_active(&g,4096,4096,alias,alias));
    assert(alias[0]==1&&alias[1]==2&&alias[2]==3&&alias[3]==4);
    printf("{\"scaled_interval_checks\":%u,\"air_geometry\":true,\"transactional_errors\":true}\n",checks);
}
int main(void)
{
    dv_source_geometry g={3840,2076,3840,2160,0,42,DV_SOURCE_CHROMA_CENTER_LEFT,0,0};unsigned char dm[512]={0},saved[512];size_t n=71;
    assert(!dv_source_place_metadata(&g,dm,&n));assert(n==84&&dm[70]==1&&dm[74]==8&&dm[75]==5);
    assert(dm[80]==0&&dm[81]==42&&dm[82]==0&&dm[83]==42);
    dm[81]=2;dm[83]=4;assert(!dv_source_place_metadata(&g,dm,&n));assert(dm[81]==44&&dm[83]==46);
    memcpy(saved,dm,sizeof(dm));size_t previous=n;g.y=43;
    assert(dv_source_place_metadata(&g,dm,&n)&&n==previous&&!memcmp(dm,saved,sizeof(dm)));
    g.y=42;g.width=3844;assert(dv_source_place_metadata(&g,dm,&n));g.width=3840;
    dm[74]=255;memcpy(saved,dm,sizeof(dm));assert(dv_source_place_metadata(&g,dm,&n)&&!memcmp(dm,saved,sizeof(dm)));
    memset(dm,0,sizeof(dm));n=71;g=(dv_source_geometry){3840,2160,3840,2160,0,0,DV_SOURCE_CHROMA_TOP_LEFT,0,0};
    memcpy(saved,dm,sizeof(dm));assert(!dv_source_place_metadata(&g,dm,&n)&&n==71&&!memcmp(dm,saved,sizeof(dm)));
    g.chroma_siting=2;assert(!dv_source_geometry_valid(&g));assert(dv_source_place_metadata(&g,dm,&n)&&n==71&&!memcmp(dm,saved,sizeof(dm)));
    puts("{\"geometry_translation\":true,\"existing_and_missing_L5\":true,\"unchanged_on_rejection\":true,\"identity_unchanged\":true}");
    scaling_tests();
}
