// AMX-BF16 2x2 blocking microbench: row-major A (stride K*2) vs packed A tiles; single thread or N threads.
#include <immintrin.h>
#include <x86intrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <sys/syscall.h>
#include <unistd.h>
static void cfg(void){ struct __attribute__((aligned(64))) {uint8_t p,s,r[14];uint16_t c[16];uint8_t rows[16];} c; memset(&c,0,64); c.p=1; for(int t=0;t<8;t++){c.rows[t]=16;c.c[t]=64;} _tile_loadconfig(&c);}
static int64_t M=512,P=128,K=4096,KCH=32; static int packed=0, reps=5, sub=1; static int inner=1;
typedef struct { uint16_t *A,*B; float *C; double mpc; } arg_t;
static void run(uint16_t*A,uint16_t*B,float*cst){
  int64_t nt=K/32, nmg=M/32, npair=P/32; float subb[2][1024] __attribute__((aligned(64)));
  for(int64_t mg=0;mg<nmg;mg++){
    for(int64_t j=0;j<npair;j++){
      const uint16_t *b0=B+(2*j)*nt*512, *b1=B+(2*j+1)*nt*512;
      for(int64_t s0=0,si=0;s0<nt;s0+=KCH,si++){
        int64_t s1=s0+KCH<nt?s0+KCH:nt;
        if(s0==0){_tile_zero(0);_tile_zero(1);_tile_zero(2);_tile_zero(3);} else if(sub){float*in=subb[(si+1)&1];_tile_loadd(0,in,64);_tile_loadd(2,in+512,64);_tile_loadd(1,in+256,64);_tile_loadd(3,in+768,64);}
        if(!packed){ const uint16_t*a0=A+mg*32*K, *a1=a0+16*K;
          for(int64_t t=s0;t<s1;t++){ _tile_loadd(4,a0+t*32,K*2); _tile_loadd(6,b0+t*512,64); _tile_loadd(5,a1+t*32,K*2); _tile_loadd(7,b1+t*512,64);
            _tile_dpbf16ps(0,4,6);_tile_dpbf16ps(1,4,7);_tile_dpbf16ps(2,5,6);_tile_dpbf16ps(3,5,7);} }
        else { const uint16_t*ap=A+mg*nt*1024;  // [mg][t][32 rows][32]
          for(int64_t t=s0;t<s1;t++){ _tile_loadd(4,ap+t*1024,64); _tile_loadd(6,b0+t*512,64); _tile_loadd(5,ap+t*1024+512,64); _tile_loadd(7,b1+t*512,64);
            _tile_dpbf16ps(0,4,6);_tile_dpbf16ps(1,4,7);_tile_dpbf16ps(2,5,6);_tile_dpbf16ps(3,5,7);} }
        if(sub || s1==nt){float*o=s1==nt?cst+(mg*npair+j)*1024:subb[si&1]; _tile_stored(0,o,64);_tile_stored(2,o+512,64);_tile_stored(1,o+256,64);_tile_stored(3,o+768,64);}
      }
    }
  }
}
static void* th(void*p){ arg_t*a=p; syscall(SYS_arch_prctl,0x1023,18); cfg(); double best=1e30;
  for(int r=0;r<reps;r++){uint64_t t0=__rdtsc(); for(int q=0;q<inner;q++) run(a->A,a->B,a->C); uint64_t t1=__rdtsc(); if(t1-t0<best)best=t1-t0;}
  a->mpc=(double)M*P*K*inner/best; return 0;}
int main(int argc,char**argv){ int nth=argc>1?atoi(argv[1]):1; if(argc>2)packed=atoi(argv[2]); if(argc>3)K=atoll(argv[3]); if(argc>4)M=atoll(argv[4]); if(argc>5)P=atoll(argv[5]); if(argc>6)sub=atoi(argv[6]); if(argc>7)KCH=atoll(argv[7]); if(getenv("INNER"))inner=atoi(getenv("INNER"));
  pthread_t t[64]; arg_t a[64];
  for(int i=0;i<nth;i++){ a[i].A=aligned_alloc(64,M*K*2); a[i].B=aligned_alloc(64,P*K*2); a[i].C=aligned_alloc(64,M*P*4+4096);
    for(int64_t x=0;x<M*K;x++)a[i].A[x]=0x3f80; for(int64_t x=0;x<P*K;x++)a[i].B[x]=0x3f80; }
  for(int i=0;i<nth;i++)pthread_create(&t[i],0,th,&a[i]); double s=0; for(int i=0;i<nth;i++){pthread_join(t[i],0); s+=a[i].mpc;}
  printf("nth=%d packed=%d K=%ld M=%ld P=%ld sub=%d kch=%ld: %.0f MAC/TSC-cycle/core (%.2f TMAC/s/core @2.4GHz TSC)\n",nth,packed,K,M,P,sub,KCH,s/nth,s/nth*2.4e9/1e12);
}
