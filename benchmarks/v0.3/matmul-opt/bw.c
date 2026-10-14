#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include <time.h>
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+1e-9*t.tv_nsec;}
int main(int c,char**v){size_t n=(size_t)4<<30; char*b=aligned_alloc(64,n); memset(b,1,n);
 for(int r=0;r<5;r++){double t=now(); long acc=0;
 #pragma omp parallel reduction(+:acc)
 { int id=omp_get_thread_num(),nt=omp_get_num_threads(); size_t a=n/nt*id,e=n/nt*(id+1); __m512i x0=_mm512_setzero_si512(),x1=x0,x2=x0,x3=x0;
   for(size_t i=a;i<e;i+=256){ x0=_mm512_add_epi32(x0,_mm512_load_si512(b+i)); x1=_mm512_add_epi32(x1,_mm512_load_si512(b+i+64)); x2=_mm512_add_epi32(x2,_mm512_load_si512(b+i+128)); x3=_mm512_add_epi32(x3,_mm512_load_si512(b+i+192)); }
   acc+=_mm512_reduce_add_epi32(_mm512_add_epi32(_mm512_add_epi32(x0,x1),_mm512_add_epi32(x2,x3))); }
 double dt=now()-t; printf("read %.1f GB/s (%d)\n", n/dt/1e9, (int)(acc&1));}
}
