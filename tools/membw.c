// Multi-threaded streaming read / copy bandwidth test (aarch64 NEON). usage: membw <threads> <cpu_list_csv> [MB]
#define _GNU_SOURCE
#include <arm_neon.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static size_t N; static uint8_t *buf, *dst; static int nth; static int cpus[16]; static volatile int sink;
static double now(){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
static int mode; static pthread_barrier_t bar; static double tmax;
static void*work(void*a){long id=(long)a; cpu_set_t s;CPU_ZERO(&s);CPU_SET(cpus[id%16],&s);sched_setaffinity(0,sizeof s,&s);
  size_t per=N/nth; uint8_t*p=buf+id*per,*q=dst+id*per; uint64x2_t acc=vdupq_n_u64(0);
  for(int it=0;it<6;it++){ pthread_barrier_wait(&bar); double t0=now();
    if(mode==0){ for(size_t i=0;i<per;i+=128){ uint64x2_t a0=vld1q_u64((uint64_t*)(p+i)),a1=vld1q_u64((uint64_t*)(p+i+16)),a2=vld1q_u64((uint64_t*)(p+i+32)),a3=vld1q_u64((uint64_t*)(p+i+48)),a4=vld1q_u64((uint64_t*)(p+i+64)),a5=vld1q_u64((uint64_t*)(p+i+80)),a6=vld1q_u64((uint64_t*)(p+i+96)),a7=vld1q_u64((uint64_t*)(p+i+112)); acc=veorq_u64(acc,veorq_u64(veorq_u64(veorq_u64(a0,a1),veorq_u64(a2,a3)),veorq_u64(veorq_u64(a4,a5),veorq_u64(a6,a7)))); } }
    else memcpy(q,p,per);
    pthread_barrier_wait(&bar); double dt=now()-t0; if(id==0&&it>0){ if(dt>0){double bw=(mode==0?1.0:2.0)*N/dt/1e9; if(bw>tmax)tmax=bw;} } }
  sink+=vgetq_lane_u64(acc,0); return 0;}
int main(int c,char**v){ nth=atoi(v[1]); char*l=v[2]; int k=0; for(char*t=strtok(l,",");t&&k<16;t=strtok(0,","))cpus[k++]=atoi(t); N=(c>3?atol(v[3]):1024)*1024UL*1024UL;
  buf=aligned_alloc(4096,N);dst=aligned_alloc(4096,N);memset(buf,1,N);memset(dst,2,N);
  for(mode=0;mode<2;mode++){ tmax=0; pthread_barrier_init(&bar,0,nth); pthread_t th[16]; for(long i=0;i<nth;i++)pthread_create(&th[i],0,work,(void*)i); for(int i=0;i<nth;i++)pthread_join(th[i],0); pthread_barrier_destroy(&bar);
    printf("%s threads=%d cpus=%s : %.1f GB/s (best)\n",mode==0?"READ ":"COPY(r+w)",nth,v[2],tmax);} return sink&1;}
