#ifndef COMMONS_CUH
#define COMMONS_CUH

#include <cuda_runtime.h>

/** 
 * Given the position of two bodies, compute the acceleration of the first body
 * due to the second body. The bodies are represented by the float4 structure
 * in order to take advantage of the coalesced memory access.
 */
inline __device__ void
bodyBodyInteraction(const float4 bi, const float4 bj, float3* ai, const float G, const float softSquared)
{
    float3 r;
    r.x = bj.x - bi.x; // 1 FLOP
    r.y = bj.y - bi.y; // 1 FLOP
    r.z = bj.z - bi.z; // 1 FLOP

    float distSqr = r.x * r.x + r.y * r.y + r.z * r.z ; // 6 FLOPS
    distSqr += softSquared; // 1 FLOP
    
    float s = G * bj.w / (distSqr * sqrtf(distSqr)); // 3 FLOPS
    
    ai->x += r.x * s; // 2 FLOPS
    ai->y += r.y * s; // 2 FLOPS
    ai->z += r.z * s; // 2 FLOPS
}

/**
 * Compute the acceleration of a body due to all the other bodies in the system.
 */
inline __device__ void
tile_calculation(const float4 bi, float3* ai, const float G, const float softSquared)
{
  int j;
  extern __shared__ float4 shPosition[];
  for (j = 0; j < blockDim.x; j+=4) { // loop unrolling of a factor of 4
    bodyBodyInteraction(bi, shPosition[j], ai, G, softSquared);
    bodyBodyInteraction(bi, shPosition[j+1], ai, G, softSquared);
    bodyBodyInteraction(bi, shPosition[j+2], ai, G, softSquared);
    bodyBodyInteraction(bi, shPosition[j+3], ai, G, softSquared);
  }
  /* for (j = 0; j < blockDim.x; j++) {
    bodyBodyInteraction(bi, shPosition[j], ai, G, softSquared);
  } */
}

#endif // COMMONS_CUH