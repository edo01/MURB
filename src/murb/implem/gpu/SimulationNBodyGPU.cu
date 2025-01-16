#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyGPU.hpp"

/**
 * The GPU implementation of the N-Body simulation is based on the O(N^2) algorithm.
 * If we consider that for each body we must compute the acceleration with all the other bodies,
 * we can see that there is a grid of nBodies x nBodies computations to be done. This grid can be 
 * divided into tiles of size blockDim x blockDim. Each tile will be computed by a block of threads.
 * The algorithm is as follows:
 * - First the positions of blockDim bodies are loaded into the shared memory. This will be resided in the
 *   shared memory of the block and are the blocks for which the acceleration will be computed.
 * - Then other blockDim bodies are loaded into the shared memory. These bodies are the ones for which the
 *   acceleration will be computed.
 * - the result of the computation is accumulated in the shared memory of the block.
 * - New bodies are loaded into the shared memory and the process is repeated until all the bodies are computed.
 * - The final result is written into the global memory.
 */

/** 
 * Given the position of two bodies, compute the acceleration of the first body
 * due to the second body. The bodies are represented by the float4 structure
 * in order to take advantage of the coalesced memory access. ????
 */
__device__ float3
bodyBodyInteraction(float4 bi, float4 bj, float3 ai)
{
  float3 r;
  r.x = bj.x - bi.x; // 1 FLOP
  r.y = bj.y - bi.y; // 1 FLOP
  r.z = bj.z - bi.z; // 1 FLOP

  float distSqr = r.x * r.x + r.y * r.y + r.z * r.z + EPS2; // 6 FLOPS

  float s = bj.w / distSqr*sqrtf(distSqr); // 3 FLOPS

  ai.x += r.x * s; // 2 FLOPS
  ai.y += r.y * s; // 2 FLOPS
  ai.z += r.z * s; // 2 FLOPS
  return ai;
}

/**
 * Compute the acceleration of a body due to all the other bodies in the system.

 */
__device__ float3
tile_calculation(float4 bi, float3 ai)
{
  int j;
  extern __shared__ float4[] shPosition;
  for (j = 0; j < blockDim.x; j++) {
    ai = bodyBodyInteraction(bi, shPosition[j], ai);
  }
  return ai;
}

__global__ void
calculate_forces(float* devX, float* devY, float* devZ, float* devM, const unsigned long N, 
                 const float softSquared, const float G, float* devAx, float* devAy, float* devAz)
{
  /* extern __shared__ float4[] shPosition;
  float4 *globalX = (float4 *)devX;
  float4 *globalA = (float4 *)devA;
  float4 bi;
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};
  int gtid = blockIdx.x * blockDim.x + threadIdx.x;
  bi = globalX[gtid];
  for (i = 0, tile = 0; i < N; i += p, tile++) {
    int idx = tile * blockDim.x + threadIdx.x;
    shPosition[threadIdx.x] = globalX[idx];
    __syncthreads();
    acc = tile_calculation(myPosition, acc);
    __syncthreads();
  }
  // Save the result in global memory for the integration step.
  float4 acc4 = {acc.x, acc.y, acc.z, 0.0f};
  globalA[gtid] = acc4; */

  extern __shared__ float4 shPosition[]; // shared memory to store the positions of the bodies
  int i, tile;
  float3 acc = {0.0f, 0.0f, 0.0f};

  for (i = 0, tile = 0; i < N; i += p, tile++) {
    
  }


}

//maybe for the cache is better this version
/*__global__
void kernelComputeBodiesAcceleration(const dataAoS_t<float>* __restrict__ d, const unsigned long N, 
                                     const float softSquared, const float G,
                                     accAoS_t<float>* accelerations ){
    // The strategy is to let each thread compute the acceleration of one body
    // THe inner loop will be executed by each thread. We must ensure that the data are fetched 
    // only once and are broadcasted to all threads in the block. In order to do that, 

    // Get the index of the body to compute the acceleration
    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    // skip if the body index is out of range
    if(iBody >= N) return;

    float accX = 0.f, accY = 0.f, accZ = 0.f;

    for (unsigned long jBody = 0; jBody < N; jBody++) {
        const float rijx = d[jBody].qx - d[iBody].qx; // 1 flop
        const float rijy = d[jBody].qy - d[iBody].qy; // 1 flop
        const float rijz = d[jBody].qz - d[iBody].qz; // 1 flop

        // compute the || rij ||² distance between body i and body j
        const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
        
        // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
        const float ai = G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        accX += ai * rijx; // 2 flops
        accY += ai * rijy; // 2 flops
        accZ += ai * rijz; // 2 flops
    }

    accelerations[iBody].ax = accX;
    accelerations[iBody].ay = accY;
    accelerations[iBody].az = accZ;
}*/

SimulationNBodyGPU::SimulationNBodyGPU(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();

    // allocate memory for the bodies
    cudaMallocManaged(&this->d_qx, this->getBodies().getN() * sizeof(float));
    cudaMallocManaged(&this->d_qy, this->getBodies().getN() * sizeof(float));
    cudaMallocManaged(&this->d_qz, this->getBodies().getN() * sizeof(float));
    cudaMallocManaged(&this->d_m, this->getBodies().getN() * sizeof(float));

    // allocate memory for the accelerations
    cudaMallocManaged(&this->d_ax, this->getBodies().getN() * sizeof(float));
    cudaMallocManaged(&this->d_ay, this->getBodies().getN() * sizeof(float));
    cudaMallocManaged(&this->d_az, this->getBodies().getN() * sizeof(float));
    
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}


void SimulationNBodyGPU::computeOneIteration()
{   
    const unsigned long N = this->getBodies().getN();
    const int NTPB = 32;
    const int NB   = (this->getBodies().getN() + NTPB - 1) / NTPB; 
    const float softSquared =  std::pow(this->soft, 2);

    // compute the positions of the bodies
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;

    std::copy(d.x.begin(), d.x.end(), this->d_qx);
    std::copy(d.y.begin(), d.y.end(), this->d_qy);
    std::copy(d.z.begin(), d.z.end(), this->d_qz);
    std::copy(d.m.begin(), d.m.end(), this->d_m);


    calculate_forces<<<NTPB, NB>>>(this->d_qx, this->d_qy, this->d_qz, this->d_m, N, 
                                                    softSquared, this->G, this->d_ax, this->d_ay, this->d_az);
    cudaDeviceSynchronize();

    std::copy(this->d_ax, this->d_ax + N, ax.begin());
    std::copy(this->d_ay, this->d_ay + N, ay.begin());
    std::copy(this->d_az, this->d_az + N, az.begin());

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

// destructor
SimulationNBodyGPU::~SimulationNBodyGPU()
{
    cudaFree(this->d_accelerations);
}