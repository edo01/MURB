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


/**
 * Here all the computation depends on the size of the tile. In order to work the tile size must be greater than the 
 * number of threads in the block.
 */
__global__ void
calculate_forces(float* devX, float* devY, float* devZ, float* devM, const unsigned long N, 
                 const float softSquared, const float G, float* devAx, float* devAy, float* devAz,
                 const float TILE_SIZE)
{
    extern __shared__ float4 shPosition[]; // shared memory to store the positions of the bodies
    int j, tile;
    float3 acc = {0.0f, 0.0f, 0.0f}; // acceleration of the body
    int id_i = blockIdx.x * blockDim.x + threadIdx.x; // id of the i-th resident body 
	
	// load the resident body
	float4 bi = (id_i<N)? make_float4(devX[id_i], devY[id_i], devZ[id_i], devM[id_i]) : make_float4(0.0f, 0.0f, 0.0f, 0.0f); // position and mass of the i-th

    for (j = 0, tile = 0; j < N; j += TILE_SIZE, tile++) {
		int id_j = tile * blockDim.x + threadIdx.x; //id_j >= N? 0: 
		shPosition[threadIdx.x] = {devX[id_j], devY[id_j], devZ[id_j], id_j >= N? 0: devM[id_j]}; // check if this is coalesced
		__syncthreads(); // wait for all the threads to load the data
		tile_calculation(bi, &acc, G, softSquared);
		__syncthreads(); // wait before overwriting the shared memory
    }

    // save the results in the global memory
	if(id_i < N)
	{
		devAx[id_i] = acc.x;
		devAy[id_i] = acc.y;
		devAz[id_i] = acc.z;
	}
}

SimulationNBodyGPU::SimulationNBodyGPU(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();

	// allocate pinned memory for the bodies
	cudaMallocHost(&this->d_qx, this->getBodies().getN() * sizeof(float));
	cudaMallocHost(&this->d_qy, this->getBodies().getN() * sizeof(float));
	cudaMallocHost(&this->d_qz, this->getBodies().getN() * sizeof(float));
	cudaMallocHost(&this->d_m, this->getBodies().getN() * sizeof(float));

	// allocate pinned memory for the accelerations
	cudaMallocHost(&this->d_ax, this->getBodies().getN() * sizeof(float));
	cudaMallocHost(&this->d_ay, this->getBodies().getN() * sizeof(float));
	cudaMallocHost(&this->d_az, this->getBodies().getN() * sizeof(float));
    
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}


void SimulationNBodyGPU::computeOneIteration()
{   
    const unsigned long N = this->getBodies().getN();
    const int NTPB = 256;
    constexpr int TILE_SIZE = NTPB;
    const int NB   = (this->getBodies().getN() + NTPB - 1) / NTPB; 
    const float softSquared =  this->soft*this->soft;

    // compute the positions of the bodies
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;

	cudaMemcpy(this->d_qx, d.qx.data(), N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_qy, d.qy.data(), N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_qz, d.qz.data(), N * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(this->d_m, d.m.data(), N * sizeof(float),   cudaMemcpyHostToDevice);

    calculate_forces<<<NB, NTPB, NTPB*sizeof(float4)>>>(this->d_qx, this->d_qy, this->d_qz, this->d_m, N, 
                                    softSquared, this->G, this->d_ax, this->d_ay, 
                                    this->d_az, TILE_SIZE);

    cudaMemcpy(ax.data(), this->d_ax, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(ay.data(), this->d_ay, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(az.data(), this->d_az, N * sizeof(float), cudaMemcpyDeviceToHost);

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

// destructor
SimulationNBodyGPU::~SimulationNBodyGPU()
{
    // free bodies memory
    cudaFree(this->d_qx);
    cudaFree(this->d_qy);
    cudaFree(this->d_qz);
    cudaFree(this->d_m);

    cudaFree(this->d_ax);
    cudaFree(this->d_ay);
    cudaFree(this->d_az);
}