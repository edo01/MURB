/**
 * @file SimulationNBodyGPU.cu
 * @brief Implementation of the N-Body simulation on the GPU.
 * 
 *
 * The GPU implementation of the N-Body simulation is based on the O(N^2) algorithm.
 * If we consider that for each body we must compute the acceleration with all the other bodies,
 * we can see that there is a grid of nBodies x nBodies computations to be done. This grid can be 
 * divided into tiles of size blockDim x blockDim. Each tile will be computed by a block of threads.
 * The algorithm is as follows:
 * - First the positions of blockDim bodies are loaded into the shared memory. These will reside in the
 *   shared memory of the block and are the bodies for which the acceleration will be computed.
 * - Then other blockDim bodies are loaded into the shared memory. These bodies are the ones for which the
 *   acceleration will be computed.
 * - the result of the computation is accumulated in the shared memory of the block.
 * - New bodies are loaded into the shared memory and the process is repeated until all the bodies are computed.
 * - The final result is written into the global memory.
 * 
 * Some notes on the implementation:
 * - We couldn't use pinned memory for the bodies because we do not manage the allocation of the memory.
 * - A better implementation would be to use the AoS format for the bodies instead of the SoA format.
 *   This would allow us to load the bodies in a single read operation instead of four. However,
 *   this would require some additional copy operations before launching the kernel. For a 
 *   lack of time, we did not implement this.
 * - The borders of the grid must be handled separately. The right boundary of the grid,
 *   namely the accumulation of the forces of the bodies for each i-th body, is done by 
 *   setting the mass of the bodies that are out of the grid to zero. The bottom boundary
 *   is more tricky. We must avoid the store and load from the global memory, in order to
 *   avoid illegal memory access.
 */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyGPU.hpp"
#include "commons.cuh"


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
		shPosition[threadIdx.x] = make_float4(devX[id_j], devY[id_j], devZ[id_j], id_j >= N? 0: devM[id_j]); // check if this is coalesced
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
	cudaMalloc(&this->d_qx, this->getBodies().getN() * sizeof(float));
	cudaMalloc(&this->d_qy, this->getBodies().getN() * sizeof(float));
	cudaMalloc(&this->d_qz, this->getBodies().getN() * sizeof(float));
	cudaMalloc(&this->d_m, this->getBodies().getN() * sizeof(float));

	// allocate pinned memory for the accelerations
	cudaMalloc(&this->d_ax, this->getBodies().getN() * sizeof(float));
	cudaMalloc(&this->d_ay, this->getBodies().getN() * sizeof(float));
	cudaMalloc(&this->d_az, this->getBodies().getN() * sizeof(float));
    
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}


void SimulationNBodyGPU::computeOneIteration()
{   
    const unsigned long N = this->getBodies().getN();
    const int NTPB = 128;
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