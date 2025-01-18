/**
 * @file SimulationNBodyHeterogeneous.cu
 * @brief Implementation of the N-Body simulation on the GPU+CPU.
 * 
 * This is an heterogeneous implementation of the N-Body simulation. As explained 
 * in the GPU implementation, the computation space is divided into tiles of size
 * blockDim x blockDim. Since the size of the tiles are fixed, we have to handle the 
 * case where the number of bodies is not a multiple of the number of threads per block.
 * In this case, in order to avoid the divergence of the threads, we can compute the bottom
 * borders with the CPU (since the right ones causes less divergence). 
 * 
 * The CPU_LOAD parameter can increase the number of tiles that will be computed by the CPU.
 * For example, if CPU_LOAD = 1, the CPU will compute the last row of tiles of the grid and the    
 * reminder of the computation.
 * 
 * For the CPU part, we use the mipp+OMP implementation, which our fastest CPU implementation.
 */
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyHeterogeneous.hpp"
#include "commons.cuh"

#define CPU_LOAD 5 // the number of blocks that will be computed by the CPU (+ the reminder of the computation)


__global__ void
calculate_forces_no_border(float* devX, float* devY, float* devZ, float* devM, const unsigned long N, 
                 const float softSquared, const float G, float* devAx, float* devAy, float* devAz,
                 const float TILE_SIZE)
{
    extern __shared__ float4 shPosition[]; // shared memory to store the positions of the bodies
    int j, tile;
    float3 acc = {0.0f, 0.0f, 0.0f}; // acceleration of the body
    int id_i = blockIdx.x * blockDim.x + threadIdx.x; // id of the i-th resident body 

	// load the resident body
    float4 bi = make_float4(devX[id_i], devY[id_i], devZ[id_i], devM[id_i]); // position and mass of the i-th

    for (j = 0, tile = 0; j < N; j += TILE_SIZE, tile++) {
		int id_j = tile * blockDim.x + threadIdx.x; 
		shPosition[threadIdx.x] = make_float4(devX[id_j], devY[id_j], devZ[id_j], id_j >= N? 0:devM[id_j]);
		__syncthreads(); // wait for all the threads to load the data
		tile_calculation(bi, &acc, G, softSquared);
		__syncthreads(); // wait before overwriting the shared memory
    }

    // save the results in the global memory
    devAx[id_i] = acc.x;
    devAy[id_i] = acc.y;
    devAz[id_i] = acc.z;
}

SimulationNBodyHeterogeneous::SimulationNBodyHeterogeneous(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    const unsigned long N = this->getBodies().getN();

    this->NTPB = 128;
    this->N_x = N - (N % this->NTPB);// - CPU_LOAD * this->NTPB; // vertical size of the grid
    this->N_res = N % this->NTPB; // number of bodies that will be computed by the CPU
    if(N_x!=0 && N_x>CPU_LOAD * this->NTPB)
    {
        this->N_x -= CPU_LOAD * this->NTPB; // vertical size of the grid
        this->N_res += CPU_LOAD * this->NTPB;
    }
    this->N_y = N; // horizontal size of the grid
    this->NB = N_x / this->NTPB;

	// allocate memory for the bodies
	cudaMalloc(&this->d_qx, N_y * sizeof(float));
	cudaMalloc(&this->d_qy, N_y * sizeof(float));
	cudaMalloc(&this->d_qz, N_y * sizeof(float));
	cudaMalloc(&this->d_m,  N_y * sizeof(float));

	// allocate memory for the accelerations
	cudaMalloc(&this->d_ax, N_x * sizeof(float));
	cudaMalloc(&this->d_ay, N_x * sizeof(float));
	cudaMalloc(&this->d_az, N_x * sizeof(float));

    // allocate pinned memory for the bodies
    cudaMallocHost(&this->p_ax, N_x * sizeof(float));
    cudaMallocHost(&this->p_ay, N_x * sizeof(float));
    cudaMallocHost(&this->p_az, N_x * sizeof(float));
    
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

/**
 * The first approach is to remove the borders.
 */
void SimulationNBodyHeterogeneous::computeOneIteration()
{   
    const unsigned long N = this->getBodies().getN();
    const float softSquared =  this->soft*this->soft;

    // alias
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;

    if(N < this->NTPB)
    {
        computeOneIterationCPU(); // if the number of bodies is less than the number of threads per block just use the CPU
    }else{
        cudaMemcpy(this->d_qx, d.qx.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_qy, d.qy.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_qz, d.qz.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(this->d_m,  d.m.data(),  N_y * sizeof(float), cudaMemcpyHostToDevice);

        calculate_forces_no_border<<<NB, NTPB, NTPB * sizeof(float4)>>>(this->d_qx, this->d_qy, this->d_qz, this->d_m, N_y, 
                                        softSquared, this->G, this->d_ax, this->d_ay, 
                                        this->d_az, this->NTPB);

        cudaMemcpyAsync(this->p_ax, this->d_ax, N_x * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(this->p_ay, this->d_ay, N_x * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpyAsync(this->p_az, this->d_az, N_x * sizeof(float), cudaMemcpyDeviceToHost);

        computeOneIterationCPU();    

        // synchronize cpu and gpu here
        cudaDeviceSynchronize();
        
        // copy the results from the pinned memory to the host memory
        std::copy(this->p_ax, this->p_ax + N_x, ax.begin());
        std::copy(this->p_ay, this->p_ay + N_x, ay.begin());
        std::copy(this->p_az, this->p_az + N_x, az.begin());
    }

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

// destructor
SimulationNBodyHeterogeneous::~SimulationNBodyHeterogeneous()
{
    // free bodies memory
    cudaFree(this->d_qx);
    cudaFree(this->d_qy);
    cudaFree(this->d_qz);
    cudaFree(this->d_m);

    cudaFreeHost(this->d_ax);
    cudaFreeHost(this->d_ay);
    cudaFreeHost(this->d_az);
}