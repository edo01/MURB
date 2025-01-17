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

/**
 * Here all the computation depends on the size of the tile. In order to work the tile size must be greater than the 
 * number of threads in the block.
 */
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

    this->NTPB = 256;
    this->N_x = N - (N % this->NTPB); // vertical size of the grid
    this->N_y = N; // horizontal size of the grid
    this->N_res = N % this->NTPB;
    this->NB = (N + this->NTPB - 1) / this->NTPB;
    
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
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    const std::vector<float> &m = d.m;
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;
    
    // constants
    const mipp::Reg<float> r_softSquared = mipp::Reg<float>(this->soft*this->soft);
    const mipp::Reg<float> r_G = mipp::Reg<float>(this->G);

    cudaMemcpy(this->d_qx, d.qx.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_qy, d.qy.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_qz, d.qz.data(), N_y * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(this->d_m, d.m.data(),   N_y * sizeof(float), cudaMemcpyHostToDevice);

    calculate_forces_no_border<<<NB, NTPB, NTPB * sizeof(float4)>>>(this->d_qx, this->d_qy, this->d_qz, this->d_m, N_y, 
                                    softSquared, this->G, this->d_ax, this->d_ay, 
                                    this->d_az, this->NTPB);

    cudaMemcpyAsync(this->p_ax, this->d_ax, N_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(this->p_ay, this->d_ay, N_x * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpyAsync(this->p_az, this->d_az, N_x * sizeof(float), cudaMemcpyDeviceToHost);

    //CPU computation
    #pragma omp parallel for
    for (unsigned long iBody = this->N_x; iBody < N; iBody+=1) {
        // accumulators
        mipp::Reg<float> r_ax(0.f);
        mipp::Reg<float> r_ay(0.f);
        mipp::Reg<float> r_az(0.f);

        const mipp::Reg<float> r_qx_i(qx[iBody]);
        const mipp::Reg<float> r_qy_i(qy[iBody]);
        const mipp::Reg<float> r_qz_i(qz[iBody]);
        unsigned long jBody;

        for (jBody = 0; jBody < N; jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // 4 flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // 4 flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // 4 flop


            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz;
        } 

        ax[iBody] = mipp::sum(r_ax);
        ay[iBody] = mipp::sum(r_ay);
        az[iBody] = mipp::sum(r_az);
    }    

    // synchronize cpu and gpu here
    cudaDeviceSynchronize();
    
    // copy the results from the pinned memory to the host memory
    for (unsigned long i = 0; i < N_x; i++) {
        ax[i] = p_ax[i];
        ay[i] = p_ay[i];
        az[i] = p_az[i];
    }

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

// destructor
SimulationNBodyHeterogeneous::~SimulationNBodyHeterogeneous()
{
    // free bodies memory
    cudaFreeHost(this->d_qx);
    cudaFreeHost(this->d_qy);
    cudaFreeHost(this->d_qz);
    cudaFreeHost(this->d_m);

    cudaFreeHost(this->d_ax);
    cudaFreeHost(this->d_ay);
    cudaFreeHost(this->d_az);
}