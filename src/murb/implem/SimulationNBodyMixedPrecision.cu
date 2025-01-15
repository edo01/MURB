#include <cuda_fp16.h> 
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyMixedPrecision.hpp"


#include <cuda_fp16.h> // 支持半精度浮点数

__global__
void kernelComputeBodiesAccelerationMixedPrecision(const dataAoS_t<float>* __restrict__ d, 
                                                   const unsigned long N, 
                                                   const float softSquared, 
                                                   const float G,
                                                   accAoS_t<float>* accelerations) {
    extern __shared__ __half sharedBodies[]; // 使用半精度共享内存
    unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    // 检查索引是否超出范围
    if (iBody >= N) return;

    float accX = 0.f, accY = 0.f, accZ = 0.f;

    // 遍历所有粒子并分块加载到共享内存
    for (unsigned long tile = 0; tile < (N + blockDim.x - 1) / blockDim.x; ++tile) {
        unsigned long j = tile * blockDim.x + threadIdx.x;

        // 将粒子数据加载到共享内存并转换为半精度
        if (j < N) {
            sharedBodies[threadIdx.x * 3 + 0] = __float2half(d[j].qx);
            sharedBodies[threadIdx.x * 3 + 1] = __float2half(d[j].qy);
            sharedBodies[threadIdx.x * 3 + 2] = __float2half(d[j].qz);
        }
        __syncthreads();

        // 遍历当前共享内存中的粒子
        for (unsigned long jBody = 0; jBody < blockDim.x && (tile * blockDim.x + jBody) < N; ++jBody) {
            const float qx = __half2float(sharedBodies[jBody * 3 + 0]);
            const float qy = __half2float(sharedBodies[jBody * 3 + 1]);
            const float qz = __half2float(sharedBodies[jBody * 3 + 2]);

            const float rijx = qx - d[iBody].qx;
            const float rijy = qy - d[iBody].qy;
            const float rijz = qz - d[iBody].qz;

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;

           
            const float ai = G / powf(rijSquared + softSquared, 1.5f);

            accX += ai * rijx;
            accY += ai * rijy;
            accZ += ai * rijz;
        }
        __syncthreads();
    }


    accelerations[iBody].ax = accX;
    accelerations[iBody].ay = accY;
    accelerations[iBody].az = accZ;
}



void SimulationNBodyMixedPrecision::computeBodiesAcceleration() {
    const unsigned long nBodies = this->getBodies().getN();
    const int NTPB = 32; 
    const int NB = (nBodies + NTPB - 1) / NTPB;
    const float softSquared = std::pow(this->soft, 2);

    
    cudaMemcpy(this->d_bodies, this->getBodies().getDataAoS().data(), 
               nBodies * sizeof(dataAoS_t<float>), cudaMemcpyHostToDevice);

    // Calling the mixed precision kernel function
    size_t sharedMemSize = NTPB * sizeof(__half) * 3;
    kernelComputeBodiesAccelerationMixedPrecision<<<NB, NTPB, sharedMemSize>>>(
        this->d_bodies, nBodies, softSquared, this->G, this->d_accelerations);
    cudaDeviceSynchronize();

    cudaMemcpy(this->accelerations.data(), this->d_accelerations, 
               nBodies * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);
}

SimulationNBodyMixedPrecision::SimulationNBodyMixedPrecision(const unsigned long nBodies, const std::string &scheme, 
                                                             const float soft, const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit) {
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();

    cudaMalloc(&this->d_bodies, this->getBodies().getN() * sizeof(dataAoS_t<float>));
    cudaMalloc(&this->d_accelerations, this->getBodies().getN() * sizeof(accAoS_t<float>));

    this->accelerations.resize(this->getBodies().getN());
}


void SimulationNBodyMixedPrecision::computeOneIteration()
{   
    const int NTPB = 32;
    const int NB   = (this->getBodies().getN() + NTPB - 1) / NTPB; 
    const float softSquared =  std::pow(this->soft, 2);

    // compute the positions of the bodies
    std::copy(this->getBodies().getDataAoS().begin(), this->getBodies().getDataAoS().end(), this->d_bodies);

    size_t sharedMemSize = NTPB * sizeof(__half) * 3;
    kernelComputeBodiesAccelerationMixedPrecision<<<NB, NTPB, sharedMemSize>>>(this->d_bodies, this->getBodies().getN(), softSquared, this->G, this->d_accelerations);
    cudaDeviceSynchronize();

    std::copy(this->d_accelerations, this->d_accelerations + this->getBodies().getN(), this->accelerations.begin());

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}



SimulationNBodyMixedPrecision::~SimulationNBodyMixedPrecision() {
    cudaFree(this->d_bodies);
    cudaFree(this->d_accelerations);
}