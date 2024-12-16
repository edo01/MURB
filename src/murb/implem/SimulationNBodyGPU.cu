#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyGPU.hpp"


//maybe for the cache is better this version
__global__
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

}


/* __global__
void kernelComputeBodiesAcceleration(const dataSoA_t<float>* __restrict__ d, const int N, 
                                     const float softSquared, const float G,
                                     accSoA_t<float>* accelerations ){
    // The strategy is to let each thread compute the acceleration of one body
    // THe inner loop will be executed by each thread. We must ensure that the data are fetched 
    // only once and are broadcasted to all threads in the block. In order to do that, 

    // Get the index of the body to compute the acceleration
    const unsigned long iBody = blockIdx.x * blockDim.x + threadIdx.x;

    float accX = 0.f, accY = 0.f, accZ = 0.f;

    for (unsigned long jBody = 0; jBody < N; jBody++) {
        const float rijx = d->qx[jBody] - d->qx[iBody]; // 1 flop
        const float rijy = d->qy[jBody] - d->qy[iBody]; // 1 flop
        const float rijz = d->qz[jBody] - d->qz[iBody]; // 1 flop

        // compute the || rij ||² distance between body i and body j
        const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2); // 5 flops
        
        // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
        const float ai = G * d->m[jBody] / std::pow(rijSquared + softSquared, 3.f / 2.f); // 5 flops

        // add the acceleration value into the acceleration vector: ai += || ai ||.rij
        accX += ai * rijx; // 2 flops
        accY += ai * rijy; // 2 flops
        accZ += ai * rijz; // 2 flops
    }

    accelerations->ax[iBody] = accX;
    accelerations->ay[iBody] = accY;
    accelerations->az[iBody] = accZ;

} */

SimulationNBodyGPU::SimulationNBodyGPU(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();

    // allocate memory for the accelerations
    cudaMallocManaged(&this->d_accelerations, this->getBodies().getN() * sizeof(accAoS_t<float>));

    // allocate memory for the positions
    //this->getBodies().getDataAoS()
    cudaMallocManaged(&this->d_bodies, this->getBodies().getN() * sizeof(dataAoS_t<float>));

    this->accelerations.resize(this->getBodies().getN());
}


void SimulationNBodyGPU::computeOneIteration()
{   
    const int NTPB = 32;
    const int NB   = (this->getBodies().getN() + NTPB - 1) / NTPB; 
    const float softSquared =  std::pow(this->soft, 2);

    // compute the positions of the bodies
    std::copy(this->getBodies().getDataAoS().begin(), this->getBodies().getDataAoS().end(), this->d_bodies);

    kernelComputeBodiesAcceleration<<<NTPB, NB>>>(this->d_bodies, this->getBodies().getN(), 
                                                    softSquared, this->G, this->d_accelerations);
    cudaDeviceSynchronize();

    std::copy(this->d_accelerations, this->d_accelerations + this->getBodies().getN(), this->accelerations.begin());

    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}

// destructor
SimulationNBodyGPU::~SimulationNBodyGPU()
{
    cudaFree(this->d_accelerations);
}