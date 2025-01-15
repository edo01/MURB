#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "SimulationNBodyHetro.hpp"

SimulationNBodyGPU::SimulationNBodyHetro(const unsigned long nBodies, const std::string &scheme, const float soft,
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

std::pair<unsigned long, unsigned long> allocateBodies(unsigned long totalBodies) {
    int gpuCapability = getGPUComputeCapacity();
    int cpuCores = getCPUCores();

    if (gpuCapability == 0) {
        return {0, totalBodies};
    }


    double totalWeight = gpuCapability + cpuCores;
    double gpuWeight = gpuCapability / totalWeight;
    double cpuWeight = cpuCores / totalWeight;

    unsigned long gpuBodies = static_cast<unsigned long>(gpuWeight * totalBodies);
    unsigned long cpuBodies = totalBodies - gpuBodies;

    return {gpuBodies, cpuBodies};
}


/* Particles are divided equally between GPU and CPU

void SimulationNBodyHetero::computeBodiesAcceleration() {
    const unsigned long nBodies = this->getBodies().getN();
    const unsigned long gpuBodies = nBodies / 2; 
    const unsigned long cpuBodies = nBodies - gpuBodies;

    const std::vector<dataAoS_t<float>> &h_bodies = this->getBodies().getDataAoS();
    const float softSquared = std::pow(this->soft, 2);

   
    cudaMemcpy(this->d_bodies, h_bodies.data(), gpuBodies * sizeof(dataAoS_t<float>), cudaMemcpyHostToDevice);

    // GPU 
    std::thread gpuThread([&]() {
        const int NTPB = 32; 
        const int NB   = (gpuBodies + NTPB - 1) / NTPB;

        kernelComputeBodiesAcceleration<<<NB, NTPB>>>(this->d_bodies, gpuBodies, softSquared, this->G, this->d_accelerations);
        cudaDeviceSynchronize();
    });

    // CPU 
    for (unsigned long iBody = gpuBodies; iBody < nBodies; iBody++) {
        this->accelerations[iBody] = {0.f, 0.f, 0.f};
        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {
            if (iBody == jBody) continue;

            const float rijx = h_bodies[jBody].qx - h_bodies[iBody].qx;
            const float rijy = h_bodies[jBody].qy - h_bodies[iBody].qy;
            const float rijz = h_bodies[jBody].qz - h_bodies[iBody].qz;

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
            const float ai = this->G * h_bodies[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f);

            this->accelerations[iBody].ax += ai * rijx;
            this->accelerations[iBody].ay += ai * rijy;
            this->accelerations[iBody].az += ai * rijz;
        }
    }

    gpuThread.join();

    cudaMemcpy(this->accelerations.data(), this->d_accelerations, gpuBodies * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);
}*/

void SimulationNBodyHetero::computeBodiesAcceleration() {
    const unsigned long totalBodies = this->getBodies().getN();

    // Dynamically allocate
    auto [gpuBodies, cpuBodies] = allocateBodies(totalBodies);

    const std::vector<dataAoS_t<float>> &h_bodies = this->getBodies().getDataAoS();
    const float softSquared = std::pow(this->soft, 2);

    cudaMemcpy(d_bodies, h_bodies.data(), gpuBodies * sizeof(dataAoS_t<float>), cudaMemcpyHostToDevice);

    // GPU 
    std::thread gpuThread([&]() {
        const int NTPB = 32;
        const int NB = (gpuBodies + NTPB - 1) / NTPB;
        kernelComputeBodiesAcceleration<<<NB, NTPB>>>(d_bodies, gpuBodies, softSquared, this->G, d_accelerations);
        cudaDeviceSynchronize();
    });

    // CPU 
    for (unsigned long iBody = gpuBodies; iBody < totalBodies; ++iBody) {
        accelerations[iBody] = {0.f, 0.f, 0.f};
        for (unsigned long jBody = 0; jBody < totalBodies; ++jBody) {
            if (iBody == jBody) continue;

            const float rijx = h_bodies[jBody].qx - h_bodies[iBody].qx;
            const float rijy = h_bodies[jBody].qy - h_bodies[iBody].qy;
            const float rijz = h_bodies[jBody].qz - h_bodies[iBody].qz;

            const float rijSquared = rijx * rijx + rijy * rijy + rijz * rijz;
            const float ai = this->G * h_bodies[jBody].m / std::pow(rijSquared + softSquared, 1.5f);

            accelerations[iBody].ax += ai * rijx;
            accelerations[iBody].ay += ai * rijy;
            accelerations[iBody].az += ai * rijz;
        }
    }

    gpuThread.join();

    cudaMemcpy(accelerations.data(), d_accelerations, gpuBodies * sizeof(accAoS_t<float>), cudaMemcpyDeviceToHost);
}


void SimulationNBodyHetro::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}



// destructor
SimulationNBodyHetero::~SimulationNBodyHetero() {
    cudaFree(this->d_bodies);
    cudaFree(this->d_accelerations);
}