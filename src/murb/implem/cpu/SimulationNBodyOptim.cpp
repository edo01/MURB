/**
 * @file SimulationNBodyOptim.cpp
 * @brief Optimized implementation of the N-body simulation. 
 * 
 * The code is optimized using transformations of the original code. The algorithm is the
 * same as the one in SimulationNBodyNaive.cpp.
 *  
 * - We avoid the use of the square root function by using algebraic manipulation
 * - We avoid to recompute the same value multiple times (softSquared)
 * - We avoid to access the data structure multiple times (qx, qy, qz, m)
 * - We use local accumulators for acceleration and avoid to access the data structure multiple times
 * - We avoid to initialize the accelerations to zero at each iteration
 * - We use pragma unroll to unroll the inner loop
 * 
 * flops = n² * 19
 */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>


#include "SimulationNBodyOptim.hpp"

SimulationNBodyOptim::SimulationNBodyOptim(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 19.f * (float)(nBodies * nBodies);
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptim::computeBodiesAcceleration() {

    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN(); 
    const float softSquared = this->soft*this->soft; 

    // flops = n² * 19 + 3
    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f; // Local accumulators for acceleration

        // constants for the body i
        const float qx_i = d[iBody].qx;
        const float qy_i = d[iBody].qy;
        const float qz_i = d[iBody].qz;
        
        #pragma unroll 4
        for (unsigned long jBody = 0; jBody < nBodies; jBody ++) {
            const float rijx = d[jBody].qx - qx_i; // 1 flops
            const float rijy = d[jBody].qy - qy_i; // 1 flops
            const float rijz = d[jBody].qz - qz_i; // 1 flops

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz; // 5 flops
            rijSquared += softSquared; // 1 flops
            const float ai = this->G * d[jBody].m / (rijSquared*std::sqrt(rijSquared)); // 4 flops

            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax = ax;
        this->accelerations[iBody].ay = ay;
        this->accelerations[iBody].az = az;
    }
}

void SimulationNBodyOptim::computeOneIteration()
{
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
