/**
 * @file SimulationNBodyOptimV2.cpp
 * @brief Optimized implementation of the N-body simulation.
 *
 * 
 * Here we combine both the transformations carried in Optim and an improved version of the 
 * algorithm that takes advantage of the symmetry of the problem using the third Newton's law.
 * 
 * - We reduce the complexity of the algorithm from n² to (n-1)*n/2 leverage the symmetry of 
 *   the problem.
 * - We avoid the use of the square root function by using algebraic manipulation
 * - We avoid to recompute the same value multiple times (softSquared)
 * - We avoid to access the data structure multiple times (qx, qy, qz, m) 
 * - We use local accumulators for acceleration and avoid to access the data structure multiple times
 * - We avoid to initialize the accelerations to zero at each iteration
 * 
 * Please note that reducing the complexity of the algorithm from n² to (n-1)*n/2 comes 
 * with a cost in terms of cache misses. Also, the algorithm cannot be easily parallelized
 * because of accumulation of the accelerations. So even if it is may be faster than the
 * n^2 algorithm, it may not be the best choice for large n and for parallelization.
 */

#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>


#include "SimulationNBodyOptimV2.hpp"

SimulationNBodyOptimV2::SimulationNBodyOptimV2(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    //flops = n*(n-1)/2 * 27 + n*3
    this->flopsPerIte = 27.f * (float)(nBodies * (nBodies-1)/2) + 3.f * nBodies;
    this->accelerations.resize(nBodies);
}

void SimulationNBodyOptimV2::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


void SimulationNBodyOptimV2::computeBodiesAcceleration() {
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN(); 
    const float softSquared = this->soft*this->soft; 

    // flops = n*(n-1)/2 * 27 + n*3
    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f; // Local accumulators for acceleration

        const float qx_i = d[iBody].qx;
        const float qy_i = d[iBody].qy;
        const float qz_i = d[iBody].qz;
        const float m_i = d[iBody].m;

        // we skip the diagonal, the interaction of a body with itself is not considered 
        // and we take advantage of the symmetry of the problem using the third Newton's law
        for (unsigned long jBody = iBody+1; jBody < nBodies; jBody++) {

            const float rijx = d[jBody].qx - qx_i; // 1 flops
            const float rijy = d[jBody].qy - qy_i; // 1 flops
            const float rijz = d[jBody].qz - qz_i; // 1 flops

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz; // 5 flops
            rijSquared += softSquared; // 1 flops
            float ai  = this->G / (rijSquared*std::sqrt(rijSquared)); // 4 flops
            float aj = ai;
            ai = ai * d[jBody].m; // 1 flops
            aj = aj * m_i; // 1 flops

            ax += ai * rijx; // 2 flops
            ay += ai * rijy; // 2 flops
            az += ai * rijz; // 2 flops

            // we take advantage of the symmetry of the problem using the third Newton's law
            this->accelerations[jBody].ax -= aj * rijx; // 2 flops
            this->accelerations[jBody].ay -= aj * rijy; // 2 flops
            this->accelerations[jBody].az -= aj * rijz; // 2 flops
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax += ax; // 1 flops
        this->accelerations[iBody].ay += ay; // 1 flops
        this->accelerations[iBody].az += az; // 1 flops
    }
}

void SimulationNBodyOptimV2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
