#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <arm_neon.h>
#include <vector>


#include "SimulationNBodyOptim.hpp"

SimulationNBodyOptim::SimulationNBodyOptim(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptim::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyOptim::computeBodiesAcceleration() {
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN(); 
    const float softSquared = std::pow(this->soft, 2); 

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f; // Local accumulators for acceleration

        for (unsigned long jBody = 0; jBody + 3 < nBodies; jBody += 4) { // Unroll by 4
            // First iteration (jBody)
            {
                const float rijx = d[jBody].qx - d[iBody].qx;
                const float rijy = d[jBody].qy - d[iBody].qy;
                const float rijz = d[jBody].qz - d[iBody].qz;

                const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
                const float ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f);
                ax += ai * rijx;
                ay += ai * rijy;
                az += ai * rijz;
            }
            // Second iteration (jBody + 1)
            {
                const float rijx = d[jBody + 1].qx - d[iBody].qx;
                const float rijy = d[jBody + 1].qy - d[iBody].qy;
                const float rijz = d[jBody + 1].qz - d[iBody].qz;

                const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
                const float ai = this->G * d[jBody + 1].m / std::pow(rijSquared + softSquared, 3.f / 2.f);
                ax += ai * rijx;
                ay += ai * rijy;
                az += ai * rijz;
            }
            // Third iteration (jBody + 2)
            {
                const float rijx = d[jBody + 2].qx - d[iBody].qx;
                const float rijy = d[jBody + 2].qy - d[iBody].qy;
                const float rijz = d[jBody + 2].qz - d[iBody].qz;

                const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
                const float ai = this->G * d[jBody + 2].m / std::pow(rijSquared + softSquared, 3.f / 2.f);
                ax += ai * rijx;
                ay += ai * rijy;
                az += ai * rijz;
            }
            // Fourth iteration (jBody + 3)
            {
                const float rijx = d[jBody + 3].qx - d[iBody].qx;
                const float rijy = d[jBody + 3].qy - d[iBody].qy;
                const float rijz = d[jBody + 3].qz - d[iBody].qz;

                const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
                const float ai = this->G * d[jBody + 3].m / std::pow(rijSquared + softSquared, 3.f / 2.f);
                ax += ai * rijx;
                ay += ai * rijy;
                az += ai * rijz;
            }
        }

        // Handle any remaining bodies (if nBodies is not a multiple of 4)
        for (unsigned long jBody = (nBodies / 4) * 4; jBody < nBodies; jBody++) {
            const float rijx = d[jBody].qx - d[iBody].qx;
            const float rijy = d[jBody].qy - d[iBody].qy;
            const float rijz = d[jBody].qz - d[iBody].qz;

            const float rijSquared = std::pow(rijx, 2) + std::pow(rijy, 2) + std::pow(rijz, 2);
            const float ai = this->G * d[jBody].m / std::pow(rijSquared + softSquared, 3.f / 2.f);
            ax += ai * rijx;
            ay += ai * rijy;
            az += ai * rijz;
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax += ax;
        this->accelerations[iBody].ay += ay;
        this->accelerations[iBody].az += az;
    }
}




void SimulationNBodyOptim::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
