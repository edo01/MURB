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
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyOptimV2::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

/**
 * Optimization made:
 * - We reduce the complexity of the algorithm from O(n²) to O(n²/2) by taking advantage of the symmetry of the problem using the third Newton's law
 * - We avoid the use of the square root function by using algebraic manipulation
 * - We avoid to recompute the same value multiple times (softSquared)
 * - We avoid to access the data structure multiple times (qx, qy, qz, m) 
 */
void SimulationNBodyOptimV2::computeBodiesAcceleration() {
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN(); 
    const float softSquared = this->soft*this->soft; 

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax = 0.0f, ay = 0.0f, az = 0.0f; // Local accumulators for acceleration

        const float qx_i = d[iBody].qx;
        const float qy_i = d[iBody].qy;
        const float qz_i = d[iBody].qz;
        const float m_i = d[iBody].m;

        // we skip the diagonal, the interaction of a body with itself is not considered 
        // and we take advantage of the symmetry of the problem using the third Newton's law
        for (unsigned long jBody = iBody+1; jBody < nBodies; jBody++) {

            const float rijx = d[jBody].qx - qx_i;
            const float rijy = d[jBody].qy - qy_i;
            const float rijz = d[jBody].qz - qz_i;

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            float ai  = this->G/ (rijSquared*std::sqrt(rijSquared));
            float aj = ai;
            ai = ai * d[jBody].m;
            aj = aj * m_i;

            ax += ai * rijx;
            ay += ai * rijy;
            az += ai * rijz;

            // we take advantage of the symmetry of the problem using the third Newton's law
            this->accelerations[jBody].ax -= aj * rijx;
            this->accelerations[jBody].ay -= aj * rijy;
            this->accelerations[jBody].az -= aj * rijz;
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax += ax;
        this->accelerations[iBody].ay += ay;
        this->accelerations[iBody].az += az;
    }
}
/**
 * 
 * @TODO: TRY AOS
 */

void SimulationNBodyOptimV2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
