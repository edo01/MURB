#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <arm_neon.h>
#include <vector>


#include "SimulationNBodySimdOptim.hpp"

SimulationNBodySimdOPtim::SimulationNBodySimdOptim(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodySimdOptim::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


void SimulationNBodySimdOptim::computeBodiesAcceleration() {
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN();
    const float softSquared_r = std::pow(this->soft, 2); // Precompute softSquared

    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax_r = 0.0f, ay_r = 0.0f, az_r = 0.0f; 

        for (unsigned long jBody = 0; jBody + 4 <= nBodies; jBody += 4) {
            // Load 4 elements of positions and masses into NEON vectors
            float32x4_t qx_j_r = vld1q_f32(&d[jBody].qx); // qx[jBody:jBody+3]
            float32x4_t qy_j_r = vld1q_f32(&d[jBody].qy); // qy[jBody:jBody+3]
            float32x4_t qz_j_r = vld1q_f32(&d[jBody].qz); // qz[jBody:jBody+3]
            float32x4_t mj_r   = vld1q_f32(&d[jBody].m);  // m[jBody:jBody+3]

            // Load single `iBody` position values into scalar registers
            float qx_i_r = d[iBody].qx;
            float qy_i_r = d[iBody].qy;
            float qz_i_r = d[iBody].qz;

            // rij
            float32x4_t rijx_r = vsubq_f32(qx_j_r, vdupq_n_f32(qx_i_r));
            float32x4_t rijy_r = vsubq_f32(qy_j_r, vdupq_n_f32(qy_i_r));
            float32x4_t rijz_r = vsubq_f32(qz_j_r, vdupq_n_f32(qz_i_r));

            // rijx^2 + rijy^2 + rijz^2
            float32x4_t rijx2_r = vmulq_f32(rijx_r, rijx_r); // rijx^2
            float32x4_t rijy2_r = vmulq_f32(rijy_r, rijy_r); // rijy^2
            float32x4_t rijz2_r = vmulq_f32(rijz_r, rijz_r); // rijz^2
            float32x4_t rijSquared_r = vaddq_f32(vaddq_f32(rijx2_r, rijy2_r), rijz2_r);

            // Add softening term
            float32x4_t softSquared_r_v = vdupq_n_f32(softSquared_r);
            rijSquared_r = vaddq_f32(rijSquared_r, softSquared_r_v);

            // Compute acceleration scalar: G * mj / (rijSquared + e²)^(3/2)
            float32x4_t rijCubed_r = vmulq_f32(rijSquared_r, vmulq_f32(rijSquared_r, rijSquared_r));
            float32x4_t invRijCubed_r = vrsqrteq_f32(rijCubed_r); // Approximate 1/sqrt(rijCubed)
            float32x4_t ai_r = vmulq_f32(vmulq_f32(vdupq_n_f32(this->G), mj_r), invRijCubed_r);

            // Accumulate acceleration components
            ax_r += vaddvq_f32(vmulq_f32(ai_r, rijx_r)); // Horizontal add for rijx component
            ay_r += vaddvq_f32(vmulq_f32(ai_r, rijy_r)); // Horizontal add for rijy component
            az_r += vaddvq_f32(vmulq_f32(ai_r, rijz_r)); // Horizontal add for rijz component
        }

        // Handle remaining bodies if nBodies is not a multiple of 4
        for (unsigned long jBody = (nBodies / 4) * 4; jBody < nBodies; jBody++) {
            float rijx_r = d[jBody].qx - d[iBody].qx;
            float rijy_r = d[jBody].qy - d[iBody].qy;
            float rijz_r = d[jBody].qz - d[iBody].qz;
            float rijSquared_r = rijx_r * rijx_r + rijy_r * rijy_r + rijz_r * rijz_r + softSquared_r;
            float ai_r = this->G * d[jBody].m / std::pow(rijSquared_r, 1.5f);

            ax_r += ai_r * rijx_r;
            ay_r += ai_r * rijy_r;
            az_r += ai_r * rijz_r;
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax += ax_r;
        this->accelerations[iBody].ay += ay_r;
        this->accelerations[iBody].az += az_r;
    }
}



void SimulationNBodySimdOptim::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
