#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(ENABLE_VECTO) && (defined(__ARM_NEON__) || defined(__ARM_NEON))

#include <arm_neon.h>
#include "SimulationNBodySimd.hpp"

SimulationNBodySimd::SimulationNBodySimd(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodySimd::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}


void SimulationNBodySimd::computeBodiesAcceleration() {
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN();
    const float softSquared_r = std::pow(this->soft, 2); 

    for (unsigned long iBody = 0; iBody < nBodies; iBody++) {
        float ax_r = 0.0f, ay_r = 0.0f, az_r = 0.0f; // Local accumulators for acceleration

        for (unsigned long jBody = 0; jBody < nBodies; jBody++) {
            // Load scalar values for iBody
            float qx_i_r = d[iBody].qx;
            float qy_i_r = d[iBody].qy;
            float qz_i_r = d[iBody].qz;

            // Load scalar values for jBody
            float qx_j_r = d[jBody].qx;
            float qy_j_r = d[jBody].qy;
            float qz_j_r = d[jBody].qz;
            float mj_r   = d[jBody].m;

            // Compute rij components
            float32x4_t rijx_r = vsubq_f32(vdupq_n_f32(qx_j_r), vdupq_n_f32(qx_i_r));
            float32x4_t rijy_r = vsubq_f32(vdupq_n_f32(qy_j_r), vdupq_n_f32(qy_i_r));
            float32x4_t rijz_r = vsubq_f32(vdupq_n_f32(qz_j_r), vdupq_n_f32(qz_i_r));

            // Compute rij squared
            float32x4_t rijx2_r = vmulq_f32(rijx_r, rijx_r);
            float32x4_t rijy2_r = vmulq_f32(rijy_r, rijy_r);
            float32x4_t rijz2_r = vmulq_f32(rijz_r, rijz_r);
            float32x4_t rijSquared_r = vaddq_f32(vaddq_f32(rijx2_r, rijy2_r), rijz2_r);

            // Add softSquared to rijSquared
            float32x4_t softSquared_r_v = vdupq_n_f32(softSquared_r);
            rijSquared_r = vaddq_f32(rijSquared_r, softSquared_r_v);

            //  G * mj / (rijSquared + e²)^(3/2)
            float32x4_t rijCubed_r = vmulq_f32(rijSquared_r, vmulq_f32(rijSquared_r, rijSquared_r));
            float32x4_t invRijCubed_r = vrsqrteq_f32(rijCubed_r); // Approximate 1/sqrt(rijCubed)
            float32x4_t ai_r = vmulq_f32(vdupq_n_f32(this->G * mj_r), invRijCubed_r);

            // Accumulate acceleration components
            float32x4_t ax_r_v = vmulq_f32(ai_r, rijx_r);
            float32x4_t ay_r_v = vmulq_f32(ai_r, rijy_r);
            float32x4_t az_r_v = vmulq_f32(ai_r, rijz_r);

            // Horizontal add to scalar accumulators
            ax_r += vaddvq_f32(ax_r_v);
            ay_r += vaddvq_f32(ay_r_v);
            az_r += vaddvq_f32(az_r_v);
        }

        // Store the accumulated accelerations
        this->accelerations[iBody].ax += ax_r;
        this->accelerations[iBody].ay += ay_r;
        this->accelerations[iBody].az += az_r;
    }
}


void SimulationNBodySimd::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
#endif