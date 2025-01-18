/**
 * @file SimulationNBodyMippV2.cpp
 * @brief Optimized implementation of the N-body simulation using mipp.
 * 
 * The triangular version of the algorithm is optimized using mipp library.
 * Here we have to use the reminder of the loop to compute the remaining bodies, since
 * the padding is not enough to avoid the reminder of the loop.
 * 
 * As in the mipp version, we implement two versions of the algorithm:
 * - The first version "SimulationNBodyMippV2::computeBodiesAcceleration" is the safe and
 *  correct version that doesn't assume any padding in the data structure. It inserts a
 * reminder of the loop at the end of the loop.
 * - The second version "SimulationNBodyMippV2::computeBodiesAccelerationMasq" uses the
 * masquerade load feature of mipp to avoid the reminder of the loop. This version is slower
 * than the padding version but is interesting to show the masquerade load feature of mipp.
 * 
 * As we noticed in the previous version, this version of the algorithm may be faster than
 * the n^2 algorithm but may not be the best choice for large n and for parallelization.
 */
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyMippV2.hpp"
#include "mipp.h"


SimulationNBodyMippV2::SimulationNBodyMippV2(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 27.f * (float)(nBodies * (nBodies-1)/2) + 15.f * nBodies;
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

void SimulationNBodyMippV2::initIteration()
{   
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;
    }
}

void SimulationNBodyMippV2::computeBodiesAcceleration()
{
    unsigned long jBody;
    const unsigned long N = this->getBodies().getN();

    //const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    // alias
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;
    const std::vector<float> &m = d.m;
    
    const mipp::Reg<float> r_softSquared = mipp::Reg<float>(this->soft*this->soft);
    const mipp::Reg<float> r_G = mipp::Reg<float>(this->G);

    const float softSquared = this->soft*this->soft;

    // flops = n*(n-1)/2 * 26
    for (unsigned long iBody = 0; iBody < N; iBody++) {
        // accumulators
        mipp::Reg<float> r_ax_i(0.f);
        mipp::Reg<float> r_ay_i(0.f);
        mipp::Reg<float> r_az_i(0.f);
        
        float ax_i=0, ay_i=0, az_i=0; // Local accumulators for acceleration

        const mipp::Reg<float> r_qx_i(qx[iBody]);
        const mipp::Reg<float> r_qy_i(qy[iBody]);
        const mipp::Reg<float> r_qz_i(qz[iBody]);
        const mipp::Reg<float> r_m_i(m[iBody]);  

        for (jBody = iBody+1; jBody < N-mipp::N<float>(); jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // VECSIZE flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // VECSIZE flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // VECSIZE flop


            // compute the || rij ||² distance between body i and body j for multiple bodies j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 VECSIZE flops
            
            r_rijSquared += r_softSquared; // VECSIZE flop
            mipp::Reg<float> r_ai = r_G / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 4 VECSIZE flops 
            mipp::Reg<float> r_aj(r_ai* r_m_i); // 1 VECSIZE flops
            r_ai = r_ai * mipp::Reg<float>(&m[jBody]);  // 1 VECSIZE flops

            r_ax_i += r_ai * r_rijx; // 2 VECSIZE flops
            r_ay_i += r_ai * r_rijy; // 2 VECSIZE flops
            r_az_i += r_ai * r_rijz; // 2 VECSIZE flops

            mipp::Reg<float> r_ax_j(&ax[jBody]);
            mipp::Reg<float> r_ay_j(&ay[jBody]);
            mipp::Reg<float> r_az_j(&az[jBody]);

            r_ax_j -= r_aj * r_rijx; // 2 VECSIZE flops
            r_ay_j -= r_aj * r_rijy; // 2 VECSIZE flops
            r_az_j -= r_aj * r_rijz; // 2 VECSIZE flops

            r_ax_j.store(&ax[jBody]);
            r_ay_j.store(&ay[jBody]);
            r_az_j.store(&az[jBody]);

        }

        // in the computation of the flops, the reminder of the loop is not taken into account
        // because is computed in the previous loop
        for(unsigned long j = jBody; j < N; j++){
            const float rijx = qx[j] - qx[iBody];
            const float rijy = qy[j] - qy[iBody];
            const float rijz = qz[j] - qz[iBody];

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            float ai = this->G / (rijSquared*std::sqrt(rijSquared));
            const float aj = ai * m[iBody];
            ai = ai * m[j];

            ax_i += ai * rijx;
            ay_i += ai * rijy;
            az_i += ai * rijz;

            ax[j] -= aj * rijx;
            ay[j] -= aj * rijy;
            az[j] -= aj * rijz;
        }

        ax[iBody] += mipp::sum(r_ax_i) + ax_i; // 5 flops
        ay[iBody] += mipp::sum(r_ay_i) + ay_i; // 5 flops
        az[iBody] += mipp::sum(r_az_i) + az_i; // 5 flops
    }
}

/**
 * Slow, but safe version of the computeBodiesAcceleration function using masquerade load
 */
void SimulationNBodyMippV2::computeBodiesAccelerationMasq(){
    const unsigned long N = this->getBodies().getN(); 

    //const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();

    // alias
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    const std::vector<float> &m  = d.m;
    std::vector<float> &ax       = this->accelerations.ax;
    std::vector<float> &ay       = this->accelerations.ay;
    std::vector<float> &az       = this->accelerations.az;

    // constants    
    const mipp::Reg<float> r_softSquared = mipp::Reg<float>(this->soft*this->soft);
    const mipp::Reg<float> r_G = mipp::Reg<float>(this->G);

    // utility vector for computing the masq
    mipp::vector<int> positions;
    for(int i = 0; i < mipp::N<int>(); i++){
        positions.push_back(i);
    }
    const mipp::Reg<int> r_positions(&positions[0]);

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < N; iBody++) {
        // accumulators
        mipp::Reg<float> r_ax_i(0.f);
        mipp::Reg<float> r_ay_i(0.f);
        mipp::Reg<float> r_az_i(0.f);
        
        mipp::Reg<float> r_qx_i(qx[iBody]);
        mipp::Reg<float> r_qy_i(qy[iBody]);
        mipp::Reg<float> r_qz_i(qz[iBody]);
        mipp::Reg<float> r_m_i(m[iBody]);  

        for (unsigned long jBody = iBody+1; jBody < N; jBody+=mipp::N<float>()) {
            const int remaining = std::min((unsigned long) mipp::N<float>(), N - jBody);
            mipp::Msk<mipp::N<float>()> mask = r_positions < remaining;

            // safe masquerade load
            mipp::Reg<float> r_qx_j = mipp::maskzlds<float>(mask, &qx[jBody]);
            mipp::Reg<float> r_qy_j = mipp::maskzlds<float>(mask, &qy[jBody]);
            mipp::Reg<float> r_qz_j = mipp::maskzlds<float>(mask, &qz[jBody]);
            mipp::Reg<float> r_m_j  = mipp::maskzlds<float>(mask, &m[jBody]);

            mipp::Reg<float> r_rijx = r_qx_j - r_qx_i; // 1 flop
            mipp::Reg<float> r_rijy = r_qy_j - r_qy_i; // 1 flop
            mipp::Reg<float> r_rijz = r_qz_j - r_qz_i; // 1 flop


            // compute the || rij ||² distance between body i and body j for multiple bodies j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_ai = r_G / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            mipp::Reg<float> r_aj(r_ai*r_m_i);
            r_ai = r_ai * mipp::Reg<float>(&m[jBody]); 

            r_ai = mipp::blend(r_ai, mipp::Reg<float>(0.f), mask); // TO CHECK


            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            r_ax_i += r_ai * r_rijx; 
            r_ay_i += r_ai * r_rijy; 
            r_az_i += r_ai * r_rijz;

            mipp::Reg<float> r_ax_j = mipp::maskzlds(mask, &ax[jBody]);
            mipp::Reg<float> r_ay_j = mipp::maskzlds(mask, &ay[jBody]);
            mipp::Reg<float> r_az_j = mipp::maskzlds(mask, &az[jBody]);

            r_ax_j -= r_aj * r_rijx;
            r_ay_j -= r_aj * r_rijy;
            r_az_j -= r_aj * r_rijz;

            r_ax_j.masksts(mask, &ax[jBody]);
            r_ay_j.masksts(mask, &ay[jBody]);
            r_az_j.masksts(mask, &az[jBody]);
        }

        ax[iBody] += mipp::sum(r_ax_i);
        ay[iBody] += mipp::sum(r_ay_i);
        az[iBody] += mipp::sum(r_az_i);
    }

}


void SimulationNBodyMippV2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
