/**
 * @file SimulationNBodyMipp.cpp
 * @brief Optimized implementation of the N-body simulation using mipp.
 * 
 * The code is optimized using transformations of the original code and mipp library. The algorithm used 
 * is the n^2 algorithm.
 * We implement three versions of the algorithm:
 * - The first version "SimulationNBodyMipp::SimulationNBodyMipp" is the safe and correct version
 *   that doesn't assume any padding in the data structure. It inserts a reminder of the loop at
 *   the end of the loop.
 * - The second version "SimulationNBodyMipp::SimulationNBodyMippPadding" assumes that the data
 *   structure is padded and avoids the reminder of the loop. This is the fastest version and 
 *   the one that should be used in practice since the padding is already inserted in the data
 * - The third version "SimulationNBodyMipp::SimulationNBodyMippMasq" uses the masquerade load
 *   feature of mipp to avoid the reminder of the loop. This version is slower than the padding
 *   version but is interesting to show the masquerade load feature of mipp.
 * 
 * Please note that the number of flops remains the same as the non-simd version of the algorithm,
 * since as we increase the number of operations per cycle, we decrease the number of cycles:
 * 
 * flops = n² * 19 + 9 * n
 */
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyMipp.hpp"
#include "mipp.h"

SimulationNBodyMipp::SimulationNBodyMipp(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = (float)(19  * nBodies * nBodies + 9 * nBodies);
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

void SimulationNBodyMipp::computeBodiesAcceleration()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    const unsigned long N = this->getBodies().getN();

    // alias
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    const std::vector<float> &m = d.m;
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;
    
    // constants
    const mipp::Reg<float> r_softSquared = mipp::Reg<float>(this->soft*this->soft);
    const mipp::Reg<float> r_G = mipp::Reg<float>(this->G);
    const float softSquared = this->soft*this->soft;
    
    // flops = n² * 20
    //#pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody+=1) {
        // accumulators
        mipp::Reg<float> r_ax(0.f);
        mipp::Reg<float> r_ay(0.f);
        mipp::Reg<float> r_az(0.f);

        const float qx_i = qx[iBody];
        const float qy_i = qy[iBody];
        const float qz_i = qz[iBody];

        float ax_i = 0.0f, ay_i = 0.0f, az_i = 0.0f; // Local accumulators for acceleration

        const mipp::Reg<float> r_qx_i(qx_i);
        const mipp::Reg<float> r_qy_i(qy_i);
        const mipp::Reg<float> r_qz_i(qz_i);
        unsigned long jBody;

        for (jBody = 0; jBody < N - mipp::N<float>(); jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // 1 VECSIZE flops
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // 1 VECSIZE flops
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // 1 VECSIZE flops


            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 VECSIZE flops
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared; // 1 VECSIZE flops
            mipp::Reg<float> r_mj(&m[jBody]); 
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 VECSIZE flops
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz; // 6 VECSIZE flops
        } 

        ax[iBody] = mipp::sum(r_ax); // 3 flops
        ay[iBody] = mipp::sum(r_ay); // 3 flops
        az[iBody] = mipp::sum(r_az); // 3 flops

        // reminder of the loop
        for(unsigned long j = jBody; j < N; j++){
            const float rijx = qx[j] - qx_i;
            const float rijy = qy[j] - qy_i;
            const float rijz = qz[j] - qz_i;

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            float ai = this->G / (rijSquared*std::sqrt(rijSquared));
            ai = ai * m[j];

            ax_i += ai * rijx; ay_i += ai * rijy; az_i += ai * rijz;
        }

        ax[iBody] += ax_i; ay[iBody] += ay_i; az[iBody] += az_i;
    }
}

void SimulationNBodyMipp::computeBodiesAccelerationPadding()
{
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    const unsigned long N = this->getBodies().getN();

    // alias
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    const std::vector<float> &m = d.m;
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;
    
    // constants
    const mipp::Reg<float> r_softSquared = mipp::Reg<float>(this->soft*this->soft);
    const mipp::Reg<float> r_G = mipp::Reg<float>(this->G);
    
    // flops = n² * 19 + 9*n
    //#pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody+=1) {
        // accumulators
        mipp::Reg<float> r_ax(0.f);
        mipp::Reg<float> r_ay(0.f);
        mipp::Reg<float> r_az(0.f);

        const mipp::Reg<float> r_qx_i(qx[iBody]);
        const mipp::Reg<float> r_qy_i(qy[iBody]);
        const mipp::Reg<float> r_qz_i(qz[iBody]);
        unsigned long jBody;

        #pragma unroll 4
        for (jBody = 0; jBody < N; jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // VECSIZE flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // VECSIZE flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // VECSIZE flop

            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5*VECSIZE flops
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared; // VECSIZE flop
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 4*VECSIZE flops
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz; // 6*VECSIZE flops
        } 

        ax[iBody] = mipp::sum(r_ax); // 3 flops
        ay[iBody] = mipp::sum(r_ay); // 3 flops
        az[iBody] = mipp::sum(r_az); // 3 flops
    }
}

void SimulationNBodyMipp::computeBodiesAccelerationMasq(){
    const dataSoA_t<float> &d = this->getBodies().getDataSoA();
    const unsigned long N = this->getBodies().getN();

    // alias
    const std::vector<float> &qx = d.qx;
    const std::vector<float> &qy = d.qy;
    const std::vector<float> &qz = d.qz;
    const std::vector<float> &m = d.m;
    std::vector<float> &ax = this->accelerations.ax;
    std::vector<float> &ay = this->accelerations.ay;
    std::vector<float> &az = this->accelerations.az;
    
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
    //#pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody+=1) {
        // accumulators
        mipp::Reg<float> r_ax(0.f);
        mipp::Reg<float> r_ay(0.f);
        mipp::Reg<float> r_az(0.f);

        const float qx_i = qx[iBody];
        const float qy_i = qy[iBody];
        const float qz_i = qz[iBody];

        const mipp::Reg<float> r_qx_i(qx_i);
        const mipp::Reg<float> r_qy_i(qy_i);
        const mipp::Reg<float> r_qz_i(qz_i);

        for (unsigned long jBody = 0; jBody < N; jBody+=mipp::N<float>()) {
            // mask for the last iteration
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

            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            
            // not necessary because r_mj acts as a masq
            //r_ai = mipp::blend(r_ai, mipp::Reg<float>(0.f), mask);
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz;
        } 

        ax[iBody] = mipp::sum(r_ax);
        ay[iBody] = mipp::sum(r_ay);
        az[iBody] = mipp::sum(r_az);
    }  
}

void SimulationNBodyMipp::computeOneIteration()
{
    this->computeBodiesAccelerationPadding(); 
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
