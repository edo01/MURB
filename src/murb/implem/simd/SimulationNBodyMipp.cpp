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
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

/**
 * Feature of this version:
 * - n^2 algorithm
 * - mipp
 * - reminder of the loop handled separately
 */
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
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // 4 flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // 4 flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // 4 flop


            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz;
        } 

        ax[iBody] = mipp::sum(r_ax);
        ay[iBody] = mipp::sum(r_ay);
        az[iBody] = mipp::sum(r_az);

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


/**
 * - Here we leverage the already inserted padding in the data structure to avoid the reminder of 
 * the loop.
 */
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
    
    // flops = n² * 20
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

        for (jBody = 0; jBody < N; jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // 4 flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // 4 flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // 4 flop


            // compute the || rij ||² distance between body i and body j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value 
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            
            // accumulate the acceleration value
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz;
        } 

        ax[iBody] = mipp::sum(r_ax);
        ay[iBody] = mipp::sum(r_ay);
        az[iBody] = mipp::sum(r_az);
    }
}

/**
 * Every iteration a masq is added 
 */
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
