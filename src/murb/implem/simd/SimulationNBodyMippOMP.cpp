#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>

#include "SimulationNBodyMippOMP.hpp"
#include "mipp.h"

SimulationNBodyMippOMP::SimulationNBodyMippOMP(const unsigned long nBodies, const std::string &scheme, const float soft,
                                           const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

/**
 * - Here we leverage the already inserted padding in the data structure to avoid the reminder of 
 * the loop.
 */
void SimulationNBodyMippOMP::computeBodiesAccelerationPadding()
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
    
    #pragma omp parallel for
    for (unsigned long iBody = 0; iBody < N; iBody+=1) {
        // accumulators
        mipp::Reg<float> r_ax(0.f);
        mipp::Reg<float> r_ay(0.f);
        mipp::Reg<float> r_az(0.f);

        const mipp::Reg<float> r_qx_i(qx[iBody]);
        const mipp::Reg<float> r_qy_i(qy[iBody]);
        const mipp::Reg<float> r_qz_i(qz[iBody]);

        for (unsigned long jBody = 0; jBody < N; jBody+=mipp::N<float>()) {
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

void SimulationNBodyMippOMP::computeOneIteration()
{
    this->computeBodiesAccelerationPadding(); 
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
