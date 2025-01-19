/**
 * @file SimulationNBodyMippOMP.cpp
 * @brief Optimized implementation of the N-body simulation using mipp and OpenMP.
 * 
 * We add OpenMP parallelization to the n^2 algorithm using mipp library. This version is
 * easier to parallelize than the triangular version of the algorithm. 
 * 
 * This version is the fastest version of the n^2 algorithm that we can get when using only
 * the CPU. However, for small problems, the mipp version of the algorithm is fastest since
 * it avoids the overhead of OpenMP.
 * 
 * We don't implement the safe versions of the algorithm since they are slower than the padding
 * one. We leverage the padding in the data structure to avoid the reminder of the loop.
 * 
 * flops = n² * 19 + 9 * n
 */
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
    this->flopsPerIte = (float)(20 * nBodies * nBodies + 9 * nBodies);
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

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
    
    // flops = (n² * 19 + 9*n)
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
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // VECSIZE flops
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // VECSIZE flops
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // VECSIZE flops

            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5*VECSIZE flops 
             
            r_rijSquared += r_softSquared; // VECSIZE flops
            mipp::Reg<float> r_mj(&m[jBody]);
            mipp::Reg<float> r_ai = r_G * r_mj / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 4 * VECSIZE * flops 
            
            r_ax += r_ai * r_rijx; r_ay += r_ai * r_rijy; r_az += r_ai * r_rijz; // 6 * VECSIZE flops
        } 

        ax[iBody] = mipp::sum(r_ax); // 3 flops
        ay[iBody] = mipp::sum(r_ay); // 3 flops
        az[iBody] = mipp::sum(r_az); // 3 flops
    }
}

void SimulationNBodyMippOMP::computeOneIteration()
{
    this->computeBodiesAccelerationPadding(); 
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
