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
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.ax.resize(this->getBodies().getN());
    this->accelerations.ay.resize(this->getBodies().getN());
    this->accelerations.az.resize(this->getBodies().getN());
}

void SimulationNBodyMippV2::initIteration()
{   
    // memset to zero maybe is faster??
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations.ax[iBody] = 0.f;
        this->accelerations.ay[iBody] = 0.f;
        this->accelerations.az[iBody] = 0.f;
    }
}

void SimulationNBodyMippV2::computeBodiesAcceleration()
{
    unsigned long jBody;

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

    // flops = n² * 20
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        // accumulators
        mipp::Reg<float> r_ax_i(0.f);
        mipp::Reg<float> r_ay_i(0.f);
        mipp::Reg<float> r_az_i(0.f);
        
        mipp::Reg<float> r_qx_i(qx[iBody]);
        mipp::Reg<float> r_qy_i(qy[iBody]);
        mipp::Reg<float> r_qz_i(qz[iBody]);
        mipp::Reg<float> r_m_i(m[iBody]);  

        for (jBody = iBody+1; jBody < this->getBodies().getN()-mipp::N<float>(); jBody+=mipp::N<float>()) {
            mipp::Reg<float> r_rijx = mipp::Reg<float>(&qx[jBody]) - r_qx_i; // 1 flop
            mipp::Reg<float> r_rijy = mipp::Reg<float>(&qy[jBody]) - r_qy_i; // 1 flop
            mipp::Reg<float> r_rijz = mipp::Reg<float>(&qz[jBody]) - r_qz_i; // 1 flop


            // compute the || rij ||² distance between body i and body j for multiple bodies j
            mipp::Reg<float> r_rijSquared = r_rijx*r_rijx + r_rijy*r_rijy + r_rijz*r_rijz; // 5 flops // TRY MAC OPERATION 
            
            // compute the acceleration value between body i and body j: || ai || = G.mj / (|| rij ||² + e²)^{3/2}
            r_rijSquared += r_softSquared;
            mipp::Reg<float> r_ai = r_G / (r_rijSquared*mipp::sqrt(r_rijSquared)); // 5 flops // TRY WITH RSQRT OPERATION
            mipp::Reg<float> r_aj(r_ai* r_m_i);
            r_ai = r_ai * mipp::Reg<float>(&m[jBody]); 

            // add the acceleration value into the acceleration vector: ai += || ai ||.rij
            r_ax_i += r_ai * r_rijx; 
            r_ay_i += r_ai * r_rijy; 
            r_az_i += r_ai * r_rijz;

            mipp::Reg<float> r_ax_j(&ax[jBody]);
            mipp::Reg<float> r_ay_j(&ay[jBody]);
            mipp::Reg<float> r_az_j(&az[jBody]);

            r_ax_j -= r_aj * r_rijx;
            r_ay_j -= r_aj * r_rijy;
            r_az_j -= r_aj * r_rijz;

            r_ax_j.store(&ax[jBody]);
            r_ay_j.store(&ay[jBody]);
            r_az_j.store(&az[jBody]);

        }

        for(unsigned long j = jBody; j < this->getBodies().getN(); j++){
            const float rijx = qx[j] - qx[iBody];
            const float rijy = qy[j] - qy[iBody];
            const float rijz = qz[j] - qz[iBody];

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            float ai = this->G / (rijSquared*std::sqrt(rijSquared));
            const float aj = ai * m[iBody];
            ai = ai * m[j];

            ax[iBody] += ai * rijx;
            ay[iBody] += ai * rijy;
            az[iBody] += ai * rijz;

            ax[j] -= aj * rijx;
            ay[j] -= aj * rijy;
            az[j] -= aj * rijz;
        }

        ax[iBody] += mipp::sum(r_ax_i);
        ay[iBody] += mipp::sum(r_ay_i);
        az[iBody] += mipp::sum(r_az_i);
    }

    /*for(unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++){
        unsigned long start = this->getBodies().getN() - ((this->getBodies().getN() - iBody - 1) % mipp::N<float>());

        const float qx_i = qx[iBody];
        const float qy_i = qy[iBody];
        const float qz_i = qz[iBody];
        const float m_i = m[iBody];

        float ax_i = 0.0f, ay_i = 0.0f, az_i = 0.0f; // Local accumulators for acceleration
        
        for(unsigned long j = start; j < this->getBodies().getN(); j++){
            printf("iBody: %lu, j: %lu\n", iBody, j);
            const float rijx = qx[j] - qx_i;
            const float rijy = qy[j] - qy_i;
            const float rijz = qz[j] - qz_i;

            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            float ai = this->G / (rijSquared*std::sqrt(rijSquared));
            const float aj = ai * m_i;
            ai = ai * m[j];

            ax_i += ai * rijx;
            ay_i += ai * rijy;
            az_i += ai * rijz;

            ax[j] -= aj * rijx;
            ay[j] -= aj * rijy;
            az[j] -= aj * rijz;
        }

        ax[iBody] += ax_i;
        ay[iBody] += ay_i;
        az[iBody] += az_i;
    }*/
}

void SimulationNBodyMippV2::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
