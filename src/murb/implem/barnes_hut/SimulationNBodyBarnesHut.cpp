/**
 * @file SimulationNBodyBarnesHut.cpp
 * @brief Optimized implementation of the N-body simulation using the Barnes-Hut algorithm.
 * 
 * This version of the algorithm uses the Barnes-Hut algorithm to reduce the complexity of the
 * n^2 algorithm to n*log(n). The algorithm is implemented using an Octree data structure since 
 * we are simulating a 3D space. The algorithm is more complex than the n^2 algorithm and the   
 * flops are difficult to compute, since it depends on the number of bodies and the distribution
 * of the bodies in the space at each iteration. Unfortunately, the algorithm is not easily
 * parallelizable and we did not dig into this direction. Moreover, the algorithm does not show
 * any improvement in terms of performance compared to the n^2 algorithm, for small theta values.
 * In the latter case, the algorithm reduces to the n^2 algorithm and is even slower.  
 */

#include "SimulationNBodyBarnesHut.hpp"
#include <cmath>
#include <iostream>


SimulationNBodyBarnesHut::SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme, const float soft,
                                                   const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit){
        this->accelerations.resize(this->getBodies().getN());
        this->flopsPerIte = 0.f; // too complex to compute

        Point<float> center = {0.0f, 0.0f, 0.0f};
        root = std::make_unique<Octree<float>>(BoundingBox<float>(center, this->range));
    }

void
SimulationNBodyBarnesHut::buildTree()
{
    for(unsigned long i = 0; i < this->getBodies().getN(); i++){
        CoM<float> com;
        com.p.x = this->getBodies().getDataAoS()[i].qx;
        com.p.y = this->getBodies().getDataAoS()[i].qy;
        com.p.z = this->getBodies().getDataAoS()[i].qz;
        com.m = this->getBodies().getDataAoS()[i].m;
        root->insert(com);
    }
}

void SimulationNBodyBarnesHut::initIteration() {
    for(unsigned long i=0; i<this->getBodies().getN(); i++){
        this->accelerations[i].ax = 0.0f;
        this->accelerations[i].ay = 0.0f;
        this->accelerations[i].az = 0.0f;
    }
}

void SimulationNBodyBarnesHut::computeMassDistribution() {
    this->root->computeCenterOfMass();
}

void SimulationNBodyBarnesHut::computeBodiesAcceleration(){
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    
    const unsigned long nBodies = this->getBodies().getN(); 
    const float softSquared = this->soft*this->soft; 

    for(unsigned long iBody=0; iBody<nBodies; iBody++){
        CoM<float> com = {d[iBody].qx, d[iBody].qy, d[iBody].qz, d[iBody].m};

        const float qx_i = d[iBody].qx;
        const float qy_i = d[iBody].qy;
        const float qz_i = d[iBody].qz;

        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        std::vector<CoM<float>> coms = this->root->getCoM(com, this->theta);
        for(unsigned long jBody=0; jBody<coms.size(); jBody++){
            float rijx = coms[jBody].p.x - qx_i;
            float rijy = coms[jBody].p.y - qy_i;
            float rijz = coms[jBody].p.z - qz_i;
            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            const float ai = this->G * coms[jBody].m / (rijSquared*std::sqrt(rijSquared));
            ax += rijx * ai;
            ay += rijy * ai;
            az += rijz * ai;
        }
        
        this->accelerations[iBody].ax = ax;
        this->accelerations[iBody].ay = ay;
        this->accelerations[iBody].az = az;
    }
}

void SimulationNBodyBarnesHut::clearTree() {
    this->root->reset();
}

void SimulationNBodyBarnesHut::computeOneIteration() {
    this->initIteration();
    this->buildTree();
    this->computeMassDistribution();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
    this->clearTree();
}
