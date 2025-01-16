#include "SimulationNBodyBarnesHut.hpp"
#include <cmath>
#include <iostream>


SimulationNBodyBarnesHut::SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme, const float soft,
                                                   const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit){
        this->accelerations.resize(this->getBodies().getN());
        this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();

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

        std::vector<CoM<float>> coms = this->root->getCoM(com, this->theta);
        //printf("testing body %lu: %f %f %f %f\n", iBody, d[iBody].qx, d[iBody].qy, d[iBody].qz, d[iBody].m);
        for(unsigned long jBody=0; jBody<coms.size(); jBody++){
            //printf("\t testing com %lu: %f %f %f %f\n", jBody, coms[jBody].p.x, coms[jBody].p.y, coms[jBody].p.z, coms[jBody].m);
            float rijx = coms[jBody].p.x - qx_i;
            float rijy = coms[jBody].p.y - qy_i;
            float rijz = coms[jBody].p.z - qz_i;
            float rijSquared = rijx*rijx + rijy*rijy + rijz*rijz;
            rijSquared += softSquared;
            const float ai = this->G * coms[jBody].m / (rijSquared*std::sqrt(rijSquared));
            this->accelerations[iBody].ax += rijx * ai;
            this->accelerations[iBody].ay += rijy * ai;
            this->accelerations[iBody].az += rijz * ai;
        }
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
