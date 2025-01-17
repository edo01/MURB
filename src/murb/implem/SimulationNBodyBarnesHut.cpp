#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>
#include <string>

#include "core/SimulationNBodyInterface.hpp"
#include "core/BarnesHutTree.hpp"
#include "SimulationNBodyBarnesHut.hpp"

SimulationNBodyBarnesHut::SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme, const float soft,
                                                   const unsigned long randInit)
    : SimulationNBodyInterface(nBodies, scheme, soft, randInit)
{
    this->flopsPerIte = 20.f * (float)this->getBodies().getN() * (float)this->getBodies().getN();
    this->accelerations.resize(this->getBodies().getN());
}

void SimulationNBodyBarnesHut::initIteration()
{
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        this->accelerations[iBody].ax = 0.f;
        this->accelerations[iBody].ay = 0.f;
        this->accelerations[iBody].az = 0.f;
    }
}

void SimulationNBodyBarnesHut::computeBodiesAcceleration()
{
    const std::vector<dataAoS_t<float>> &d = this->getBodies().getDataAoS();
    
    // Get bounds of the simulation space
    auto [minX, maxX, minY, maxY, minZ, maxZ] = this->getBodies().getBounds();
    
    // Construct Barnes-Hut tree
    BarnesHutTree tree(minX, minY, minZ, maxX, maxY, maxZ);
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        tree.insert(d[iBody]);
    }
    
    // Compute accelerations using the Barnes-Hut tree
    for (unsigned long iBody = 0; iBody < this->getBodies().getN(); iBody++) {
        tree.computeAcceleration(d[iBody], this->accelerations[iBody], this->soft, this->G);
    }
}


void SimulationNBodyBarnesHut::computeOneIteration()
{
    this->initIteration();
    this->computeBodiesAcceleration();
    // time integration
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
