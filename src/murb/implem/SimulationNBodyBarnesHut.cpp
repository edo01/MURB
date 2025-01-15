#include "SimulationNBodyBarnesHut.hpp"
#include <cmath>
#include <iostream>

void SimulationNBodyBarnesHut::initIteration() {
    for (auto& acc : accelerations) {
        acc.ax = 0.f;
        acc.ay = 0.f;
        acc.az = 0.f;
    }
}

void SimulationNBodyBarnesHut::insertParticle(TreeNode& node, const dataAoS_t<float>& particle) {
    if (node.isLeaf) {
        if (node.particles.empty()) {
            node.particles.push_back(particle);
            return;
        }

        node.isLeaf = false;
        for (const auto& existingParticle : node.particles) {
            insertParticle(node, existingParticle);
        }
        node.particles.clear();
    }

    int index = 0;
    if (particle.qx > (node.xmin + node.xmax) / 2) index |= 1;
    if (particle.qy > (node.ymin + node.ymax) / 2) index |= 2;
    if (particle.qz > (node.zmin + node.zmax) / 2) index |= 4;

    if (!node.children[index]) {
        float xMid = (node.xmin + node.xmax) / 2;
        float yMid = (node.ymin + node.ymax) / 2;
        float zMid = (node.zmin + node.zmax) / 2;

        if (index & 1) node.children[index] = std::make_unique<TreeNode>(xMid, node.xmax, yMid, node.ymax, zMid, node.zmax);
        else node.children[index] = std::make_unique<TreeNode>(node.xmin, xMid, node.ymin, yMid, node.zmin, zMid);
    }

    insertParticle(*node.children[index], particle);
}

void SimulationNBodyBarnesHut::computeMassDistribution(TreeNode& node) {
    if (node.isLeaf) {
        for (const auto& particle : node.particles) {
            node.mass += particle.m;
            node.cx += particle.qx * particle.m;
            node.cy += particle.qy * particle.m;
            node.cz += particle.qz * particle.m;
        }
        if (node.mass > 0) {
            node.cx /= node.mass;
            node.cy /= node.mass;
            node.cz /= node.mass;
        }
        return;
    }

    for (const auto& child : node.children) {
        if (child) {
            computeMassDistribution(*child);
            node.mass += child->mass;
            node.cx += child->cx * child->mass;
            node.cy += child->cy * child->mass;
            node.cz += child->cz * child->mass;
        }
    }
    if (node.mass > 0) {
        node.cx /= node.mass;
        node.cy /= node.mass;
        node.cz /= node.mass;
    }
}

void SimulationNBodyBarnesHut::computeForce(const TreeNode& node, const dataAoS_t<float>& target, accAoS_t<float>& acceleration) const {
    if (node.isLeaf) {
        for (const auto& particle : node.particles) {
            if (&particle == &target) continue;

            float dx = particle.qx - target.qx;
            float dy = particle.qy - target.qy;
            float dz = particle.qz - target.qz;
            float distSq = dx * dx + dy * dy + dz * dz + std::pow(this->soft, 2);
            float dist = std::sqrt(distSq);
            float force = this->G * particle.m / (distSq * dist);

            acceleration.ax += force * dx;
            acceleration.ay += force * dy;
            acceleration.az += force * dz;
        }
        return;
    }

    float dx = node.cx - target.qx;
    float dy = node.cy - target.qy;
    float dz = node.cz - target.qz;
    float distSq = dx * dx + dy * dy + dz * dz;

    if ((node.xmax - node.xmin) / std::sqrt(distSq) < this->theta) {
        float dist = std::sqrt(distSq);
        float force = this->G * node.mass / (distSq * dist);

        acceleration.ax += force * dx;
        acceleration.ay += force * dy;
        acceleration.az += force * dz;
    } else {
        for (const auto& child : node.children) {
            if (child) computeForce(*child, target, acceleration);
        }
    }
}

void SimulationNBodyBarnesHut::computeBodiesAcceleration() {
    const auto& bodies = this->getBodies().getDataAoS();
    const unsigned long nBodies = this->getBodies().getN();

    root = std::make_unique<TreeNode>(-1.f, 1.f, -1.f, 1.f, -1.f, 1.f);
    for (const auto& body : bodies) {
        insertParticle(*root, body);
    }

    computeMassDistribution(*root);

    for (unsigned long i = 0; i < nBodies; ++i) {
        computeForce(*root, bodies[i], this->accelerations[i]);
    }
}

void SimulationNBodyBarnesHut::computeOneIteration() {
    this->initIteration();
    this->computeBodiesAcceleration();
    this->bodies.updatePositionsAndVelocities(this->accelerations, this->dt);
}
