#include "BarnesHutTree.hpp"
#include <cmath>


BarnesHutTree::BarnesHutTree(float minX, float minY, float minZ, float maxX, float maxY, float maxZ, float theta)
    : theta(theta) {
    root = std::make_unique<Node>(minX, minY, minZ, maxX, maxY, maxZ);
}


void BarnesHutTree::insert(const dataAoS_t<float>& body) {
    insert(*root, body);
}

void BarnesHutTree::insert(Node& node, const dataAoS_t<float>& body) {
    if (node.bodies.empty() && !node.children[0]) {
        node.bodies.push_back(body);
        updateCenterOfMass(node, body);
        return;
    }
    if (!node.children[0]) {
        float midX = (node.boundaryMinX + node.boundaryMaxX) / 2;
        float midY = (node.boundaryMinY + node.boundaryMaxY) / 2;
        float midZ = (node.boundaryMinZ + node.boundaryMaxZ) / 2;

        for (int i = 0; i < 8; ++i) {
            node.children[i] = std::make_unique<Node>(
                (i & 1) ? midX : node.boundaryMinX, // X 
                (i & 2) ? midY : node.boundaryMinY, // Y 
                (i & 4) ? midZ : node.boundaryMinZ, // Z 
                (i & 1) ? node.boundaryMaxX : midX,
                (i & 2) ? node.boundaryMaxY : midY,
                (i & 4) ? node.boundaryMaxZ : midZ
            );
        }

        for (const auto& existingBody : node.bodies) {
            int index = 0;
            if (existingBody.qx > midX) index |= 1;
            if (existingBody.qy > midY) index |= 2;
            if (existingBody.qz > midZ) index |= 4;

            insert(*node.children[index], existingBody);
        }

        node.bodies.clear();
    }

    float midX = (node.boundaryMinX + node.boundaryMaxX) / 2;
    float midY = (node.boundaryMinY + node.boundaryMaxY) / 2;
    float midZ = (node.boundaryMinZ + node.boundaryMaxZ) / 2;
    int index = 0;
    if (body.qx > midX) index |= 1;
    if (body.qy > midY) index |= 2;
    if (body.qz > midZ) index |= 4;

    
    insert(*node.children[index], body);

    updateCenterOfMass(node, body);
}


bool BarnesHutTree::isFarEnough(const Node& node, const dataAoS_t<float>& body) {
    float dx = node.centerX - body.qx;
    float dy = node.centerY - body.qy;
    float dz = node.centerZ - body.qz;
    float dist = std::sqrt(dx * dx + dy * dy + dz * dz);
    float size = node.boundaryMaxX - node.boundaryMinX;
    return (size / dist) < theta;
}


void BarnesHutTree::updateCenterOfMass(Node& node, const dataAoS_t<float>& body) {
    float totalMass = node.totalMass + body.m;
    node.centerX = (node.centerX * node.totalMass + body.qx * body.m) / totalMass;
    node.centerY = (node.centerY * node.totalMass + body.qy * body.m) / totalMass;
    node.centerZ = (node.centerZ * node.totalMass + body.qz * body.m) / totalMass;
    node.totalMass = totalMass;
}

void BarnesHutTree::computeAcceleration(const dataAoS_t<float>& body, accAoS_t<float>& acceleration, float soft, float G) {
    computeAcceleration(*root, body, acceleration, soft, G);
}

void BarnesHutTree::computeAcceleration(Node& node, const dataAoS_t<float>& body, accAoS_t<float>& acceleration, float soft, float G) {
    if (node.bodies.empty() && !node.children[0]) return; 

    if (isFarEnough(node, body) || node.bodies.size() == 1) {
        float dx = node.centerX - body.qx;
        float dy = node.centerY - body.qy;
        float dz = node.centerZ - body.qz;
        float distSquared = dx * dx + dy * dy + dz * dz + soft * soft;
        float dist = std::sqrt(distSquared);
        float force = G * node.totalMass / (distSquared * dist);

        acceleration.ax += force * dx;
        acceleration.ay += force * dy;
        acceleration.az += force * dz;
        return;
    }

    
    for (const auto& child : node.children) {
        if (child) {
            computeAcceleration(*child, body, acceleration, soft, G);
        }
    }
}



