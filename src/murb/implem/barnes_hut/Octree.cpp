/**
 * @file Octree.cpp
 * @brief Implementation of the Octree data structure.
 */

#include "Octree.hpp"
#include <iostream>

template <typename T>
bool BoundingBox<T>::containsPoint(Point<T> point) {
    return (point.x >= center.x - halfDimension &&
            point.x <= center.x + halfDimension &&
            point.y >= center.y - halfDimension &&
            point.y <= center.y + halfDimension &&
            point.z >= center.z - halfDimension &&
            point.z <= center.z + halfDimension);
}

template <typename T>
Point<T> BoundingBox<T>::getCenter() {
    return center;
}

template <typename T>
T BoundingBox<T>::getRadius() {
    return halfDimension;
}

template <typename T>
Octree<T>::Octree(BoundingBox<T> boundary):
    empty(true), divided(false), boundary(boundary){
}

template <typename T>
void Octree<T>::explore() {
    if(!divided) {
        std::cout << "Center of mass: " << com.p.x << " " << com.p.y << " " << com.p.z << " " << com.m << std::endl;
    } else {
        northWestUp->explore();
        northWestDown->explore();
        northEastUp->explore();
        northEastDown->explore();
        southWestUp->explore();
        southWestDown->explore();
        southEastUp->explore();
        southEastDown->explore();
    }
}

template <typename T>
void Octree<T>::subdivide() {
    T x = this->boundary.getCenter().x;
    T y = this->boundary.getCenter().y;
    T z = this->boundary.getCenter().z;
    T h = this->boundary.getRadius();
    Point<T> p;
    p = {x - h / 2, y + h / 2, z + h / 2};
    BoundingBox<T> nwu(p, h / 2);
    p = {x - h / 2, y - h / 2, z + h / 2};
    BoundingBox<T> nwb(p, h / 2);
    p = {x + h / 2, y + h / 2, z + h / 2};
    BoundingBox<T> neu(p, h / 2);
    p = {x + h / 2, y - h / 2, z + h / 2};
    BoundingBox<T> neb(p, h / 2);
    p = {x - h / 2, y + h / 2, z - h / 2};
    BoundingBox<T> swu(p, h / 2);
    p = {x - h / 2, y - h / 2, z - h / 2};
    BoundingBox<T> swb(p, h / 2);
    p = {x + h / 2, y + h / 2, z - h / 2};
    BoundingBox<T> seu(p, h / 2);
    p = {x + h / 2, y - h / 2, z - h / 2};
    BoundingBox<T> seb(p, h / 2);
    this->northWestUp = std::make_unique<Octree<T>>(nwu);
    this->northWestDown = std::make_unique<Octree<T>>(nwb);
    this->northEastUp = std::make_unique<Octree<T>>(neu);
    this->northEastDown = std::make_unique<Octree<T>>(neb);
    this->southWestUp = std::make_unique<Octree<T>>(swu);
    this->southWestDown = std::make_unique<Octree<T>>(swb);
    this->southEastUp = std::make_unique<Octree<T>>(seu);
    this->southEastDown = std::make_unique<Octree<T>>(seb);
    divided = true;
}

template <typename T>
bool Octree<T>::insert(CoM<T> el) {
    // If the point is outside the boundary, return false
    if (!boundary.containsPoint(el.p)) {
        return false;
    }

    // If the node is empty and not divided, insert the point
    if (empty && !divided) {
        this->empty = false;
        this->com = el;
        return true;
    }

    // If the node is full but not divided, subdivide and redistribute
    if (!divided) {
        subdivide();
        insert(this->com); // Reinsert the existing point into a child node
    }

    // Attempt to insert the new point into one of the child nodes
    for (auto& child : {northWestUp.get(), northWestDown.get(), northEastUp.get(), northEastDown.get(),
                        southWestUp.get(), southWestDown.get(), southEastUp.get(), southEastDown.get()}) {
        if (child && child->insert(el)) {
            return true;
        }
    }

    return false; 
}

template <typename T>
CoM<T> Octree<T>::computeCenterOfMass() {
    CoM<T> com = {0.0, 0.0, 0.0, 0.0};
    // if the region is not divided and full, return the center of mass
    if(!divided && !empty) {
        return this->com; // return the center of mass of the current region
    } else if(divided){ // if the region is divided, then compute the center of mass of the children regions
        CoM<T> comNWU = northWestUp->computeCenterOfMass();
        CoM<T> comNWD = northWestDown->computeCenterOfMass();
        CoM<T> comNEU = northEastUp->computeCenterOfMass();
        CoM<T> comNED = northEastDown->computeCenterOfMass();
        CoM<T> comSWU = southWestUp->computeCenterOfMass();
        CoM<T> comSWD = southWestDown->computeCenterOfMass();
        CoM<T> comSEU = southEastUp->computeCenterOfMass();
        CoM<T> comSED = southEastDown->computeCenterOfMass();
        com.m = comNWU.m + comNWD.m + comNEU.m + comNED.m + comSWU.m + comSWD.m + comSEU.m + comSED.m;
        com.p.x = (comNWU.p.x * comNWU.m + comNWD.p.x * comNWD.m + comNEU.p.x * comNEU.m + comNED.p.x * comNED.m +
            comSWU.p.x * comSWU.m + comSWD.p.x * comSWD.m + comSEU.p.x * comSEU.m + comSED.p.x * comSED.m) / com.m;
        com.p.y = (comNWU.p.y * comNWU.m + comNWD.p.y * comNWD.m + comNEU.p.y * comNEU.m + comNED.p.y * comNED.m + 
            comSWU.p.y * comSWU.m + comSWD.p.y * comSWD.m + comSEU.p.y * comSEU.m + comSED.p.y * comSED.m) / com.m;
        com.p.z = (comNWU.p.z * comNWU.m + comNWD.p.z * comNWD.m + comNEU.p.z * comNEU.m + comNED.p.z * comNED.m + 
            comSWU.p.z * comSWU.m + comSWD.p.z * comSWD.m + comSEU.p.z * comSEU.m + comSED.p.z * comSED.m) / com.m;

        // update the center of mass of the current region
        this->com = com;
    }

    return com;
}

template <typename T>
std::vector<CoM<T>> Octree<T>::getCoM(CoM<T> b, float theta) {
    if(this->empty) {
        return std::vector<CoM<T>>(); // if the region is empty, return an empty vector
    }
    // construct a vector of center of mass such that the ration between the size of the region and the distance
    // between the center of mass and the point is less than theta
    std::vector<CoM<T>> coms;
    if(!divided) {
        coms.push_back(this->com); // if the region is not divided and full, it contains only one point
    } else {
        float d = boundary.getRadius() / b.p.distance(this->com.p);
        if (d < theta) {
            coms.push_back(this->com); // if the region is divided and the ratio is less than theta, add the center of mass
        } else { // if does not satisfy the condition, then compute the center of mass of the children regions
            std::vector<CoM<T>> comNWU = northWestUp->getCoM(b, theta);
            std::vector<CoM<T>> comNWD = northWestDown->getCoM(b, theta);
            std::vector<CoM<T>> comNEU = northEastUp->getCoM(b, theta);
            std::vector<CoM<T>> comNED = northEastDown->getCoM(b, theta);
            std::vector<CoM<T>> comSWU = southWestUp->getCoM(b, theta);
            std::vector<CoM<T>> comSWD = southWestDown->getCoM(b, theta);
            std::vector<CoM<T>> comSEU = southEastUp->getCoM(b, theta);
            std::vector<CoM<T>> comSED = southEastDown->getCoM(b, theta);
            coms.insert(coms.end(), comNWU.begin(), comNWU.end());
            coms.insert(coms.end(), comNWD.begin(), comNWD.end());
            coms.insert(coms.end(), comNEU.begin(), comNEU.end());
            coms.insert(coms.end(), comNED.begin(), comNED.end());
            coms.insert(coms.end(), comSWU.begin(), comSWU.end());
            coms.insert(coms.end(), comSWD.begin(), comSWD.end());
            coms.insert(coms.end(), comSEU.begin(), comSEU.end());
            coms.insert(coms.end(), comSED.begin(), comSED.end());
        
        }
    }

    return coms;
}

template <typename T>
void Octree<T>::reset() {
    if(divided) {
        northWestUp->reset();
        northWestDown->reset();
        northEastUp->reset();
        northEastDown->reset();
        southWestUp->reset();
        southWestDown->reset();
        southEastUp->reset();
        southEastDown->reset();
    }
    empty = true;
    divided = false;
    this->com = {0.0, 0.0, 0.0, 0.0};
}

/**
 * ------------------------------
 * Explicit template instantiation
 * ------------------------------
 */
template class Octree<float>;
template class Point<float>;
template class Octree<double>;
template class Point<double>;
template class BoundingBox<float>;
template class BoundingBox<double>;
template class CoM<float>;
template class CoM<double>;