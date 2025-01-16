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
    northWestUp = nullptr;
    northWestDown = nullptr;
    northEastUp = nullptr;
    northEastDown = nullptr;
    southWestUp = nullptr;
    southWestDown = nullptr;
    southEastUp = nullptr;
    southEastDown = nullptr;
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
    this->northWestUp = new Octree<T>(nwu);
    this->northWestDown = new Octree<T>(nwb);
    this->northEastUp = new Octree<T>(neu);
    this->northEastDown = new Octree<T>(neb);
    this->southWestUp = new Octree<T>(swu);
    this->southWestDown = new Octree<T>(swb);
    this->southEastUp = new Octree<T>(seu);
    this->southEastDown = new Octree<T>(seb);
    divided = true;
}

template <typename T>
bool Octree<T>::insert(CoM<T> el) {
    // if the point does not belong to the boundary, return false 
    if (!boundary.containsPoint(el.p)) {
        /*std::cout << "Point does not belong to the boundary" << std::endl;
        std::cout << "Point: " << el.p.x << " " << el.p.y << " " << el.p.z << std::endl;
        std::cout << "Boundary: " << boundary.getCenter().x << " " << boundary.getCenter().y << " " << boundary.getCenter().z << std::endl;
        */
        return false;
    }
    // if the region is not full, not divided, and the point belongs to the boundary, insert the point
    if (this->empty && !divided) { 
        /* std::cout << "Point belong to the boundary" << std::endl;
        std::cout << "Point: " << el.p.x << " " << el.p.y << " " << el.p.z << std::endl;
        std::cout << "Boundary: " << boundary.getCenter().x << " " << boundary.getCenter().y << " " << boundary.getCenter().z << std::endl;
        std::cout << "Boundary radius: " << boundary.getRadius() << std::endl; */
        this->empty = false;
        this->com = el;
        return true;
    
    // if the region is not divided and full, divide it and insert both the previous point and the new point
    }else if (!divided) { 
        subdivide();
        if(northWestUp->insert(this->com));
        else if(northWestDown->insert(this->com)) ;
        else if(northEastUp->insert(this->com)) ;
        else if(northEastDown->insert(this->com)) ;
        else if(southWestUp->insert(this->com)) ;
        else if(southWestDown->insert(this->com)) ;
        else if(southEastUp->insert(this->com)) ;
        else if(southEastDown->insert(this->com)) ;
    }
    // if the region is already divided and full, insert the point in one of the children regions
    if (northWestUp->insert(el)) return true;
    if (northWestDown->insert(el)) return true;
    if (northEastUp->insert(el)) return true;
    if (northEastDown->insert(el)) return true;
    if (southWestUp->insert(el)) return true;
    if (southWestDown->insert(el)) return true;
    if (southEastUp->insert(el)) return true;
    if (southEastDown->insert(el)) return true;

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
    delete northWestUp;
    delete northWestDown;
    delete northEastUp;
    delete northEastDown;
    delete southWestUp;
    delete southWestDown;
    delete southEastUp;
    delete southEastDown;
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