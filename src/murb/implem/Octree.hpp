#ifndef OCTREE_HPP
#define OCTREE_HPP

#include <vector>
#include <cmath>

/**
 * Is it really necessary to create a body?????
 */
template <typename T>
struct Point {
    T x;
    T y;
    T z;

    T distance(Point<T> p) {
        return sqrt((x - p.x) * (x - p.x) + (y - p.y) * (y - p.y) + (z - p.z) * (z - p.z));
    }
};

template <typename T>
struct CoM { // center of mass
    Point<T> p;
    T m;
};

template <typename T>
class BoundingBox 
{
private:
    Point<T> center;
    T halfDimension;

public: 
    BoundingBox(Point<T> center, T halfDimension): 
    center(center), halfDimension(halfDimension) {};

    Point<T> getCenter();
    T getRadius();
    bool containsPoint(Point<T> point); // check if the point is in the bounding box
};

template <typename T>
class Octree // point octree
{
public:
    Octree(BoundingBox<T> boundary);
    void subdivide(); // create four children that fully divide this quad into four same-sized quads
    bool insert(CoM<T> body);
    CoM<T> computeCenterOfMass();
    std::vector<CoM<T>> getCoM(CoM<T> b, float theta);
    void explore();
    void reset();

private:
    // int capacity; // assumed to be 1
    bool empty; // to check weather the region is full or not
    bool divided; // to check if the region is divided or not
    CoM<T> com; // center of mass
    BoundingBox<T> boundary; // the boundary of the region
    Octree<T> *northWestUp;
    Octree<T> *northEastUp;
    Octree<T> *southWestUp;
    Octree<T> *southEastUp;
    Octree<T> *northWestDown;
    Octree<T> *northEastDown;
    Octree<T> *southWestDown;
    Octree<T> *southEastDown;
};

#endif // OCTREE_HPP