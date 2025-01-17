#ifndef BARNES_HUT_TREE_HPP_
#define BARNES_HUT_TREE_HPP_

#include <vector>
#include <memory>
#include "Bodies.hpp"

class BarnesHutTree {
  private:
    struct Node {
        float centerX, centerY, centerZ; /*!< Center of mass of the node. */
        float totalMass; /*!< Total mass of the bodies in this node. */
        float boundaryMinX, boundaryMinY, boundaryMinZ; /*!< Minimum boundaries of the node. */
        float boundaryMaxX, boundaryMaxY, boundaryMaxZ; /*!< Maximum boundaries of the node. */
        std::vector<dataAoS_t<float>> bodies; /*!< Bodies contained in this node. */
        std::unique_ptr<Node> children[8]; /*!< Child nodes (octants). */

        Node(float minX, float minY, float minZ, float maxX, float maxY, float maxZ)
            : centerX(0), centerY(0), centerZ(0), totalMass(0), 
              boundaryMinX(minX), boundaryMinY(minY), boundaryMinZ(minZ), 
              boundaryMaxX(maxX), boundaryMaxY(maxY), boundaryMaxZ(maxZ) {}
    };

    std::unique_ptr<Node> root; /*!< Root node of the Barnes-Hut tree. */
    float theta; /*!< Threshold for Barnes-Hut approximation. */

  public:
    BarnesHutTree(float minX, float minY, float minZ, float maxX, float maxY, float maxZ, float theta = 0.5f);
    void insert(const dataAoS_t<float>& body);
    void computeAcceleration(const dataAoS_t<float>& body, accAoS_t<float>& acceleration, float soft, float G);

  private:
    void insert(Node& node, const dataAoS_t<float>& body);
    void computeAcceleration(Node& node, const dataAoS_t<float>& body, accAoS_t<float>& acceleration, float soft, float G);
    bool isFarEnough(const Node& node, const dataAoS_t<float>& body);
    void updateCenterOfMass(Node& node, const dataAoS_t<float>& body);
};

#endif /* BARNES_HUT_TREE_HPP_ */
