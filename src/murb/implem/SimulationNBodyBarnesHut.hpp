#ifndef SIMULATION_N_BODY_BARNES_HUT_HPP_
#define SIMULATION_N_BODY_BARNES_HUT_HPP_

#include <string>
#include <vector>
#include <memory>
#include "core/SimulationNBodyInterface.hpp"

class TreeNode {
  public:
    float mass;        /*!< Total mass in the node. */
    float cx, cy, cz;  /*!< Center of mass coordinates. */
    float xmin, xmax;  /*!< X range of the node. */
    float ymin, ymax;  /*!< Y range of the node. */
    float zmin, zmax;  /*!< Z range of the node. */

    std::vector<dataAoS_t<float>> particles; /*!< Particles contained in the node. */
    std::unique_ptr<TreeNode> children[8];  /*!< Children of the node in the octree. */

    bool isLeaf; /*!< Is this node a leaf? */

    TreeNode(float xMin, float xMax, float yMin, float yMax, float zMin, float zMax)
        : mass(0), cx(0), cy(0), cz(0), xmin(xMin), xmax(xMax), ymin(yMin), ymax(yMax), zmin(zMin), zmax(zMax), isLeaf(true) {}
};

class SimulationNBodyBarnesHut : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    std::unique_ptr<TreeNode> root; /*!< Root of the octree. */
    float theta; /*!< Distance threshold for Barnes-Hut approximation. */

  public:
    SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                             const unsigned long randInit = 0, const float theta = 0.5f)
        : SimulationNBodyInterface(nBodies, scheme, soft, randInit), theta(theta) {
        this->accelerations.resize(this->getBodies().getN());
        float range = 1.0f; // Default range for the simulation space.
        root = std::make_unique<TreeNode>(-range, range, -range, range, -range, range);
    }
    virtual ~SimulationNBodyBarnesHut() = default;

    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
    void insertParticle(TreeNode& node, const dataAoS_t<float>& particle);
    void computeMassDistribution(TreeNode& node);
    void computeForce(const TreeNode& node, const dataAoS_t<float>& target, accAoS_t<float>& acceleration) const;
    void clearTree(TreeNode& node);
};

#endif /* SIMULATION_N_BODY_BARNES_HUT_HPP_ */
