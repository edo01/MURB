#ifndef SIMULATION_N_BODY_BARNES_HUT_HPP_
#define SIMULATION_N_BODY_BARNES_HUT_HPP_

#include <string>
#include <vector>
#include <memory>
#include "core/SimulationNBodyInterface.hpp"
#include "Octree.hpp"

class SimulationNBodyBarnesHut : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */
    std::unique_ptr<Octree<float>> root;
    const float theta = 0.1; /* Distance threshold for Barnes-Hut approximation. */
    const float range = 1e13; /* Default range for the simulation space. */
    
  public:
    SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                             const unsigned long randInit = 0);
    virtual ~SimulationNBodyBarnesHut() = default;

    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
    void computeMassDistribution();
    void buildTree();
    void clearTree();
};

#endif /* SIMULATION_N_BODY_BARNES_HUT_HPP_ */
