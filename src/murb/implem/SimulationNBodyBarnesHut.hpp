#ifndef SIMULATION_N_BODY_BARNES_HUT_HPP_
#define SIMULATION_N_BODY_BARNES_HUT_HPP_

#include <string>
#include <vector>

#include "core/SimulationNBodyInterface.hpp"
#include "core/BarnesHutTree.hpp"

class SimulationNBodyBarnesHut : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyBarnesHut(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                             const unsigned long randInit = 0);
    virtual ~SimulationNBodyBarnesHut() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_BARNES_HUT_HPP_ */
