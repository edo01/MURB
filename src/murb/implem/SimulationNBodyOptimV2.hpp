#ifndef SIMULATION_N_BODY_OPTIM_V2_HPP_
#define SIMULATION_N_BODY_OPTIM_V2_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyOptimV2 : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyOptimV2(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyOptimV2() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_OPTIM_V2_HPP_ */
