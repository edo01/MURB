#ifndef SIMULATION_N_BODY_MIPP_V2_HPP
#define SIMULATION_N_BODY_MIPP_V2_HPP

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyMippV2 : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyMippV2(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyMippV2() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
