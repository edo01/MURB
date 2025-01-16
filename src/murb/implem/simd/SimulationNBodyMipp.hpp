#ifndef SIMULATION_N_BODY_MIPP_HPP_
#define SIMULATION_N_BODY_MIPP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyMipp : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyMipp(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyMipp() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
    void computeBodiesAccelerationPadding();
    void computeBodiesAccelerationMasq();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
