#ifndef SIMULATION_N_BODY_MIPP_OMP_HPP_
#define SIMULATION_N_BODY_MIPP_OMP_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyMippOMP : public SimulationNBodyInterface {
  protected:
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyMippOMP(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyMippOMP() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAccelerationPadding();
};

#endif /* SIMULATION_N_BODY_NAIVE_HPP_ */
