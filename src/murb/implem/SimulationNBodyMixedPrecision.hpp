#ifndef SIMULATION_N_BODY_MIXED_PRECISION_HPP_
#define SIMULATION_N_BODY_MIXED_PRECISION_HPP_

#include <vector>
#include <string>
#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyMixedPrecision : public SimulationNBodyInterface {
  protected:
    accAoS_t<float> *d_accelerations; /*!< GPU: Acceleration structure. */
    dataAoS_t<float> *d_bodies;       /*!< GPU: Bodies structure. */
    std::vector<accAoS_t<float>> accelerations; /*!< Host: Accelerations array. */

  public:
    SimulationNBodyMixedPrecision(const unsigned long nBodies, const std::string &scheme = "galaxy", 
                                   const float soft = 0.035f, const unsigned long randInit = 0);
    virtual ~SimulationNBodyMixedPrecision();

    virtual void computeOneIteration();
    void computeBodiesAcceleration();
};

#endif /* SIMULATION_N_BODY_MIXED_PRECISION_HPP_ */
