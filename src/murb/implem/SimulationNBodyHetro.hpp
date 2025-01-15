#ifndef SIMULATION_N_BODY_HETRO_HPP_
#define SIMULATION_N_BODY_HETRO_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyHetro : public SimulationNBodyInterface {
  protected:
    accAoS_t<float> *d_accelerations; /*!< Acceleration structure on the device. */
    dataAoS_t<float> *d_bodies;       /*!< Bodies structure on the device. */
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyHetro(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyHetro();
    virtual void computeOneIteration();
};

#endif /* SIMULATION_N_BODY_HETRO_HPP_ */
