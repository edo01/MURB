#ifndef SIMULATION_N_BODY_GPU_HPP_
#define SIMULATION_N_BODY_GPU_HPP_

#include <string>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyGPU : public SimulationNBodyInterface {
  protected:
    float* d_ax, *d_ay, *d_az; /*!< Acceleration components on the device. */
    float* d_qx, *d_qy, *d_qz, *d_m; /*!< Position components on the device. */

    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodyGPU(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyGPU();
    virtual void computeOneIteration();
};

#endif /* SIMULATION_N_BODY_GPU_HPP_ */