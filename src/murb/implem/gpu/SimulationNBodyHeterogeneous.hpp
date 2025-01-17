#ifndef SIMULATION_N_BODY_HETEROGENEOUS_HPP_
#define SIMULATION_N_BODY_HETEROGENEOUS_HPP_

#include <string>
#include <cuda_runtime.h>

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodyHeterogeneous : public SimulationNBodyInterface {
  protected:
    float* d_ax, *d_ay, *d_az; /*!< Acceleration components on the device. */
    float* d_qx, *d_qy, *d_qz, *d_m; /*!< Position components on the device. */

    float* p_ax, *p_ay, *p_az; /*!< Pinned memory for acceleration components. */
    accSoA_t<float> accelerations; /*!< Array of body acceleration structures. */

    int NTPB; /*!< Number of threads per block for the kernel. */
    int NB; /*!< Number of blocks for the kernel. */
    int N_res; // residual in the y direction to be computed in the cpu
    int N_y; // grid in the y direction
    int N_x; // grid in the x direction

  public:
    SimulationNBodyHeterogeneous(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodyHeterogeneous();
    virtual void computeOneIteration();
};

#endif /* SIMULATION_N_BODY_HETEROGENEOUS_HPP_ */
