#ifndef SIMULATION_N_BODY_SIMD_HPP_
#define SIMULATION_N_BODY_SIMD_HPP_

#include <string>

#if defined(ENABLE_VECTO) && (defined(__ARM_NEON__) || defined(__ARM_NEON))

#include "core/SimulationNBodyInterface.hpp"

class SimulationNBodySimd : public SimulationNBodyInterface {
  protected:
    std::vector<accAoS_t<float>> accelerations; /*!< Array of body acceleration structures. */

  public:
    SimulationNBodySimd(const unsigned long nBodies, const std::string &scheme = "galaxy", const float soft = 0.035f,
                         const unsigned long randInit = 0);
    virtual ~SimulationNBodySimd() = default;
    virtual void computeOneIteration();

  protected:
    void initIteration();
    void computeBodiesAcceleration();
};

#endif // ENABLE_VECTO && (__ARM_NEON__ || __ARM_NEON)

#endif /* SIMULATION_N_BODY_SIMD_HPP_ */
