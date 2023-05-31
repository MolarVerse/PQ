#ifndef _INTEGRATOR_H_

#define _INTEGRATOR_H_

#include <string>

#include "simulationBox.hpp"
#include "timings.hpp"

/**
 * @class Integrator
 *
 * @brief Integrator is a base class for all integrators
 *
 */
class Integrator
{
protected:
    std::string _integratorType;

public:
    Integrator() = default;
    explicit Integrator(const std::string_view integratorType) : _integratorType(integratorType){};
    virtual ~Integrator() = default;

    // standard getter and setters
    [[nodiscard]] std::string_view getIntegratorType() const { return _integratorType; };

    virtual void firstStep(SimulationBox &, const Timings &) = 0;
    virtual void secondStep(SimulationBox &, const Timings &) = 0;
};

/**
 * @class VelocityVerlet inherits Integrator
 *
 * @brief VelocityVerlet is a class for velocity verlet integrator
 *
 */
class VelocityVerlet : public Integrator
{
public:
    explicit VelocityVerlet() : Integrator("VelocityVerlet"){};

    void firstStep(SimulationBox &, const Timings &) override;
    void secondStep(SimulationBox &, const Timings &) override;
};

#endif