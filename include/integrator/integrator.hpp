#ifndef _INTEGRATOR_H_

#define _INTEGRATOR_H_

#include "simulationBox.hpp"

#include <string>

namespace integrator
{
    class Integrator;
    class VelocityVerlet;
}   // namespace integrator

/**
 * @class Integrator
 *
 * @brief Integrator is a base class for all integrators
 *
 */
class integrator::Integrator
{
  protected:
    std::string _integratorType;
    double      _dt;

  public:
    Integrator() = default;
    explicit Integrator(const std::string_view integratorType) : _integratorType(integratorType){};
    virtual ~Integrator() = default;

    virtual void firstStep(simulationBox::SimulationBox &)  = 0;
    virtual void secondStep(simulationBox::SimulationBox &) = 0;

    void applyPBC(const simulationBox::SimulationBox &simBox, Vec3D &positions) const { simBox.applyPBC(positions); }
    void integrateVelocities(const double, simulationBox::Molecule &, const size_t) const;
    void integratePositions(const double, simulationBox::Molecule &, const size_t, const simulationBox::SimulationBox &) const;

    // standard getter and setters
    [[nodiscard]] std::string_view getIntegratorType() const { return _integratorType; }

    void setDt(const double dt) { _dt = dt; }
};

/**
 * @class VelocityVerlet inherits Integrator
 *
 * @brief VelocityVerlet is a class for velocity verlet integrator
 *
 */
class integrator::VelocityVerlet : public integrator::Integrator
{
  public:
    explicit VelocityVerlet() : Integrator("VelocityVerlet"){};

    void firstStep(simulationBox::SimulationBox &) override;
    void secondStep(simulationBox::SimulationBox &) override;
};

#endif