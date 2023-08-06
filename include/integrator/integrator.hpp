#ifndef _INTEGRATOR_HPP_

#define _INTEGRATOR_HPP_

#include "molecule.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

#include <cstddef>
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

    void applyPBC(const simulationBox::SimulationBox &simBox, linearAlgebra::Vec3D &positions) const { simBox.applyPBC(positions); }

    void integrateVelocities(simulationBox::Molecule &, const size_t) const;
    void integratePositions(simulationBox::Molecule &, const size_t, const simulationBox::SimulationBox &) const;

    /********************************
     * standard getters and setters *
     ********************************/

    std::string_view getIntegratorType() const { return _integratorType; }
    double           getDt() const { return _dt; }

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

#endif   // _INTEGRATOR_HPP_