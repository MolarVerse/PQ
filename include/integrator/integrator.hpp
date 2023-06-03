#ifndef _INTEGRATOR_H_

#define _INTEGRATOR_H_

#include <string>

#include "simulationBox.hpp"

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
    double _dt;

public:
    Integrator() = default;
    explicit Integrator(const std::string_view integratorType) : _integratorType(integratorType){};
    virtual ~Integrator() = default;

    virtual void firstStep(SimulationBox &) = 0;
    virtual void secondStep(SimulationBox &) = 0;

    void applyPBC(const SimulationBox &simBox, Vec3D &positions) const { simBox._box.applyPBC(positions); };
    void integrateVelocities(const double, Molecule &, const size_t) const;
    void integratePositions(const double, Molecule &, const size_t, const SimulationBox &) const;

    // standard getter and setters
    [[nodiscard]] std::string_view getIntegratorType() const { return _integratorType; };

    void setDt(const double dt) { _dt = dt; };
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

    void firstStep(SimulationBox &) override;
    void secondStep(SimulationBox &) override;
};

#endif