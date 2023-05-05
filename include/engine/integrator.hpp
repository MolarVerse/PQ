#ifndef _INTEGRATOR_H_

#define _INTEGRATOR_H_

#include <string>

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
    explicit Integrator(std::string_view integratorType) : _integratorType(integratorType){};
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
};

#endif