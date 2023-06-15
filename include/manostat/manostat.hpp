#ifndef _MANOSTAT_H_

#define _MANOSTAT_H_

#include "virial.hpp"
#include "physicalData.hpp"

#include <vector>

/**
 * @class Manostat
 *
 * @brief Manostat is a base class for all manostats
 *
 */
class Manostat
{
protected:
    Vec3D _pressureVector = {0.0, 0.0, 0.0};
    double _pressure;
    double _targetPressure;

    double _timestep;

public:
    Manostat() = default;
    explicit Manostat(const double targetPressure) : _targetPressure(targetPressure) {}
    virtual ~Manostat() = default;

    void calculatePressure(PhysicalData &physicalData);
    virtual void applyManostat(simulationBox::SimulationBox &, PhysicalData &);

    // standard getters and setters
    void setTimestep(const double timestep) { _timestep = timestep; }
};

class BerendsenManostat : public Manostat
{
private:
    double _tau;

public:
    using Manostat::Manostat;
    explicit BerendsenManostat(const double targetPressure, const double tau) : Manostat(targetPressure), _tau(tau) {}

    void applyManostat(simulationBox::SimulationBox &, PhysicalData &) override;
};

#endif