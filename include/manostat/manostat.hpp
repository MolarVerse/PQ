#ifndef _MANOSTAT_H_

#define _MANOSTAT_H_

#include "physicalData.hpp"
#include "virial.hpp"

#include <vector>

namespace manostat
{
    class Manostat;
    class BerendsenManostat;
}   // namespace manostat

/**
 * @class Manostat
 *
 * @brief Manostat is a base class for all manostats
 *
 */
class manostat::Manostat
{
  protected:
    vector3d::Vec3D _pressureVector = {0.0, 0.0, 0.0};
    double          _pressure;
    double          _targetPressure;

    double _timestep;

  public:
    Manostat() = default;
    explicit Manostat(const double targetPressure) : _targetPressure(targetPressure) {}
    virtual ~Manostat() = default;

    void         calculatePressure(physicalData::PhysicalData &physicalData);
    virtual void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &);

    // standard getters and setters
    void setTimestep(const double timestep) { _timestep = timestep; }
};

class manostat::BerendsenManostat : public manostat::Manostat
{
  private:
    double _tau;
    double _compressability = 4.591e-5;   // TODO: make as input parameter

  public:
    using Manostat::Manostat;
    explicit BerendsenManostat(const double targetPressure, const double tau) : Manostat(targetPressure), _tau(tau) {}

    void applyManostat(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;
};

#endif