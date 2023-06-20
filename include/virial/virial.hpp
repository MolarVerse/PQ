#ifndef _VIRIAL_H_

#define _VIRIAL_H_

#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

#include <vector>

/**
 * @namespace virial
 *
 * @brief Namespace for virial calculation
 */
namespace virial
{
    class Virial;
    class VirialMolecular;
    class VirialAtomic;
}   // namespace virial

/**
 * @class Virial
 *
 * @brief Base class for virial calculation
 */
class virial::Virial
{
  protected:
    Vec3D _virial;

  public:
    virtual ~Virial() = default;

    virtual void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &);

    Vec3D getVirial() const { return _virial; }
    void  setVirial(const Vec3D &virial) { _virial = virial; }
};

/**
 * @class VirialMolecular
 *
 * @brief Class for virial calculation of molecular systems
 */
class virial::VirialMolecular : public virial::Virial
{
  public:
    void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &) override;
    void intraMolecularVirialCorrection(simulationBox::SimulationBox &);
};

/**
 * @class VirialAtomic
 *
 * @brief Class for virial calculation of atomic systems
 *
 */
class virial::VirialAtomic : public virial::Virial
{
};

#endif