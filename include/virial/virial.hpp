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
    std::string _virialType;

    vector3d::Vec3D _virial;

  public:
    virtual ~Virial() = default;

    virtual void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &);

    vector3d::Vec3D getVirial() const { return _virial; }
    void            setVirial(const vector3d::Vec3D &virial) { _virial = virial; }

    std::string getVirialType() const { return _virialType; }
};

/**
 * @class VirialMolecular
 *
 * @brief Class for virial calculation of molecular systems
 */
class virial::VirialMolecular : public virial::Virial
{
  public:
    VirialMolecular() : Virial() { _virialType = "molecular"; }

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
  public:
    VirialAtomic() : Virial() { _virialType = "atomic"; }
};

#endif