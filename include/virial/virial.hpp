#ifndef _VIRIAL_HPP_

#define _VIRIAL_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <string>   // for string

namespace simulationBox
{
    class SimulationBox;
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

/**
 * @namespace virial
 *
 * @brief Namespace for virial calculation
 */
namespace virial
{
    /**
     * @class Virial
     *
     * @brief Base class for virial calculation
     */
    class Virial
    {
      protected:
        std::string _virialType;

        linearAlgebra::Vec3D _virial;

      public:
        virtual ~Virial() = default;

        virtual void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &);

        void                               setVirial(const linearAlgebra::Vec3D &virial) { _virial = virial; }
        [[nodiscard]] linearAlgebra::Vec3D getVirial() const { return _virial; }

        [[nodiscard]] std::string getVirialType() const { return _virialType; }
    };

    /**
     * @class VirialMolecular
     *
     * @brief Class for virial calculation of molecular systems
     */
    class VirialMolecular : public Virial
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
    class VirialAtomic : public Virial
    {
      public:
        VirialAtomic() : Virial() { _virialType = "atomic"; }
    };

}   // namespace virial

#endif   // _VIRIAL_HPP_