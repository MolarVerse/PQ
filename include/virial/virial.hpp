#ifndef _VIRIAL_HPP_

#define _VIRIAL_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <string>   // for string

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

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
     *
     * @details implements virial calculation, which is valid for both atomic and molecular systems
     */
    class Virial
    {
      protected:
        std::string _virialType;

        linearAlgebra::Vec3D _virial;

      public:
        virtual ~Virial() = default;

        virtual void calculateVirial(simulationBox::SimulationBox &, physicalData::PhysicalData &);

        void setVirial(const linearAlgebra::Vec3D &virial) { _virial = virial; }

        [[nodiscard]] linearAlgebra::Vec3D getVirial() const { return _virial; }
        [[nodiscard]] std::string          getVirialType() const { return _virialType; }
    };

    /**
     * @class VirialMolecular
     *
     * @brief Class for virial calculation of molecular systems
     *
     * @details overrides calculateVirial() function to include intra-molecular virial correction
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
     * @details dummy class for atomic systems, since no virial correction is needed
     *
     */
    class VirialAtomic : public Virial
    {
      public:
        VirialAtomic() : Virial() { _virialType = "atomic"; }
    };

}   // namespace virial

#endif   // _VIRIAL_HPP_