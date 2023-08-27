#ifndef _INTRA_NON_BONDED_MAP_HPP_

#define _INTRA_NON_BONDED_MAP_HPP_

#include "defaults.hpp"
#include "intraNonBondedContainer.hpp"

#include <vector>   // for vector

namespace simulationBox
{
    class SimulationBox;
    class Molecule;
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;
}   // namespace physicalData

namespace potential
{
    class CoulombPotential;
    class NonCoulombPotential;
}   // namespace potential

namespace intraNonBonded
{
    /**
     * @class IntraNonBondedInteraction
     *
     * @brief
     */
    class IntraNonBondedMap
    {
      private:
        simulationBox::Molecule *_molecule;
        IntraNonBondedContainer *_intraNonBondedType;

      public:
        explicit IntraNonBondedMap(simulationBox::Molecule *molecule, IntraNonBondedContainer *intraNonBondedType)
            : _molecule(molecule), _intraNonBondedType(intraNonBondedType)
        {
        }

        void calculate(const potential::CoulombPotential *,
                       potential::NonCoulombPotential *,
                       const simulationBox::SimulationBox &,
                       physicalData::PhysicalData &);

        [[nodiscard]] IntraNonBondedContainer      *getIntraNonBondedType() const { return _intraNonBondedType; }
        [[nodiscard]] simulationBox::Molecule      *getMolecule() const { return _molecule; }
        [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const { return _intraNonBondedType->getAtomIndices(); }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_MAP_HPP_