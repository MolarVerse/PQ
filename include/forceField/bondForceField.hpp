#ifndef _BOND_FORCE_FIELD_HPP_

#define _BOND_FORCE_FIELD_HPP_

#include "bond.hpp"

#include <cstddef>

namespace simulationBox
{
    class SimulationBox;
    class Molecule;
}   // namespace simulationBox

namespace potential
{
    class CoulombPotential;
    class NonCoulombPotential;
}   // namespace potential

namespace physicalData
{
    class PhysicalData;
}

namespace forceField
{
    /**
     * @class BondForceField inherits from Bond
     *
     * @brief force field object for single bond length
     *
     */
    class BondForceField : public connectivity::Bond
    {
      private:
        size_t _type;
        bool   _isLinker = false;

        double _equilibriumBondLength;
        double _forceConstant;

      public:
        BondForceField(simulationBox::Molecule *molecule1,
                       simulationBox::Molecule *molecule2,
                       size_t                   atomIndex1,
                       size_t                   atomIndex2,
                       size_t                   type)
            : connectivity::Bond(molecule1, molecule2, atomIndex1, atomIndex2), _type(type){};

        void calculateEnergyAndForces(const simulationBox::SimulationBox &,
                                      physicalData::PhysicalData &,
                                      const potential::CoulombPotential &,
                                      potential::NonCoulombPotential &);

        void setIsLinker(const bool isLinker) { _isLinker = isLinker; }
        void setEquilibriumBondLength(const double equilibriumBondLength) { _equilibriumBondLength = equilibriumBondLength; }
        void setForceConstant(const double forceConstant) { _forceConstant = forceConstant; }

        [[nodiscard]] size_t getType() const { return _type; }
        [[nodiscard]] bool   isLinker() const { return _isLinker; }
        [[nodiscard]] double getEquilibriumBondLength() const { return _equilibriumBondLength; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    };

}   // namespace forceField

#endif   // _BOND_FORCE_FIELD_HPP_