/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#ifndef _BOND_FORCE_FIELD_HPP_

#define _BOND_FORCE_FIELD_HPP_

#include "bond.hpp"

#include <cstddef>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
}   // namespace simulationBox

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace physicalData
{
    class PhysicalData;   // forward declaration
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

        /***************************
         * standard setter methods *
         ***************************/

        void setIsLinker(const bool isLinker) { _isLinker = isLinker; }
        void setEquilibriumBondLength(const double equilibriumBondLength) { _equilibriumBondLength = equilibriumBondLength; }
        void setForceConstant(const double forceConstant) { _forceConstant = forceConstant; }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getType() const { return _type; }
        [[nodiscard]] bool   isLinker() const { return _isLinker; }
        [[nodiscard]] double getEquilibriumBondLength() const { return _equilibriumBondLength; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
    };

}   // namespace forceField

#endif   // _BOND_FORCE_FIELD_HPP_