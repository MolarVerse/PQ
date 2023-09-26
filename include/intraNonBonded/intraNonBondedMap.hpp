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

#ifndef _INTRA_NON_BONDED_MAP_HPP_

#define _INTRA_NON_BONDED_MAP_HPP_

#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "vector3d.hpp"                  // for Vec3D

#include <cstddef>   // for size_t
#include <utility>   // for pair
#include <vector>    // for vector

namespace simulationBox
{
    class SimulationBox;   // forward declaration
    class Molecule;        // forward declaration
}   // namespace simulationBox

namespace physicalData
{
    class PhysicalData;   // forward declaration
}

namespace potential
{
    class CoulombPotential;      // forward declaration
    class NonCoulombPotential;   // forward declaration
}   // namespace potential

namespace intraNonBonded
{
    /**
     * @class IntraNonBondedMap
     *
     * @brief defines a map for a single molecule to its intra non bonded interactions
     */
    class IntraNonBondedMap
    {
      private:
        simulationBox::Molecule *_molecule;
        IntraNonBondedContainer *_intraNonBondedContainer;

      public:
        explicit IntraNonBondedMap(simulationBox::Molecule *molecule, IntraNonBondedContainer *intraNonBondedType)
            : _molecule(molecule), _intraNonBondedContainer(intraNonBondedType){};

        void calculate(const potential::CoulombPotential *,
                       potential::NonCoulombPotential *,
                       const simulationBox::SimulationBox &,
                       physicalData::PhysicalData &) const;

        [[nodiscard]] std::pair<double, double> calculateSingleInteraction(const size_t                atomIndex1,
                                                                           const int                   atomIndex2AsInt,
                                                                           const linearAlgebra::Vec3D &box,
                                                                           physicalData::PhysicalData &,
                                                                           const potential::CoulombPotential *,
                                                                           potential::NonCoulombPotential *) const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] IntraNonBondedContainer      *getIntraNonBondedType() const { return _intraNonBondedContainer; }
        [[nodiscard]] simulationBox::Molecule      *getMolecule() const { return _molecule; }
        [[nodiscard]] std::vector<std::vector<int>> getAtomIndices() const { return _intraNonBondedContainer->getAtomIndices(); }
    };

}   // namespace intraNonBonded

#endif   // _INTRA_NON_BONDED_MAP_HPP_