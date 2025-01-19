/*****************************************************************************
<GPL_HEADER>

    PQ
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

#ifndef __SIMULATION_BOX_STRUCT_OF_ARRAYS_HPP__
#define __SIMULATION_BOX_STRUCT_OF_ARRAYS_HPP__

#include <cstddef>
#include <vector>

#include "typeAliases.hpp"

namespace simulationBox
{
    class SimulationBoxSoA
    {
       protected:
        std::vector<Real> _charges;
        std::vector<Real> _masses;
        std::vector<Real> _molMasses;

        std::vector<size_t> _atomsPerMolecule;
        std::vector<size_t> _moleculeIndices;
        std::vector<size_t> _atomTypes;
        std::vector<size_t> _molTypes;
        std::vector<size_t> _internalGlobalVDWTypes;
        std::vector<size_t> _moleculeOffsets;

       public:
        virtual ~SimulationBoxSoA() = default;

        virtual void resizeHostVectors(cul nAtoms, cul nMolecules);

        [[nodiscard]] virtual Real* getChargesPtr();
        [[nodiscard]] virtual Real* getMassesPtr();
        [[nodiscard]] virtual Real* getMolMassesPtr();

        [[nodiscard]] virtual size_t* getAtomsPerMoleculePtr();
        [[nodiscard]] virtual size_t* getMoleculeIndicesPtr();
        [[nodiscard]] virtual size_t* getAtomTypesPtr();
        [[nodiscard]] virtual size_t* getMolTypesPtr();
        [[nodiscard]] virtual size_t* getInternalGlobalVDWTypesPtr();
        [[nodiscard]] virtual size_t* getMoleculeOffsetsPtr();

        [[nodiscard]] std::vector<Real> getCharges() const;
        [[nodiscard]] std::vector<Real> getMasses() const;
        [[nodiscard]] std::vector<Real> getMolMasses() const;

        [[nodiscard]] std::vector<size_t> getAtomsPerMolecule() const;
        [[nodiscard]] std::vector<size_t> getMoleculeIndices() const;
        [[nodiscard]] std::vector<size_t> getAtomTypes() const;
        [[nodiscard]] std::vector<size_t> getMolTypes() const;
        [[nodiscard]] std::vector<size_t> getInternalGlobalVDWTypes() const;
        [[nodiscard]] std::vector<size_t> getMoleculeOffsets() const;
    };
}   // namespace simulationBox

#endif   // __SIMULATION_BOX_STRUCT_OF_ARRAYS_HPP__