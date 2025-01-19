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

#include "simulationBox_SoA.hpp"

#include <cassert>   // for assert

using namespace simulationBox;

/**
 * @brief resize host vectors
 *
 * @param nAtoms
 * @param nMolecules
 */
void SimulationBoxSoA::resizeHostVectors(
    const size_t nAtoms,
    const size_t nMolecules
)
{
    assert(nMolecules > 0);
    assert(nAtoms >= nMolecules);

    _charges.resize(nAtoms);
    _masses.resize(nAtoms);
    _molMasses.resize(nMolecules);
    _atomsPerMolecule.resize(nMolecules);
    _moleculeIndices.resize(nAtoms);
    _atomTypes.resize(nAtoms);
    _molTypes.resize(nMolecules);
    _internalGlobalVDWTypes.resize(nAtoms);
    _moleculeOffsets.resize(nMolecules);
}

/**
 * @brief get charges
 *
 * @return charges
 */
std::vector<Real> SimulationBoxSoA::getCharges() const { return _charges; }

/**
 * @brief get masses
 *
 * @return masses
 */
std::vector<Real> SimulationBoxSoA::getMasses() const { return _masses; }

/**
 * @brief get mol masses
 *
 * @return mol masses
 */
std::vector<Real> SimulationBoxSoA::getMolMasses() const { return _molMasses; }

/**
 * @brief get atoms per molecule
 *
 * @return atoms per molecule
 */
std::vector<size_t> SimulationBoxSoA::getAtomsPerMolecule() const
{
    return _atomsPerMolecule;
}

/**
 * @brief get atom types
 *
 * @return atom types
 */
std::vector<size_t> SimulationBoxSoA::getAtomTypes() const
{
    return _atomTypes;
}

/**
 * @brief get internal global VDW types
 *
 * @return internal global VDW types
 */
std::vector<size_t> SimulationBoxSoA::getInternalGlobalVDWTypes() const
{
    return _internalGlobalVDWTypes;
}

/**
 * @brief get molecule indices
 *
 * @return molecule indices
 */
std::vector<size_t> SimulationBoxSoA::getMoleculeIndices() const
{
    return _moleculeIndices;
}

/**
 * @brief get molecule types
 *
 * @return molecule types
 */
std::vector<size_t> SimulationBoxSoA::getMolTypes() const { return _molTypes; }

/**
 * @brief get molecule offsets
 *
 * @return molecule offsets
 */
std::vector<size_t> SimulationBoxSoA::getMoleculeOffsets() const
{
    return _moleculeOffsets;
}

/**
 * @brief get charges pointer
 *
 * @return charges pointer
 */
Real* SimulationBoxSoA::getChargesPtr() { return _charges.data(); }

/**
 * @brief get masses pointer
 *
 * @return masses pointer
 */
Real* SimulationBoxSoA::getMassesPtr() { return _masses.data(); }

/**
 * @brief get mol masses pointer
 *
 * @return mol masses pointer
 */
Real* SimulationBoxSoA::getMolMassesPtr() { return _molMasses.data(); }

/**
 * @brief get atoms per molecule pointer
 *
 * @return atoms per molecule pointer
 */
size_t* SimulationBoxSoA::getAtomsPerMoleculePtr()
{
    return _atomsPerMolecule.data();
}

/**
 * @brief get atom types pointer
 *
 * @return atom types pointer
 */
size_t* SimulationBoxSoA::getAtomTypesPtr() { return _atomTypes.data(); }

/**
 * @brief get internal global VDW types pointer
 *
 * @return internal global VDW types pointer
 */
size_t* SimulationBoxSoA::getInternalGlobalVDWTypesPtr()
{
    return _internalGlobalVDWTypes.data();
}

/**
 * @brief get molecule indices pointer
 *
 * @return molecule indices pointer
 */
size_t* SimulationBoxSoA::getMoleculeIndicesPtr()
{
    return _moleculeIndices.data();
}

/**
 * @brief get molecule types pointer
 *
 * @return molecule types pointer
 */
size_t* SimulationBoxSoA::getMolTypesPtr() { return _molTypes.data(); }

/**
 * @brief get molecule offsets pointer
 *
 * @return molecule offsets pointer
 */
size_t* SimulationBoxSoA::getMoleculeOffsetsPtr()
{
    return _moleculeOffsets.data();
}