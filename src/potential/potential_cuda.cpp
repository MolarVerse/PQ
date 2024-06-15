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
#include <cstddef>   // for size_t

#include "box.hpp"                // for Box
#include "coulombPotential.hpp"   // for CoulombPotential
#include "cuda_runtime.h"
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "potential.hpp"
#include "simulationBox.hpp"   // for SimulationBox

namespace simulationBox
{
    class CellList;
}   // namespace simulationBox

using namespace potential;

PotentialCuda::~PotentialCuda() = default;

/**
 * @brief calculates forces, coulombic and non-coulombic energy for CUDA
 * routine.
 *
 * @param simBox
 * @param physicalData
 */
inline void PotentialCuda::
    calculateForces(simulationBox::SimulationBox &simBox, physicalData::PhysicalData &physicalData, simulationBox::CellList &)
{
    startTimingsSection("InterNonBonded - Transfer");
    double totalCoulombEnergy    = 0.0;
    double totalNonCoulombEnergy = 0.0;

    simBox.flattenAtomTypes();
    simBox.flattenForces();
    simBox.flattenInternalGlobalVDWTypes();
    simBox.flattenMolTypes();
    simBox.flattenPartialCharges();
    simBox.flattenPositions();
    simBox.flattenVelocities();
        
    const size_t numberOfMolecules = simBox.getNumberOfMolecules();

    stopTimingsSection("InterNonBonded - Transfer");
    startTimingsSection("InterNonBonded");
    stopTimingsSection("InterNonBonded");
    startTimingsSection("InterNonBonded - Transfer");
    stopTimingsSection("InterNonBonded - Transfer");
    return;
}
