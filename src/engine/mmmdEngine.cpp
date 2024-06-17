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

#include "mmmdEngine.hpp"

#include <memory>   // for unique_ptr

#include "celllist.hpp"          // for CellList
#include "constraints.hpp"       // for Constraints
#include "engineOutput.hpp"      // for engine
#include "forceFieldClass.hpp"   // for ForceField
#include "integrator.hpp"        // for Integrator
#include "intraNonBonded.hpp"    // for IntraNonBonded
#include "manostat.hpp"          // for Manostat
#include "physicalData.hpp"      // for PhysicalData
#include "potential.hpp"         // for Potential
#include "resetKinetics.hpp"     // for ResetKinetics
#include "thermostat.hpp"        // for Thermostat
#include "virial.hpp"            // for Virial

#ifdef WITH_CUDA
#include "potential_cuda.cuh"   // for CudaPotential
#endif
using namespace engine;

/**
 * @brief Takes one step in the simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply SHAKE
 * - Update cell list
 * - Calculate forces
 * - Calculate intra non bonded forces
 * - Calculate virial
 * - Calculate constraint bond references
 * - calculate intra molecular virial correction
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply RATTLE
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void MMMDEngine::takeStep()
{
    _thermostat->applyThermostatHalfStep(_simulationBox, _physicalData);

    _integrator->firstStep(_simulationBox);

    _constraints.applyShake(_simulationBox);

    _cellList.updateCellList(_simulationBox);

#ifdef WITH_CUDA
    _cudaPotential.calculateForces(
        _simulationBox,
        _physicalData,
        _cudaSimulationBox,
        _cudaLennardJones,
        _cudaCoulombWolf
    );
#else
    _potential->calculateForces(_simulationBox, _physicalData, _cellList);
#endif
    _intraNonBonded.calculate(_simulationBox, _physicalData);

    _virial->calculateVirial(_simulationBox, _physicalData);

    _forceField.calculateBondedInteractions(_simulationBox, _physicalData);

    _constraints.applyDistanceConstraints(
        _simulationBox,
        _physicalData,
        calculateTotalSimulationTime()
    );

    _constraints.calculateConstraintBondRefs(_simulationBox);

    _virial->intraMolecularVirialCorrection(_simulationBox, _physicalData);

    _thermostat->applyThermostatOnForces(_simulationBox);

    _integrator->secondStep(_simulationBox);

    _constraints.applyRattle(_simulationBox);

    _thermostat->applyThermostat(_simulationBox, _physicalData);

    _physicalData.calculateKinetics(_simulationBox);

    _manostat->applyManostat(_simulationBox, _physicalData);

    _resetKinetics.reset(_step, _physicalData, _simulationBox);

    _thermostat->applyTemperatureRamping();
}
