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

#include "qmmdEngine.hpp"

#include "integrator.hpp"        // for Integrator
#include "manostat.hpp"          // for Manostat
#include "physicalData.hpp"      // for PhysicalData
#include "resetKinetics.hpp"     // for ResetKinetics
#include "thermostat.hpp"        // for Thermostat
#include "timingsSettings.hpp"   // for TimingsSettings

using engine::QMMDEngine;

/**
 * @brief Takes one step in a QM MD simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply thermostat half step
 * - Run QM calculations
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void QMMDEngine::takeStep()
{
    _thermostat->applyThermostatHalfStep(*_simulationBox, *_physicalData);

    _integrator->firstStep(*_simulationBox);

    _constraints->applyShake(*_simulationBox);

    _qmRunner->run(*_simulationBox, *_physicalData);

    _constraints->applyDistanceConstraints(
        *_simulationBox,
        *_physicalData,
        calculateTotalSimulationTime()
    );

    _constraints->calculateConstraintBondRefs(*_simulationBox);

    _thermostat->applyThermostatOnForces(*_simulationBox);

    _integrator->secondStep(*_simulationBox);

    _constraints->applyRattle(*_simulationBox);

    _thermostat->applyThermostat(*_simulationBox, *_physicalData);

    _physicalData->calculateKinetics(*_simulationBox);

    _manostat->applyManostat(*_simulationBox, *_physicalData);

    _resetKinetics.reset(_step, *_physicalData, *_simulationBox);

    _thermostat->applyTemperatureRamping();

    _physicalData->setNumberOfQMAtoms(_simulationBox->getNumberOfQMAtoms());
}