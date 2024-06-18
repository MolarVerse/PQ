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

#include "engine.hpp"

#include "constants/conversionFactors.hpp"   // for _FS_TO_PS_
#include "logOutput.hpp"                     // for LogOutput
#include "outputFileSettings.hpp"            // for OutputFileSettings
#include "progressbar.hpp"                   // for progressbar
#include "referencesOutput.hpp"              // for ReferencesOutput
#include "settings.hpp"                      // for Settings
#include "stdoutOutput.hpp"                  // for StdoutOutput
#include "timingsSettings.hpp"               // for TimingsSettings
#include "vector3d.hpp"                      // for norm

using namespace engine;

/**
 * @brief Adds a timings section to the timingsSection vector.
 *
 */
void Engine::addTimer(const timings::Timer &timings)
{
    _timer.addTimer(timings);
}

/**
 * @brief Calculate total simulation time.
 *
 * @return double
 */
double Engine::calculateTotalSimulationTime() const
{
    const auto step0   = settings::TimingsSettings::getStepCount();
    const auto dt      = settings::TimingsSettings::getTimeStep();
    const auto effStep = _step + step0;

    return static_cast<double>(effStep) * dt;
}

/**
 * @brief checks if the force field is activated
 *
 * @return true
 * @return false
 */
bool Engine::isForceFieldNonCoulombicsActivated() const
{
    return _forceField.isNonCoulombicActivated();
}

/**
 * @brief checks if the guff formalism is activated
 *
 * @return true
 * @return false
 */
bool Engine::isGuffActivated() const
{
    return !_forceField.isNonCoulombicActivated();
}

/**
 * @brief checks if the cell list is activated
 *
 * @return true
 * @return false
 */
bool Engine::isCellListActivated() const { return _cellList.isActive(); }

/**
 * @brief checks if any constraints are activated
 *
 * @return true
 * @return false
 */
bool Engine::isConstraintsActivated() const { return _constraints.isActive(); }

/**
 * @brief checks if the intra non bonded interactions are activated
 *
 * @return true
 * @return false
 */
bool Engine::isIntraNonBondedActivated() const
{
    return _intraNonBonded.isActive();
}

/**
 * @brief get the reference to the cell list
 *
 * @return simulationBox::CellList&
 */
simulationBox::CellList &Engine::getCellList() { return _cellList; }

/**
 * @brief get the reference to the simulation box
 *
 * @return simulationBox::SimulationBox&
 */
simulationBox::SimulationBox &Engine::getSimulationBox()
{
    return *_simulationBox;
}

/**
 * @brief get the reference to the physical data
 *
 * @return physicalData::PhysicalData&
 */
physicalData::PhysicalData &Engine::getPhysicalData() { return *_physicalData; }

/**
 * @brief get the reference to the average physical data
 *
 * @return physicalData::PhysicalData&
 */
physicalData::PhysicalData &Engine::getAveragePhysicalData()
{
    return _averagePhysicalData;
}

/**
 * @brief get the reference to the Constraints
 *
 * @return timings::Timer&
 */
constraints::Constraints &Engine::getConstraints() { return _constraints; }

/**
 * @brief get the reference to the force field
 *
 * @return forceField::ForceField&
 */
forceField::ForceField &Engine::getForceField() { return _forceField; }

/**
 * @brief get the reference to the intra non bonded interactions
 *
 * @return intraNonBonded::IntraNonBonded&
 */
intraNonBonded::IntraNonBonded &Engine::getIntraNonBonded()
{
    return _intraNonBonded;
}

/**
 * @brief get the reference to the virial
 *
 * @return virial::Virial&
 */
virial::Virial &Engine::getVirial() { return *_virial; }

/**
 * @brief get the reference to the potential
 *
 * @return potential::Potential&
 */
potential::Potential &Engine::getPotential() { return *_potential; }

/**
 * @brief get the pointer to the force field
 *
 * @return forceField::ForceField*
 */
forceField::ForceField *Engine::getForceFieldPtr() { return &_forceField; }

/**
 * @brief get the pointer to the potential
 *
 * @return potential::Potential*
 */
potential::Potential *Engine::getPotentialPtr() { return _potential.get(); }

/**
 * @brief get the pointer to the virial
 *
 * @return virial::Virial*
 */
virial::Virial *Engine::getVirialPtr() { return _virial.get(); }

/**
 * @brief get the pointer to the cell list
 *
 * @return simulationBox::CellList*
 */
simulationBox::CellList *Engine::getCellListPtr() { return &_cellList; }

/**
 * @brief get the pointer to the simulation box
 *
 * @return simulationBox::SimulationBox*
 */
simulationBox::SimulationBox *Engine::getSimulationBoxPtr()
{
    return _simulationBox.get();
}

/**
 * @brief get the pointer to the physical data
 *
 * @return physicalData::PhysicalData*
 */
physicalData::PhysicalData *Engine::getPhysicalDataPtr()
{
    return _physicalData.get();
}

/**
 * @brief get the pointer to the constraints
 *
 * @return constraints::Constraints*
 */
constraints::Constraints *Engine::getConstraintsPtr() { return &_constraints; }

/**
 * @brief get the pointer to the intra non bonded interactions
 *
 * @return intraNonBonded::IntraNonBonded*
 */
intraNonBonded::IntraNonBonded *Engine::getIntraNonBondedPtr()
{
    return &_intraNonBonded;
}

/**
 * @brief get the reference to the engine output
 *
 * @return EngineOutput&
 */
EngineOutput &Engine::getEngineOutput() { return _engineOutput; }

/**
 * @brief get the reference to the log output
 *
 * @return output::LogOutput&
 */
output::LogOutput &Engine::getLogOutput()
{
    return _engineOutput.getLogOutput();
}

/**
 * @brief get the reference to the stdout output
 *
 * @return output::StdoutOutput&
 */
output::StdoutOutput &Engine::getStdoutOutput()
{
    return _engineOutput.getStdoutOutput();
}

/**
 * @brief get the TimingsOutput
 *
 * @return output::TimingsOutput&
 */
output::TimingsOutput &Engine::getTimingsOutput()
{
    return _engineOutput.getTimingsOutput();
}

/**
 * @brief get the reference to the energy output
 *
 * @return output::EnergyOutput&
 */
output::EnergyOutput &Engine::getEnergyOutput()
{
    return _engineOutput.getEnergyOutput();
}

/**
 * @brief get the reference to the xyz output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &Engine::getXyzOutput()
{
    return _engineOutput.getXyzOutput();
}

/**
 * @brief get the reference to the force output
 *
 * @return output::TrajectoryOutput&
 */
output::TrajectoryOutput &Engine::getForceOutput()
{
    return _engineOutput.getForceOutput();
}

/**
 * @brief get the reference to the rst file output
 *
 * @return output::RstFileOutput&
 */
output::RstFileOutput &Engine::getRstFileOutput()
{
    return _engineOutput.getRstFileOutput();
}

/**
 * @brief get the reference to the info output
 *
 * @return output::InfoOutput&
 */
output::InfoOutput &Engine::getInfoOutput()
{
    return _engineOutput.getInfoOutput();
}