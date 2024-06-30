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
using namespace simulationBox;
using namespace physicalData;
using namespace forceField;
using namespace intraNonBonded;
using namespace virial;
using namespace potential;
using namespace constraints;
using namespace output;
using namespace timings;
using namespace settings;

/**
 * @brief Adds a timings section to the timingsSection vector.
 *
 */
void Engine::addTimer(const Timer &timings) { _timer.addTimer(timings); }

/**
 * @brief Calculate total simulation time.
 *
 * @return double
 */
double Engine::calculateTotalSimulationTime() const
{
    const auto step0   = TimingsSettings::getStepCount();
    const auto dt      = TimingsSettings::getTimeStep();
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
    return _forceField->isNonCoulombicActivated();
}

/**
 * @brief checks if the guff formalism is activated
 *
 * @return true
 * @return false
 */
bool Engine::isGuffActivated() const
{
    return !_forceField->isNonCoulombicActivated();
}

/**
 * @brief checks if the cell list is activated
 *
 * @return true
 * @return false
 */
bool Engine::isCellListActivated() const { return _cellList->isActive(); }

/**
 * @brief checks if any constraints are activated
 *
 * @return true
 * @return false
 */
bool Engine::isConstraintsActivated() const { return _constraints->isActive(); }

/**
 * @brief checks if the intra non bonded interactions are activated
 *
 * @return true
 * @return false
 */
bool Engine::isIntraNonBondedActivated() const
{
    return _intraNonBonded->isActive();
}

/**
 * @brief get the reference to the cell list
 *
 * @return CellList&
 */
CellList &Engine::getCellList() { return *_cellList; }

/**
 * @brief get the reference to the simulation box
 *
 * @return SimulationBox&
 */
SimulationBox &Engine::getSimulationBox() { return *_simulationBox; }

/**
 * @brief get the reference to the physical data
 *
 * @return PhysicalData&
 */
PhysicalData &Engine::getPhysicalData() { return *_physicalData; }

/**
 * @brief get the reference to the average physical data
 *
 * @return PhysicalData&
 */
PhysicalData &Engine::getAveragePhysicalData() { return _averagePhysicalData; }

/**
 * @brief get the reference to the Constraints
 *
 * @return timings::Timer&
 */
Constraints &Engine::getConstraints() { return *_constraints; }

/**
 * @brief get the reference to the force field
 *
 * @return ForceField&
 */
ForceField &Engine::getForceField() { return *_forceField; }

/**
 * @brief get the reference to the intra non bonded interactions
 *
 * @return IntraNonBonded&
 */
IntraNonBonded &Engine::getIntraNonBonded() { return *_intraNonBonded; }

/**
 * @brief get the reference to the virial
 *
 * @return Virial&
 */
Virial &Engine::getVirial() { return *_virial; }

/**
 * @brief get the reference to the potential
 *
 * @return Potential&
 */
Potential &Engine::getPotential() { return *_potential; }

/**
 * @brief get the pointer to the force field
 *
 * @return ForceField*
 */
ForceField *Engine::getForceFieldPtr() { return _forceField.get(); }

/**
 * @brief get the pointer to the potential
 *
 * @return Potential*
 */
Potential *Engine::getPotentialPtr() { return _potential.get(); }

/**
 * @brief get the pointer to the virial
 *
 * @return Virial*
 */
Virial *Engine::getVirialPtr() { return _virial.get(); }

/**
 * @brief get the pointer to the cell list
 *
 * @return CellList*
 */
CellList *Engine::getCellListPtr() { return _cellList.get(); }

/**
 * @brief get the pointer to the simulation box
 *
 * @return SimulationBox*
 */
SimulationBox *Engine::getSimulationBoxPtr() { return _simulationBox.get(); }

/**
 * @brief get the pointer to the physical data
 *
 * @return PhysicalData*
 */
PhysicalData *Engine::getPhysicalDataPtr() { return _physicalData.get(); }

/**
 * @brief get the pointer to the constraints
 *
 * @return Constraints*
 */
Constraints *Engine::getConstraintsPtr() { return _constraints.get(); }

/**
 * @brief get the pointer to the intra non bonded interactions
 *
 * @return IntraNonBonded*
 */
IntraNonBonded *Engine::getIntraNonBondedPtr() { return _intraNonBonded.get(); }

/**
 * @brief get the reference to the engine output
 *
 * @return EngineOutput&
 */
EngineOutput &Engine::getEngineOutput() { return _engineOutput; }

/**
 * @brief get the reference to the log output
 *
 * @return LogOutput&
 */
LogOutput &Engine::getLogOutput() { return _engineOutput.getLogOutput(); }

/**
 * @brief get the reference to the stdout output
 *
 * @return StdoutOutput&
 */
StdoutOutput &Engine::getStdoutOutput()
{
    return _engineOutput.getStdoutOutput();
}

/**
 * @brief get the TimingsOutput
 *
 * @return TimingsOutput&
 */
TimingsOutput &Engine::getTimingsOutput()
{
    return _engineOutput.getTimingsOutput();
}

/**
 * @brief get the reference to the energy output
 *
 * @return EnergyOutput&
 */
EnergyOutput &Engine::getEnergyOutput()
{
    return _engineOutput.getEnergyOutput();
}

/**
 * @brief get the reference to the xyz output
 *
 * @return TrajectoryOutput&
 */
TrajectoryOutput &Engine::getXyzOutput()
{
    return _engineOutput.getXyzOutput();
}

/**
 * @brief get the reference to the force output
 *
 * @return TrajectoryOutput&
 */
TrajectoryOutput &Engine::getForceOutput()
{
    return _engineOutput.getForceOutput();
}

/**
 * @brief get the reference to the rst file output
 *
 * @return RstFileOutput&
 */
RstFileOutput &Engine::getRstFileOutput()
{
    return _engineOutput.getRstFileOutput();
}

/**
 * @brief get the reference to the info output
 *
 * @return InfoOutput&
 */
InfoOutput &Engine::getInfoOutput() { return _engineOutput.getInfoOutput(); }

/******************************
 *                            *
 * get shared pointer methods *
 *                            *
 ******************************/

/**
 * @brief get the shared pointer to the force field
 *
 * @return std::shared_ptr<ForceField>
 */
std::shared_ptr<ForceField> Engine::getSharedForceField() const
{
    return _forceField;
}

/**
 * @brief get the shared pointer to the simulation box
 *
 * @return std::shared_ptr<SimulationBox>
 */
std::shared_ptr<SimulationBox> Engine::getSharedSimulationBox() const
{
    return _simulationBox;
}

/**
 * @brief get the shared pointer to the physical data
 *
 * @return std::shared_ptr<PhysicalData>
 */
std::shared_ptr<PhysicalData> Engine::getSharedPhysicalData() const
{
    return _physicalData;
}

/**
 * @brief get the shared pointer to the cell list
 *
 * @return std::shared_ptr<CellList>
 */
std::shared_ptr<CellList> Engine::getSharedCellList() const
{
    return _cellList;
}

/**
 * @brief get the shared pointer to the constraints
 *
 * @return std::shared_ptr<Constraints>
 */
std::shared_ptr<Constraints> Engine::getSharedConstraints() const
{
    return _constraints;
}

/**
 * @brief get the shared pointer to the intra non bonded interactions
 *
 * @return std::shared_ptr<IntraNonBonded>
 */
std::shared_ptr<IntraNonBonded> Engine::getSharedIntraNonBonded() const
{
    return _intraNonBonded;
}

/**
 * @brief get the shared pointer to the virial
 *
 * @return std::shared_ptr<Virial>
 */
std::shared_ptr<Virial> Engine::getSharedVirial() const { return _virial; }

/**
 * @brief get the shared pointer to the potential
 *
 * @return std::shared_ptr<Potential>
 */
std::shared_ptr<Potential> Engine::getSharedPotential() const
{
    return _potential;
}