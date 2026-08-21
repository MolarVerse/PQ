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

#include <filesystem>   // for remove
#include <memory>

#include "fileSettings.hpp"   // for FileSettings
#include "logOutput.hpp"      // for LogOutput
#include "potentialBruteForce.hpp"
#include "stdoutOutput.hpp"      // for StdoutOutput
#include "timingsSettings.hpp"   // for TimingsSettings

using namespace engine;
using namespace molsys;
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
 * @brief Construct an Engine object with default simulation components.
 *
 * @details Initializes virial, potential, physical data, simulation box,
 * cell list, intra-non-bonded handler, force field, and constraints.
 */
Engine::Engine()
    : _potential{std::make_shared<potential::PotentialBruteForce>()},
      _physicalData{std::make_shared<physicalData::PhysicalData>()},
      _simulationBox{std::make_shared<molsys::SimulationBox>()},
      _cellList{std::make_shared<molsys::CellList>()},
      _intraNonBonded{std::make_shared<intraNonBonded::IntraNonBonded>()},
      _forceField{std::make_shared<forceField::ForceField>()},
      _constraints{std::make_shared<constraints::Constraints>()}
{
}

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
 * @brief Delete temporary files
 *
 * @details This function removes temporary files created during calculations.
 * The files are safely deleted using std::filesystem::remove, which does not
 * throw exceptions if the files do not exist.
 */
void Engine::deleteTmpFiles()
{
    using std::filesystem::remove;

    const auto qm_forces     = FileSettings::getQMForcesTempFileName();
    const auto qm_charges    = FileSettings::getQMChargesTempFileName();
    const auto stress_tensor = FileSettings::getStressTensorTempFileName();
    const auto pointcharges  = FileSettings::getPointChargeFileName();

    remove(qm_forces);
    remove(qm_charges);
    remove(stress_tensor);
    remove(pointcharges);
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
 * @brief get the reference to the force field
 *
 * @return const std::shared_ptr<forceField::ForceField>&
 */
const std::shared_ptr<forceField::ForceField> &Engine::getForceField() const
{
    return _forceField;
}

/**
 * @brief get the reference to the intra non bonded interactions
 *
 * @return const std::shared_ptr<IntraNonBonded>&
 */
const std::shared_ptr<IntraNonBonded> &Engine::getIntraNonBonded() const
{
    return _intraNonBonded;
}

/**
 * @brief get the reference to the potential
 *
 * @return const Potential&
 */
const std::shared_ptr<potential::Potential> &Engine::getPotential() const
{
    return _potential;
}

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
 * @brief set the inter-water interactions handler
 *
 * @param interWater The new inter-water handler to use
 */
void Engine::setInterWater(std::unique_ptr<waterModel::InterWater> interWater)
{
    _interWater = std::move(interWater);
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
 * @return const std::shared_ptr<CellList>&
 */
const std::shared_ptr<CellList> &Engine::getCellList() const
{
    return _cellList;
}

/**
 * @brief get the shared pointer to the constraints
 *
 * @return const std::shared_ptr<Constraints>&
 */
const std::shared_ptr<Constraints> &Engine::getConstraints() const
{
    return _constraints;
}
