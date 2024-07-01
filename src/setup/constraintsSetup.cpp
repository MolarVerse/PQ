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

#include "constraintsSetup.hpp"

#include "constraintSettings.hpp"   // for getShakeMaxIter, getShakeTolerance, getRattleMaxIter, getRattleTolerance
#include "constraints.hpp"    // for Constraints
#include "engine.hpp"         // for Engine
#include "mShakeReader.hpp"   // for readMShake

using namespace setup;
using namespace engine;
using namespace settings;

using input::mShake::readMShake;

/**
 * @brief constructs a new Constraints Setup:: Constraints Setup object and
 * calls setup
 *
 * @param engine
 */
void setup::setupConstraints(Engine &engine)
{
    if (!engine.isConstraintsActivated())
        return;

    engine.getStdoutOutput().writeSetup("Constraints");
    engine.getLogOutput().writeSetup("Constraints");

    ConstraintsSetup constraintsSetup(engine);
    constraintsSetup.setup();
}

/**
 * @brief constructor
 *
 * @param engine
 */
ConstraintsSetup::ConstraintsSetup(Engine &engine) : _engine(engine){};

/**
 * @brief sets constraints data in constraints object
 *
 * @details sets tolerances, max iterations, reference bond lengths and timestep
 */
void ConstraintsSetup::setup()
{
    setupTolerances();
    setupMaxIterations();
    setupRefBondLengths();
    setupMShake();
    setupDegreesOfFreedom();

    writeSetupInfo();
}

/**
 * @brief setup M-SHAKE
 *
 */
void ConstraintsSetup::setupMShake()
{
    auto &constraints = _engine.getConstraints();

    if (!constraints.isMShakeActive())
        return;

    readMShake(_engine);

    constraints.initMShake();
}

/**
 * @brief sets constraints tolerances
 *
 */
void ConstraintsSetup::setupTolerances()
{
    _shakeTolerance  = ConstraintSettings::getShakeTolerance();
    _rattleTolerance = ConstraintSettings::getRattleTolerance();

    auto &constraints = _engine.getConstraints();

    constraints.setShakeTolerance(_shakeTolerance);
    constraints.setRattleTolerance(_rattleTolerance);
}

/**
 * @brief sets constraints max iterations
 *
 */
void ConstraintsSetup::setupMaxIterations()
{
    _shakeMaxIter    = ConstraintSettings::getShakeMaxIter();
    _rattleTolerance = ConstraintSettings::getRattleMaxIter();

    auto &constraints = _engine.getConstraints();

    constraints.setShakeMaxIter(_shakeMaxIter);
    constraints.setRattleMaxIter(_rattleTolerance);
}

/**
 * @brief sets constraints reference bond lengths
 *
 */
void ConstraintsSetup::setupRefBondLengths()
{
    auto       &constraints = _engine.getConstraints();
    const auto &simBox      = _engine.getSimulationBox();

    constraints.calculateConstraintBondRefs(simBox);
}

/**
 * @brief corrects the number of degrees of freedom for the constraints
 *
 */
void ConstraintsSetup::setupDegreesOfFreedom()
{
    auto       &simBox      = _engine.getSimulationBox();
    const auto &constraints = _engine.getConstraints();

    _shakeConstraints  = constraints.getNumberOfBondConstraints();
    _mShakeConstraints = constraints.getNumberOfMShakeConstraints(simBox);

    auto dof  = simBox.getDegreesOfFreedom();
    dof      -= _shakeConstraints;
    dof      -= _mShakeConstraints;

    simBox.setDegreesOfFreedom(dof);
}

/**
 * @brief write setup information to log output
 *
 */
void ConstraintsSetup::writeSetupInfo()
{
    const auto &constraints = _engine.getConstraints();

    writeEnabledMessage();

    if (constraints.isShakeLikeActive())
        writeToleranceMessage();

    if (constraints.isShakeActive())
        writeMaxIterMessage();

    writeDofMessage();
}

/**
 * @brief write enabled message to log output
 *
 */
void ConstraintsSetup::writeEnabledMessage()
{
    auto &constraints = _engine.getConstraints();

    // clang-format off
    std::string shakeMsg  = constraints.isShakeActive() ? "enabled" : "disabled";
    std::string mShakeMsg = constraints.isMShakeActive() ? "enabled" : "disabled";
    // clang-format on

    shakeMsg  = std::format("SHAKE:   {}", shakeMsg);
    mShakeMsg = std::format("M-SHAKE: {}", mShakeMsg);

    auto &logOutput = _engine.getLogOutput();

    logOutput.writeSetup(shakeMsg);
    logOutput.writeSetup(mShakeMsg);
    logOutput.writeEmptyLine();
}

/**
 * @brief write degrees of freedom message to log output
 *
 */
void ConstraintsSetup::writeDofMessage()
{
    auto &simBox = _engine.getSimulationBox();

    const auto totalDof = simBox.getDegreesOfFreedom();

    // clang-format off
    const auto shakeDofMsg  = std::format("SHAKE DOF:   {}", _shakeConstraints);
    const auto mShakeDofMsg = std::format("M-SHAKE DOF: {}", _mShakeConstraints);
    const auto totalDofMsg  = std::format("Total DOF:   {}", totalDof);
    // clang-format on

    auto &logOutput = _engine.getLogOutput();

    logOutput.writeSetup(shakeDofMsg);
    logOutput.writeSetup(mShakeDofMsg);
    logOutput.writeSetup(totalDofMsg);
    logOutput.writeEmptyLine();
}

/**
 * @brief write tolerances message to log output
 *
 */
void ConstraintsSetup::writeToleranceMessage()
{
    // clang-format off
    const auto shakeTolMsg  = std::format("SHAKE Tolerance:  {}", _shakeTolerance);
    const auto rattleTolMsg = std::format("RATTLE Tolerance: {}", _rattleTolerance);
    // clang-format on

    auto &logOutput = _engine.getLogOutput();

    logOutput.writeSetup(shakeTolMsg);
    logOutput.writeSetup(rattleTolMsg);
    logOutput.writeEmptyLine();
}

/**
 * @brief write max iterations message to log output
 *
 */
void ConstraintsSetup::writeMaxIterMessage()
{
    // clang-format off
    const auto shakeMaxIterMsg  = std::format("SHAKE Max Iter:  {}", _shakeMaxIter);
    const auto rattleMaxIterMsg = std::format("RATTLE Max Iter: {}", _rattleMaxIter);
    // clang-format on

    auto &logOutput = _engine.getLogOutput();

    logOutput.writeSetup(shakeMaxIterMsg);
    logOutput.writeSetup(rattleMaxIterMsg);
    logOutput.writeEmptyLine();
}
