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

/**
 * @brief constructs a new Constraints Setup:: Constraints Setup object and
 * calls setup
 *
 * @param engine
 */
void setup::setupConstraints(engine::Engine &engine)
{
    if (!engine.isConstraintsActivated())
        return;

    engine.getStdoutOutput().writeSetup("constraints (e.g. SHAKE, RATTLE)");
    engine.getLogOutput().writeSetup("constraints (e.g. SHAKE, RATTLE)");

    ConstraintsSetup constraintsSetup(engine);
    constraintsSetup.setup();
}

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
    setupDegreesOfFreedom();
}

/**
 * @brief setup M-SHAKE
 *
 */
void ConstraintsSetup::setupMShake()
{
    if (!_engine.getConstraints().isMShakeActive())
        return;

    input::mShake::readMShake(_engine);

    _engine.getConstraints().initMShake(_engine.getSimulationBox());

    throw customException::UserInputException("M-SHAKE is not implemented yet");
}

/**
 * @brief sets constraints tolerances
 *
 */
void ConstraintsSetup::setupTolerances()
{
    const auto shakeTol  = settings::ConstraintSettings::getShakeTolerance();
    const auto rattleTol = settings::ConstraintSettings::getRattleTolerance();

    _engine.getConstraints().setShakeTolerance(shakeTol);
    _engine.getConstraints().setRattleTolerance(rattleTol);
}

/**
 * @brief sets constraints max iterations
 *
 */
void ConstraintsSetup::setupMaxIterations()
{
    const auto shakeMaxIter  = settings::ConstraintSettings::getShakeMaxIter();
    const auto rattleMaxIter = settings::ConstraintSettings::getRattleMaxIter();

    _engine.getConstraints().setShakeMaxIter(shakeMaxIter);
    _engine.getConstraints().setRattleMaxIter(rattleMaxIter);
}

/**
 * @brief sets constraints reference bond lengths
 *
 */
void ConstraintsSetup::setupRefBondLengths()
{
    _engine.getConstraints().calculateConstraintBondRefs(
        _engine.getSimulationBox()
    );
}

/**
 * @brief corrects the number of degrees of freedom for the constraints
 *
 */
void ConstraintsSetup::setupDegreesOfFreedom()
{
    auto      &constraints      = _engine.getConstraints();
    const auto nBondConstraints = constraints.getNumberOfBondConstraints();

    auto dof  = _engine.getSimulationBox().getDegreesOfFreedom();
    dof      -= nBondConstraints;

    _engine.getSimulationBox().setDegreesOfFreedom(dof);

    _engine.getLogOutput().writeSetupInfo(
        std::format("constraint DOF:   {:8d}", nBondConstraints)
    );

    _engine.getLogOutput().writeSetupInfo(
        std::format("simulation DOF:   {:8d}", dof)
    );
}
