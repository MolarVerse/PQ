/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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
#include "constraints.hpp"          // for Constraints
#include "engine.hpp"               // for Engine

using namespace setup;

/**
 * @brief constructs a new Constraints Setup:: Constraints Setup object and calls setup
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
}

/**
 * @brief sets constraints tolerances
 *
 */
void ConstraintsSetup::setupTolerances()
{
    _engine.getConstraints().setShakeTolerance(settings::ConstraintSettings::getShakeTolerance());
    _engine.getConstraints().setRattleTolerance(settings::ConstraintSettings::getRattleTolerance());
}

/**
 * @brief sets constraints max iterations
 *
 */
void ConstraintsSetup::setupMaxIterations()
{
    _engine.getConstraints().setShakeMaxIter(settings::ConstraintSettings::getShakeMaxIter());
    _engine.getConstraints().setRattleMaxIter(settings::ConstraintSettings::getRattleMaxIter());
}

/**
 * @brief sets constraints reference bond lengths
 *
 */
void ConstraintsSetup::setupRefBondLengths() { _engine.getConstraints().calculateConstraintBondRefs(_engine.getSimulationBox()); }