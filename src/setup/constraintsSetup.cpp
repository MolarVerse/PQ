#include "constraintsSetup.hpp"

#include "constraintSettings.hpp"   // for getShakeMaxIter, getShakeTolerance, getRattleMaxIter, getRattleTolerance
#include "constraints.hpp"          // for Constraints
#include "engine.hpp"               // for Engine
#include "settings.hpp"             // for Settings
#include "timings.hpp"              // for Timings

using namespace setup;

/**
 * @brief constructs a new Constraints Setup:: Constraints Setup object and calls setup
 *
 * @param engine
 */
void setup::setupConstraints(engine::Engine &engine)
{
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
    if (!_engine.isConstraintsActivated())
        return;

    setupTolerances();
    setupMaxIterations();
    setupRefBondLengths();
    setupTimestep();
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

/**
 * @brief sets timestep in constraints
 *
 */
void ConstraintsSetup::setupTimestep() { _engine.getConstraints().setDt(_engine.getTimings().getTimestep()); }