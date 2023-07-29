#include "constraintsSetup.hpp"

using namespace std;
using namespace setup;
using namespace constraints;

/**
 * @brief wrapper for setup constraints
 *
 */
void setup::setupConstraints(engine::Engine &engine)
{
    ConstraintsSetup constraintsSetup(engine);
    constraintsSetup.setup();
}

/**
 * @brief sets constraints data in constraints object
 *
 */
void ConstraintsSetup::setup()
{
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
    _engine.getConstraints().setShakeTolerance(_engine.getSettings().getShakeTolerance());
    _engine.getConstraints().setRattleTolerance(_engine.getSettings().getRattleTolerance());
}

/**
 * @brief sets constraints max iterations
 *
 */
void ConstraintsSetup::setupMaxIterations()
{
    _engine.getConstraints().setShakeMaxIter(_engine.getSettings().getShakeMaxIter());
    _engine.getConstraints().setRattleMaxIter(_engine.getSettings().getRattleMaxIter());
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