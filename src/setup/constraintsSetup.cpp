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
 * @brief sets shake data in constraints object
 *
 */
void ConstraintsSetup::setup()
{
    _engine.getConstraints().setShakeMaxIter(_engine.getSettings().getShakeMaxIter());
    _engine.getConstraints().setShakeTolerance(_engine.getSettings().getShakeTolerance());
    _engine.getConstraints().setRattleMaxIter(_engine.getSettings().getRattleMaxIter());
    _engine.getConstraints().setRattleTolerance(_engine.getSettings().getRattleTolerance());
}