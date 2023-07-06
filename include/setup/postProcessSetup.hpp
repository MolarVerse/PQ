#ifndef _POST_PROCESS_SETUP_H_

#define _POST_PROCESS_SETUP_H_

#include "engine.hpp"

namespace setup
{
    class PostProcessSetup;
    void postProcessSetup(engine::Engine &);
}   // namespace setup

/**
 * @class PostProcessSetup
 *
 * @brief Setup post processing before reading guffdat file
 *
 */
class setup::PostProcessSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit PostProcessSetup(engine::Engine &engine) : _engine(engine){};

    void setup();

    void setAtomMasses();
    void setAtomicNumbers();

    void calculateMolMass();
    void calculateTotalMass();
    void calculateTotalCharge();

    void resizeAtomShiftForces();

    void checkBoxSettings();
    void checkRcCutoff();

    void setupCellList();
    void setupPotential();
    void setupTimestep();
};

#endif