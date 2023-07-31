#ifndef _SIMULATION_BOX_SETUP_HPP_

#define _SIMULATION_BOX_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class SimulationBoxSetup;
    void setupSimulationBox(engine::Engine &);
}   // namespace setup

/**
 * @class SetupSimulationBox
 *
 * @brief Setup simulation box
 *
 */
class setup::SimulationBoxSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit SimulationBoxSetup(engine::Engine &engine) : _engine(engine){};

    void setup();

    void setAtomMasses();
    void setAtomicNumbers();

    void calculateMolMass();
    void calculateTotalMass();
    void calculateTotalCharge();

    void resizeAtomShiftForces();

    void checkBoxSettings();
    void checkRcCutoff();
};

#endif   // _SIMULATION_BOX_SETUP_HPP_