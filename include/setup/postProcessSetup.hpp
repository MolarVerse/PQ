#ifndef _POST_PROCESS_SETUP_H_

#define _POST_PROCESS_SETUP_H_

#include "engine.hpp"

/**
 * @class PostProcessSetup
 *
 * @brief Setup post processing before reading guffdat file
 *
 */
class PostProcessSetup
{
private:
    Engine &_engine;

public:
    explicit PostProcessSetup(Engine &engine) : _engine(engine){};

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
    void setPotential();
};

/**
 * @brief Setup post processing
 *
 * @param engine
 */
void postProcessSetup(Engine &);

#endif