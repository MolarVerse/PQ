#ifndef _POST_PROCESS_SETUP_H_

#define _POST_PROCESS_SETUP_H_

#include "engine.hpp"

class PostProcessSetup
{
private:
    Engine _engine;

public:
    explicit PostProcessSetup(const Engine &engine) : _engine(engine){};

    void setup();
    void setAtomMasses();
};

/**
 * @brief Setup post processing
 *
 * @param engine
 */
void postProcessSetup(Engine &);

#endif