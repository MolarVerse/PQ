#ifndef _CONSTRAINTS_SETUP_HPP_

#define _CONSTRAINTS_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class ConstraintsSetup;
    void setupConstraints(engine::Engine &);
}   // namespace setup

/**
 * @class ConstraintsSetup
 *
 * @brief Setup constraints before reading guffdat file
 *
 */
class setup::ConstraintsSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit ConstraintsSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif   // _CONSTRAINTS_SETUP_HPP_