#ifndef _CELL_LIST_SETUP_HPP_

#define _CELL_LIST_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    class CellListSetup;
    void setupCellList(engine::Engine &);
}   // namespace setup

/**
 * @class SetupCellList
 *
 */
class setup::CellListSetup
{
  private:
    engine::Engine &_engine;

  public:
    explicit CellListSetup(engine::Engine &engine) : _engine(engine){};

    void setup();
};

#endif