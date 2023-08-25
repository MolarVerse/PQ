#ifndef _CELL_LIST_SETUP_HPP_

#define _CELL_LIST_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupCellList(engine::Engine &);

    /**
     * @class SetupCellList
     *
     */
    class CellListSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit CellListSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _CELL_LIST_SETUP_HPP_