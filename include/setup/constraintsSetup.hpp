#ifndef _CONSTRAINTS_SETUP_HPP_

#define _CONSTRAINTS_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupConstraints(engine::Engine &);

    /**
     * @class ConstraintsSetup
     *
     * @brief Setup constraints before reading guffdat file
     *
     */
    class ConstraintsSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit ConstraintsSetup(engine::Engine &engine) : _engine(engine){};

        void setup();

        void setupTolerances();
        void setupMaxIterations();
        void setupRefBondLengths();
        void setupTimestep();
    };

}   // namespace setup

#endif   // _CONSTRAINTS_SETUP_HPP_