#ifndef _RESET_KINETICS_SETUP_HPP_

#define _RESET_KINETICS_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupResetKinetics(engine::Engine &);

    /**
     * @class ResetKineticsSetup
     *
     * @brief Setup reset kinetics
     *
     */
    class ResetKineticsSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit ResetKineticsSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _RESET_KINETICS_SETUP_HPP_