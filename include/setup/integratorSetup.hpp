#ifndef _INTEGRATOR_SETUP_HPP_

#define _INTEGRATOR_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupIntegrator(engine::Engine &);

    /**
     * @class IntegratorSetup
     *
     * @brief Setup Integrator
     *
     */
    class IntegratorSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit IntegratorSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _INTEGRATOR_SETUP_HPP_