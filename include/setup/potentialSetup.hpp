#ifndef _POTENTIAL_SETUP_HPP_

#define _POTENTIAL_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupPotential(engine::Engine &);

    /**
     * @class PotentialSetup
     *
     * @brief Setup potential
     *
     */
    class PotentialSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit PotentialSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
        void setupCoulomb();
        void setupNonCoulomb();
        void setupNonCoulombicPairs();
    };

}   // namespace setup

#endif   // _POTENTIAL_SETUP_HPP_