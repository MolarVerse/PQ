#ifndef _FORCE_FIELD_SETUP_HPP_

#define _FORCE_FIELD_SETUP_HPP_

namespace engine
{
    class Engine;   // forward declaration
}

namespace setup
{
    void setupForceField(engine::Engine &);

    /**
     * @class ForceFieldSetup
     *
     * @brief setup all bonded contributions in the force field
     *
     */
    class ForceFieldSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit ForceFieldSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
        void setupBonds();
        void setupAngles();
        void setupDihedrals();
        void setupImproperDihedrals();
    };

}   // namespace setup

#endif   // _FORCE_FIELD_SETUP_HPP_