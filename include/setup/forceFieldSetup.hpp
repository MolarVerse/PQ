#ifndef _FORCE_FIELD_SETUP_HPP_

#define _FORCE_FIELD_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupForceField(engine::Engine &);

    /**
     * @class ForceFieldSetup
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