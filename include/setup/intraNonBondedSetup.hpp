#ifndef _INTRA_NON_BONDED_SETUP_HPP_

#define _INTRA_NON_BONDED_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupIntraNonBonded(engine::Engine &);

    /**
     * @class IntraNonBondedSetup
     *
     * @brief Setup intra non bonded interactions
     *
     */
    class IntraNonBondedSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit IntraNonBondedSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _INTRA_NON_BONDED_SETUP_HPP_