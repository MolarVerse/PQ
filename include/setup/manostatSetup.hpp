#ifndef _MANOSTAT_SETUP_HPP_

#define _MANOSTAT_SETUP_HPP_

namespace engine
{
    class Engine;
}   // namespace engine

namespace setup
{
    void setupManostat(engine::Engine &);

    /**
     * @class ManostatSetup
     *
     * @brief Setup manostat
     *
     */
    class ManostatSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit ManostatSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _MANOSTAT_SETUP_HPP_