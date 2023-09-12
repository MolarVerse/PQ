#ifndef _QM_SETUP_HPP_

#define _QM_SETUP_HPP_

namespace engine
{
    class QMMDEngine;   // forward declaration
}

namespace setup
{
    void setupQM(engine::QMMDEngine &);

    /**
     * @class QMSetup
     *
     * @brief Setup QM
     *
     */
    class QMSetup
    {
      private:
        engine::QMMDEngine &_engine;

      public:
        explicit QMSetup(engine::QMMDEngine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _QM_SETUP_HPP_