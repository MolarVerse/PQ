#ifndef _OUTPUT_FILES_SETUP_HPP_

#define _OUTPUT_FILES_SETUP_HPP_

#include "engine.hpp"

namespace setup
{
    void setupOutputFiles(engine::Engine &engine);

    /**
     * @class OutputFilesSetup
     *
     * @brief Class to setup the output file names
     *
     */
    class OutputFilesSetup
    {
      private:
        engine::Engine &_engine;

      public:
        explicit OutputFilesSetup(engine::Engine &engine) : _engine(engine){};

        void setup();
    };

}   // namespace setup

#endif   // _OUTPUT_FILES_SETUP_HPP_