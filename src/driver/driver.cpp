#include "driver.hpp"

#include "engine.hpp"
#include "inputFileReader.hpp"
#include "setup.hpp"

namespace driver
{
    /**
     * @brief Run a PQ simulation from an input file.
     *
     * @param inputFileName The name of the input file containing the simulation
     * setup.
     */
    void Driver::run(const std::string &inputFileName)
    {
        auto engine = std::unique_ptr<engine::Engine>();
        input::readJobType(inputFileName, engine);

        setup::setupRequestedJob(inputFileName, *engine);

        engine->run();
    }

}   // namespace driver
