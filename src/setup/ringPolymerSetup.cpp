#include "ringPolymerSetup.hpp"

#include "fileSettings.hpp"                   // for FileSettings
#include "maxwellBoltzmann.hpp"               // for MaxwellBoltzmann
#include "ringPolymerRestartFileReader.hpp"   // for RingPolymerRestartFileReader
#include "ringPolymerSettings.hpp"            // for RingPolymerSettings

using setup::RingPolymerSetup;

/**
 * @brief wrapper to build RingPolymerSetup object and call setup
 *
 * @param engine
 */
void setup::setupRingPolymer(engine::RingPolymerEngine &engine)
{
    RingPolymerSetup ringPolymerSetup(engine);
    ringPolymerSetup.setup();
}

/**
 * @brief setup a ring polymer simulation
 *
 */
void RingPolymerSetup::setup()
{
    if (!settings::RingPolymerSettings::isNumberOfBeadsSet())
        throw customException::InputFileException("Number of beads not set for ring polymer simulation");

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
    {
        simulationBox::SimulationBox bead;
        bead.copy(_engine.getSimulationBox());

        _engine.addRingPolymerBead(bead);
    }

    if (settings::FileSettings::isRingPolymerStartFileNameSet())
    {
        std::cout << "read ring polymer restart file" << std::endl;
        readInput::ringPolymer::readRingPolymerRestartFile(_engine);
    }
    else
    {
        auto initVelocities = [](auto &bead)
        {
            // TODO: test if initializeVelocities is set
            maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
            maxwellBoltzmann.initializeVelocities(bead);
        };

        std::ranges::for_each(_engine.getRingPolymerBeads(), initVelocities);
    }
}