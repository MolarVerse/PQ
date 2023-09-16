#include "ringPolymerSetup.hpp"

#include "maxwellBoltzmann.hpp"      // for MaxwellBoltzmann
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings

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

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads() - 1; ++i)
    {
        auto bead = _engine.getSimulationBox();

        maxwellBoltzmann::MaxwellBoltzmann maxwellBoltzmann;
        maxwellBoltzmann.initializeVelocities(bead);

        _engine.addRingPolymerBead(bead);
    }
}