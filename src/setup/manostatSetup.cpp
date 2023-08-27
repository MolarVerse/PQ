#include "manostatSetup.hpp"

#include "constants.hpp"          // for _PS_TO_FS_
#include "engine.hpp"             // for Engine
#include "exceptions.hpp"         // for InputFileException, customException
#include "manostat.hpp"           // for BerendsenManostat, Manostat, manostat
#include "manostatSettings.hpp"   // for ManostatSettings
#include "timings.hpp"            // for Timings

#include <format>   // for format
#include <string>   // for operator==

using namespace setup;

/**
 * @brief wrapper for setupManostat
 *
 */
void setup::setupManostat(engine::Engine &engine)
{
    ManostatSetup manostatSetup(engine);
    manostatSetup.setup();
}

/**
 * @brief setup manostat
 *
 * @details checks if a manostat was set in the input file,
 * If a manostat was selected than the user has to provide a target pressure for the manostat.
 *
 * @note the base class manostat does not apply any pressure coupling to the system and therefore it represents the none
 * manostat.
 *
 * @throws InputFileException if no pressure was set for the manostat
 *
 */
void ManostatSetup::setup()
{
    const auto manostatType = settings::ManostatSettings::getManostatType();

    if (manostatType != "none")
        if (!settings::ManostatSettings::isPressureSet())
            throw customException::InputFileException(std::format("Pressure not set for {} manostat", manostatType));

    if (manostatType == "berendsen")
        _engine.makeManostat(manostat::BerendsenManostat(settings::ManostatSettings::getTargetPressure(),
                                                         settings::ManostatSettings::getTauManostat() * constants::_PS_TO_FS_,
                                                         settings::ManostatSettings::getCompressibility()));
    else
        _engine.makeManostat(manostat::Manostat());

    _engine.getManostat().setTimestep(_engine.getTimings().getTimestep());
}