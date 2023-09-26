#include "manostatSetup.hpp"

#include "berendsenManostat.hpp"             // for BerendsenManostat
#include "constants/conversionFactors.hpp"   // for _PS_TO_FS_
#include "engine.hpp"                        // for Engine
#include "exceptions.hpp"                    // for InputFileException, customException
#include "manostat.hpp"                      // for BerendsenManostat, Manostat, manostat
#include "manostatSettings.hpp"              // for ManostatSettings
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat

#include <format>   // for format
#include <string>   // for operator==

using namespace setup;

/**
 * @brief wrapper for setupManostat
 *
 * @param engine
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

    if (manostatType != settings::ManostatType::NONE)
        if (!settings::ManostatSettings::isPressureSet())
            throw customException::InputFileException(std::format("Pressure not set for {} manostat", string(manostatType)));

    if (manostatType == settings::ManostatType::BERENDSEN)
        _engine.makeManostat(manostat::BerendsenManostat(settings::ManostatSettings::getTargetPressure(),
                                                         settings::ManostatSettings::getTauManostat() * constants::_PS_TO_FS_,
                                                         settings::ManostatSettings::getCompressibility()));

    else if (manostatType == settings::ManostatType::STOCHASTIC_RESCALING)
        _engine.makeManostat(
            manostat::StochasticRescalingManostat(settings::ManostatSettings::getTargetPressure(),
                                                  settings::ManostatSettings::getTauManostat() * constants::_PS_TO_FS_,
                                                  settings::ManostatSettings::getCompressibility()));

    else
        _engine.makeManostat(manostat::Manostat());
}