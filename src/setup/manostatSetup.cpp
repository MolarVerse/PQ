/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

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