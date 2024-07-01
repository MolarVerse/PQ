/*****************************************************************************
<GPL_HEADER>

    PQ
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

#include <format>   // for format
#include <string>   // for operator==

#include "berendsenManostat.hpp"             // for BerendsenManostat
#include "constants/conversionFactors.hpp"   // for _PS_TO_FS_
#include "exceptions.hpp"         // for InputFileException, customException
#include "manostat.hpp"           // for BerendsenManostat, Manostat, manostat
#include "manostatSettings.hpp"   // for ManostatSettings
#include "mdEngine.hpp"           // for Engine
#include "settings.hpp"           // for IsMDJobType
#include "stochasticRescalingManostat.hpp"   // for StochasticRescalingManostat

using namespace setup;
using namespace engine;
using namespace settings;
using namespace manostat;
using namespace customException;
using namespace constants;

/**
 * @brief wrapper for setupManostat
 *
 * @param engine
 */
void setup::setupManostat(Engine &engine)
{
    if (!Settings::isMDJobType())
        return;

    engine.getStdoutOutput().writeSetup("Manostat");
    engine.getLogOutput().writeSetup("Manostat");

    ManostatSetup manostatSetup(dynamic_cast<MDEngine &>(engine));
    manostatSetup.setup();
}

/**
 * @brief Construct a new Manostat Setup:: Manostat Setup object
 *
 * @param engine
 */
ManostatSetup::ManostatSetup(MDEngine &engine) : _engine(engine){};

/**
 * @brief setup manostat
 *
 * @details checks if a manostat was set in the input file,
 * If a manostat was selected than the user has to provide a target pressure for
 * the manostat.
 *
 * @note the base class manostat does not apply any pressure coupling to the
 * system and therefore it represents the none manostat.
 *
 * @throws InputFileException if no pressure was set for the manostat
 *
 */
void ManostatSetup::setup()
{
    using enum ManostatType;

    const auto manostatType = ManostatSettings::getManostatType();

    if (manostatType != NONE)
        isPressureSet();

    if (manostatType == BERENDSEN)
        setupBerendsenManostat();

    else if (manostatType == STOCHASTIC_RESCALING)
        setupStochasticRescalingManostat();

    else
        _engine.makeManostat(Manostat());

    writeSetupInfo();

    _engine.getLogOutput().writeEmptyLine();
}

/**
 * @brief check if pressure is set for the manostat
 *
 * @throws InputFileException if no pressure was set for the manostat
 *
 */
void ManostatSetup::isPressureSet() const
{
    if (!ManostatSettings::isPressureSet())
        throw InputFileException(std::format(
            "Pressure not set for {} manostat",
            string(ManostatSettings::getManostatType())
        ));
}

/**
 * @brief setup berendsen manostat
 *
 * @details constructs a berendsen manostat and adds it to the engine
 *
 */
void ManostatSetup::setupBerendsenManostat()
{
    using enum Isotropy;

    const auto isotropy = ManostatSettings::getIsotropy();

    if (isotropy == SEMI_ISOTROPIC)
    {
        _engine.makeManostat(SemiIsotropicBerendsenManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility(),
            ManostatSettings::get2DAnisotropicAxis(),
            ManostatSettings::get2DIsotropicAxes()
        ));
    }

    else if (isotropy == ANISOTROPIC)
    {
        _engine.makeManostat(AnisotropicBerendsenManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }

    else if (isotropy == FULL_ANISOTROPIC)
    {
        _engine.makeManostat(FullAnisotropicBerendsenManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }

    else
    {
        _engine.makeManostat(BerendsenManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }
}

/**
 * @brief setup stochastic rescaling manostat
 *
 * @details constructs a stochastic rescaling manostat and adds it to the engine
 *
 */
void ManostatSetup::setupStochasticRescalingManostat()
{
    using enum Isotropy;

    const auto isotropy = ManostatSettings::getIsotropy();

    if (isotropy == SEMI_ISOTROPIC)
    {
        _engine.makeManostat(SemiIsotropicStochasticRescalingManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility(),
            ManostatSettings::get2DAnisotropicAxis(),
            ManostatSettings::get2DIsotropicAxes()
        ));
    }

    else if (isotropy == ANISOTROPIC)
    {
        _engine.makeManostat(AnisotropicStochasticRescalingManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }

    else if (isotropy == FULL_ANISOTROPIC)
    {
        _engine.makeManostat(AnisotropicStochasticRescalingManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }

    else
    {
        _engine.makeManostat(StochasticRescalingManostat(
            ManostatSettings::getTargetPressure(),
            ManostatSettings::getTauManostat() * _PS_TO_FS_,
            ManostatSettings::getCompressibility()
        ));
    }
}

/**
 * @brief write setup info
 *
 */
void ManostatSetup::writeSetupInfo() const { writeManostatSelection(); }

/**
 * @brief write manostat selection
 *
 */
void ManostatSetup::writeManostatSelection() const
{
    auto &logOutput = _engine.getLogOutput();

    switch (ManostatSettings::getManostatType())
    {
        using enum ManostatType;

        case BERENDSEN:
            logOutput.writeSetupInfo("Berendsen manostat selected");
            break;

        case STOCHASTIC_RESCALING:
            logOutput.writeSetupInfo("Stochastic rescaling manostat selected");
            break;

        default: logOutput.writeSetupInfo("No manostat selected");
    }

    logOutput.writeEmptyLine();
}

/**
 * @brief Get the Engine object
 *
 * @return MDEngine&
 */
MDEngine &ManostatSetup::getEngine() const { return _engine; }