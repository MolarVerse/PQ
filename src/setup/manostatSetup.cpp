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
#include "typeAliases.hpp"

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
ManostatSetup::ManostatSetup(MDEngine &engine) : _engine(engine) {}

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
        throw InputFileException(
            std::format(
                "Pressure not set for {} manostat",
                string(ManostatSettings::getManostatType())
            )
        );
}

/**
 * @brief setup berendsen manostat
 *
 * @details constructs a berendsen manostat and adds it to the engine
 *
 */
void ManostatSetup::setupBerendsenManostat()
{
    const auto isotropy = ManostatSettings::getIsotropy();
    const auto pTarget  = ManostatSettings::getTargetPressure();
    const auto tau      = ManostatSettings::getTauManostat() * _PS_TO_FS_;
    const auto compress = ManostatSettings::getCompressibility();
    const auto aniso    = ManostatSettings::get2DAnisotropicAxis();
    const auto iso      = ManostatSettings::get2DIsotropicAxes();

    switch (isotropy)
    {
        using enum Isotropy;

            // clang-format off
        case SEMI_ISOTROPIC:
            _engine.makeManostat(SemiIsotropicBerendsenManostat(pTarget, tau, compress, aniso, iso));
            break;

        case ANISOTROPIC:
            _engine.makeManostat(AnisotropicBerendsenManostat(pTarget, tau, compress));
            break;

        case FULL_ANISOTROPIC:
            _engine.makeManostat(FullAnisotropicBerendsenManostat(pTarget, tau, compress));
            break;

        case ISOTROPIC: // fall through
        default:
            _engine.makeManostat(BerendsenManostat(pTarget, tau, compress));

            // clang-format on
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
    const auto isotropy = ManostatSettings::getIsotropy();
    const auto pTarget  = ManostatSettings::getTargetPressure();
    const auto tau      = ManostatSettings::getTauManostat() * _PS_TO_FS_;
    const auto compress = ManostatSettings::getCompressibility();
    const auto aniso    = ManostatSettings::get2DAnisotropicAxis();
    const auto iso      = ManostatSettings::get2DIsotropicAxes();

    switch (isotropy)
    {
        using enum Isotropy;

            // clang-format off

        case SEMI_ISOTROPIC:
            _engine.makeManostat(pq::SemiIsoStochasticManostat(pTarget, tau, compress, aniso, iso));
            break;

        case ANISOTROPIC:
            _engine.makeManostat(pq::AnisoStochasticManostat(pTarget, tau, compress));
            break;

        case FULL_ANISOTROPIC:
            _engine.makeManostat(pq::FullAnisoStochasticManostat(pTarget, tau, compress));
            break;

        case ISOTROPIC: // fall through
        default:
            _engine.makeManostat(pq::StochasticManostat(pTarget, tau, compress));

            // clang-format on
    }
}

/**
 * @brief write setup info
 *
 */
void ManostatSetup::writeSetupInfo() const
{
    writeManostatSelection();

    if (ManostatSettings::isBerendsenBased())
        writeBerendsenSetup();

    if (ManostatSettings::getManostatType() != ManostatType::NONE)
        writeIsotropy();
}

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
 * @brief write berendsen setup
 *
 */
void ManostatSetup::writeBerendsenSetup() const
{
    auto &logOutput = _engine.getLogOutput();

    const auto pressure = ManostatSettings::getTargetPressure();
    const auto tau      = ManostatSettings::getTauManostat();
    const auto compr    = ManostatSettings::getCompressibility();

    logOutput.writeSetupInfo(std::format("Target pressure: {}", pressure));
    logOutput.writeSetupInfo(std::format("Relaxation time: {} ps", tau));
    logOutput.writeSetupInfo(std::format("Compressibility: {} bar⁻¹", compr));
    logOutput.writeEmptyLine();
}

/**
 * @brief write isotropy setup
 *
 */
void ManostatSetup::writeIsotropy() const
{
    auto &logOutput = _engine.getLogOutput();

    switch (ManostatSettings::getIsotropy())
    {
        using enum Isotropy;

        case ISOTROPIC: logOutput.writeSetupInfo("Isotropy: isotropic"); break;

        case SEMI_ISOTROPIC:
        {
            const auto  anisoAxis = ManostatSettings::get2DAnisotropicAxis();
            std::string anisoAxisStr;
            std::string isoAxesStr;

            if (anisoAxis == 0)
            {
                isoAxesStr   = "y, z";
                anisoAxisStr = "x";
            }

            else if (anisoAxis == 1)
            {
                isoAxesStr   = "x, z";
                anisoAxisStr = "y";
            }

            else
            {
                isoAxesStr   = "x, y";
                anisoAxisStr = "z";
            }

            // clang-format off
            logOutput.writeSetupInfo(std::format("Isotropy:         semi-isotropic"));
            logOutput.writeSetupInfo(std::format("Anisotropic axis: {}", anisoAxisStr));
            logOutput.writeSetupInfo(std::format("Isotropic axes:   {}", isoAxesStr));
            // clang-format on
            break;
        }

        case ANISOTROPIC:
            logOutput.writeSetupInfo("Isotropy: anisotropic");
            break;

        case FULL_ANISOTROPIC:
            logOutput.writeSetupInfo("Isotropy: full anisotropic");
            break;

        default: logOutput.writeSetupInfo("Isotropy: isotropic");
    }

    logOutput.writeEmptyLine();
}

/**
 * @brief Get the Engine object
 *
 * @return MDEngine&
 */
MDEngine &ManostatSetup::getEngine() const { return _engine; }