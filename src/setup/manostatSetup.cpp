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
#include "exceptions.hpp"
#include "manostat.hpp"           // for BerendsenManostat, Manostat, manostat
#include "manostatSettings.hpp"   // for settings::ManostatSettings
#include "mdEngine.hpp"           // for Engine
#include "settings.hpp"           // for IsMDJobType
#include "stochasticRescalingManostat.hpp"

namespace setup
{

    /**
     * @brief wrapper for setupManostat
     *
     * @param engine
     */
    void setupManostat(engine::Engine &engine)
    {
        if (!settings::Settings::isMDJobType())
            return;

        out::StdoutOutput::writeSetup("Manostat");
        engine.getLogOutput().writeSetup("Manostat");

        ManostatSetup manostatSetup(dynamic_cast<engine::MDEngine &>(engine));
        manostatSetup.setup();
    }

    /**
     * @brief Construct a new Manostat Setup:: Manostat Setup object
     *
     * @param engine
     */
    ManostatSetup::ManostatSetup(engine::MDEngine &engine) : _engine(engine) {}

    /**
     * @brief setup manostat
     *
     * @note the base class manostat does not apply any pressure coupling to the
     * system and therefore it represents the none manostat.
     */
    void ManostatSetup::setup()
    {
        using enum settings::ManostatType;

        const auto manostatType = settings::ManostatSettings::getManostatType();

        if (manostatType == BERENDSEN)
            setupBerendsenManostat();

        else if (manostatType == STOCHASTIC_RESCALING)
            setupStochasticRescalingManostat();

        else
            _engine.makeManostat(manostat::Manostat());

        validateIsotropyFixedAxisCombination();
        writeSetupInfo();
    }

    /**
     * @brief setup berendsen manostat
     *
     * @details constructs a berendsen manostat and adds it to the engine
     *
     */
    void ManostatSetup::setupBerendsenManostat()
    {
        const auto isotropy = settings::ManostatSettings::getIsotropy();
        const auto pTarget  = settings::ManostatSettings::getTargetPressure();
        const auto tau =
            settings::ManostatSettings::getTauManostat() * PS_TO_FS;
        const auto compress = settings::ManostatSettings::getCompressibility();
        const auto aniso = settings::ManostatSettings::get2DAnisotropicAxis();
        const auto iso   = settings::ManostatSettings::get2DIsotropicAxes();
        const auto fixedAxis = settings::ManostatSettings::getFixedAxis();

        switch (isotropy)
        {
            using enum settings::Isotropy;

                // clang-format off
        case SEMI_ISOTROPIC:
            _engine.makeManostat(manostat::SemiIsotropicBerendsenManostat(pTarget, tau, compress, aniso, iso, fixedAxis));
            break;

        case ANISOTROPIC:
            _engine.makeManostat(manostat::AnisotropicBerendsenManostat(pTarget, tau, compress, fixedAxis));
            break;

        case FULL_ANISOTROPIC:
            _engine.makeManostat(manostat::FullAnisotropicBerendsenManostat(pTarget, tau, compress, fixedAxis));
            break;

        case NONE: // fall through
        case ISOTROPIC:
            _engine.makeManostat(manostat::BerendsenManostat(pTarget, tau, compress, fixedAxis));

                // clang-format on
        }
    }

    /**
     * @brief setup stochastic rescaling manostat
     *
     * @details constructs a stochastic rescaling manostat and adds it to the
     * engine
     *
     */
    void ManostatSetup::setupStochasticRescalingManostat()
    {
        const auto isotropy = settings::ManostatSettings::getIsotropy();
        const auto pTarget  = settings::ManostatSettings::getTargetPressure();
        const auto tau =
            settings::ManostatSettings::getTauManostat() * PS_TO_FS;
        const auto compress = settings::ManostatSettings::getCompressibility();
        const auto aniso = settings::ManostatSettings::get2DAnisotropicAxis();
        const auto iso   = settings::ManostatSettings::get2DIsotropicAxes();
        const auto fixedAxis = settings::ManostatSettings::getFixedAxis();

        switch (isotropy)
        {
            using enum settings::Isotropy;

                // clang-format off

        case SEMI_ISOTROPIC:
            _engine.makeManostat(manostat::SemiIsotropicStochasticRescalingManostat(pTarget, tau, compress, aniso, iso, fixedAxis));
            break;

        case ANISOTROPIC:
            _engine.makeManostat(manostat::AnisotropicStochasticRescalingManostat(pTarget, tau, compress, fixedAxis));
            break;

        case FULL_ANISOTROPIC:
            _engine.makeManostat(manostat::FullAnisotropicStochasticRescalingManostat(pTarget, tau, compress, fixedAxis));
            break;

        case NONE: // fall through
        case ISOTROPIC:
            _engine.makeManostat(manostat::StochasticRescalingManostat(pTarget, tau, compress, fixedAxis));

                // clang-format on
        }
    }

    /**
     * @brief validate isotropy and fixed_axis combination
     *
     * @throws SetupException if semi-isotropic mode conflicts with fixed_axis
     */
    void ManostatSetup::validateIsotropyFixedAxisCombination()
    {
        using enum settings::Isotropy;

        const auto isotropy     = settings::ManostatSettings::getIsotropy();
        const auto fixedAxis    = settings::ManostatSettings::getFixedAxis();
        const auto manostatType = settings::ManostatSettings::getManostatType();

        if (manostatType != settings::ManostatType::NONE &&
            fixedAxis == settings::FixedAxis::ALL)
        {
            throw exc::UserInputException(
                "Invalid combination: all axes cannot be fixed while a "
                "manostat is selected."
            );
        }

        if (isotropy == SEMI_ISOTROPIC &&
            fixedAxis != settings::FixedAxis::NONE)
        {
            const auto anisoAxis =
                settings::ManostatSettings::get2DAnisotropicAxis();
            const auto allowedFixedAxis =
                static_cast<settings::FixedAxis>(1U << anisoAxis);

            if (fixedAxis != allowedFixedAxis)
            {
                throw exc::UserInputException(
                    "Invalid combination: semi-isotropic pressure coupling "
                    "only "
                    "allows fixing the anisotropic axis or none."
                );
            }
        }
    }

    /**
     * @brief write setup info
     *
     */
    void ManostatSetup::writeSetupInfo() const
    {
        writeManostatSelection();

        if (settings::ManostatSettings::isBerendsenBased())
            writeBerendsenSetup();

        if (settings::ManostatSettings::getManostatType() !=
            settings::ManostatType::NONE)
            writeIsotropy();
    }

    /**
     * @brief write manostat selection
     *
     */
    void ManostatSetup::writeManostatSelection() const
    {
        auto &logOutput = _engine.getLogOutput();

        switch (settings::ManostatSettings::getManostatType())
        {
            using enum settings::ManostatType;

            case BERENDSEN:
                logOutput.writeSetupInfo("Berendsen manostat selected");
                break;

            case STOCHASTIC_RESCALING:
                logOutput.writeSetupInfo(
                    "Stochastic rescaling manostat selected"
                );
                break;

            case NONE: logOutput.writeSetupInfo("No manostat selected");
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

        const auto pressure = settings::ManostatSettings::getTargetPressure();
        const auto tau      = settings::ManostatSettings::getTauManostat();
        const auto compr    = settings::ManostatSettings::getCompressibility();

        logOutput.writeSetupInfo(std::format("Target pressure: {}", pressure));
        logOutput.writeSetupInfo(std::format("Relaxation time: {} ps", tau));
        logOutput.writeSetupInfo(
            std::format("Compressibility: {} bar⁻¹", compr)
        );
        logOutput.writeEmptyLine();
    }

    /**
     * @brief write isotropy setup
     *
     */
    void ManostatSetup::writeIsotropy() const
    {
        auto &logOutput = _engine.getLogOutput();

        switch (settings::ManostatSettings::getIsotropy())
        {
            using enum settings::Isotropy;

            case ISOTROPIC:
                logOutput.writeSetupInfo("Isotropy: isotropic");
                break;

            case SEMI_ISOTROPIC:
            {
                const auto anisoAxis =
                    settings::ManostatSettings::get2DAnisotropicAxis();
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

            case NONE: logOutput.writeSetupInfo("Isotropy: isotropic");
        }

        logOutput.writeEmptyLine();
    }

    /**
     * @brief Get the Engine object
     *
     * @return MDEngine&
     */
    engine::MDEngine &ManostatSetup::getEngine() const { return _engine; }

}   // namespace setup
