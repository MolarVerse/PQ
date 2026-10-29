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

#include <algorithm>
#include <format>
#include <string>

#include "enums/manostat.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "manostat.hpp"
#include "manostatSettings.hpp"
#include "mdEngine.hpp"
#include "simulationBox.hpp"

namespace setup
{

    /**
     * @brief wrapper for setupManostat
     *
     * @param engine
     */
    void setupManostat(engine::Engine &engine)
    {
        if (!settings::GeneralSettings::isMDJobType())
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
        using enum ManostatType;

        const auto &manostatSettings = _engine.getSettings().manostatSettings;

        if (manostatSettings.isBerendsenBased() &&
            settings::GeneralSettings::getVirialType() == VirialType::ATOMIC &&
            std::ranges::any_of(
                _engine.getSimulationBox().getMolecules(),
                [](const auto &molecule)
                { return molecule.getNumberOfAtoms() > 1; }
            ))
        {
            const std::string warning =
                "Atomic virial is inconsistent with molecular scaling for "
                "multi-atom molecules. Recommended: virial = molecular. "
                "Continuing with virial = atomic.";
            out::StdoutOutput::writeSetupWarning(warning);
            _engine.getLogOutput().writeSetupWarning(warning);
        }

        validateIsotropyFixedAxisCombination();
        writeSetupInfo();
    }

    /**
     * @brief validate isotropy and fixed_axis combination
     *
     * @throws SetupException if semi-isotropic mode conflicts with fixed_axis
     */
    void ManostatSetup::validateIsotropyFixedAxisCombination() const
    {
        using enum Isotropy;

        const auto isotropy =
            _engine.getSettings().manostatSettings.getIsotropy();
        const auto fixedAxis =
            _engine.getSettings().manostatSettings.getFixedAxis();
        const auto manostatType =
            _engine.getSettings().manostatSettings.getManostatType();

        if (manostatType != ManostatType::NONE && fixedAxis == FixedAxis::ALL)
        {
            throw exc::UserInputException(
                "Invalid combination: all axes cannot be fixed while a "
                "manostat is selected."
            );
        }

        if (isSemiIsotropic(isotropy) && fixedAxis != FixedAxis::NONE)
        {
            const auto anisoAxis = get2DAnisotropicAxis(isotropy);
            const auto allowedFixedAxis =
                static_cast<FixedAxis>(1U << anisoAxis);

            if (fixedAxis != allowedFixedAxis)
            {
                throw exc::UserInputException(
                    "Invalid combination: semi-isotropic pressure coupling "
                    "only allows fixing the anisotropic axis or none."
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

        if (_engine.getSettings().manostatSettings.isBerendsenBased())
            writeBerendsenSetup();

        if (_engine.getSettings().manostatSettings.getManostatType() !=
            ManostatType::NONE)
            writeIsotropy();
    }

    /**
     * @brief write manostat selection
     *
     */
    void ManostatSetup::writeManostatSelection() const
    {
        auto &logOutput = _engine.getLogOutput();

        switch (_engine.getSettings().manostatSettings.getManostatType())
        {
            using enum ManostatType;

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

        const auto pressure =
            _engine.getSettings().manostatSettings.getTargetPressure();
        const auto tau =
            _engine.getSettings().manostatSettings.getTauManostat();
        const auto compr =
            _engine.getSettings().manostatSettings.getCompressibility();

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

        switch (_engine.getSettings().manostatSettings.getIsotropy())
        {
            using enum Isotropy;

            case ISOTROPIC:
                logOutput.writeSetupInfo("Isotropy: isotropic");
                break;

            case SEMI_ISOTROPIC_XY:
            case SEMI_ISOTROPIC_XZ:
            case SEMI_ISOTROPIC_YZ:
            {
                std::string isoAxesStr;
                std::string anisoAxisStr;

                if (_engine.getSettings().manostatSettings.getIsotropy() ==
                    SEMI_ISOTROPIC_XY)
                {
                    isoAxesStr   = "x, y";
                    anisoAxisStr = "z";
                }
                else if (_engine.getSettings().manostatSettings.getIsotropy() ==
                         SEMI_ISOTROPIC_XZ)
                {
                    isoAxesStr   = "x, z";
                    anisoAxisStr = "y";
                }
                else if (_engine.getSettings().manostatSettings.getIsotropy() ==
                         SEMI_ISOTROPIC_YZ)
                {
                    isoAxesStr   = "y, z";
                    anisoAxisStr = "x";
                }

                logOutput.writeSetupInfo(
                    std::format("Isotropy:         semi-isotropic")
                );
                logOutput.writeSetupInfo(
                    std::format("Anisotropic axis: {}", anisoAxisStr)
                );
                logOutput.writeSetupInfo(
                    std::format("Isotropic axes:   {}", isoAxesStr)
                );
                break;
            }

            case ANISOTROPIC:
                logOutput.writeSetupInfo("Isotropy: anisotropic");
                break;

            case FULL_ANISOTROPIC:
                logOutput.writeSetupInfo("Isotropy: full anisotropic");
                break;
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
