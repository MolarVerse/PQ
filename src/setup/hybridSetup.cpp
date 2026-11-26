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

#include "hybridSetup.hpp"

#include <algorithm>
#include <format>
#include <string>
#include <vector>

#include "engine.hpp"
#include "enums/hybrid.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "hybridSettings.hpp"
#include "qmSettings.hpp"

namespace setup
{

    /**
     * @brief wrapper to build HybridSetup object and call setup
     *
     * @param engine
     */
    void setupHybrid(engine::Engine &engine)
    {
        if (!isHybridJobtype(engine.getSettings().generalSettings.getJobtype()))
            return;

        out::StdoutOutput::writeSetup("Hybrid Configuration");
        engine.getLogOutput().writeSetup("Hybrid Configuration");

        HybridSetup hybridSetup(engine);
        hybridSetup.setup();
    }

    /**
     * @brief Construct a new HybridSetup object
     *
     * @param engine
     */
    HybridSetup::HybridSetup(engine::Engine &engine) : _engine(engine) {}

    /**
     * @brief setup Hybrid-MD
     *
     */
    void HybridSetup::setup()
    {
        validateQMMethod(_engine.getSettings().qmSettings.getQMMethod());
        setupInnerRegionCenter();
        setupForcedCoreList();
        setupForcedLayerList();
        setupForcedOuterList();
        validateQMChargeSettings();
        validateQMReferenceEnergies();
        checkZoneRadii();

        setupWriteInfo();
    }

    /**
     * @brief Check if chosen QM method is available for hybrid type
     * calculations
     *
     * @throws exc::InputFileException if the QM method is not supported
     * for hybrid type calculations
     */
    void HybridSetup::validateQMMethod(QMMethod qmMethod)
    {
        using enum QMMethod;

        const auto errorMsg = std::format(
            "QM method \"{}\" is not supported for hybrid type "
            "calculations. Supported QM methods are \"dftbplus\", "
            "\"turbomole\" and \"orca\".",
            QMMethodMeta::toString(qmMethod)

        );

        switch (qmMethod)
        {
            case DFTBPLUS:
            case TURBOMOLE:
            case ORCA: break;
            case PYSCF:
            case ASE_DFTBPLUS:
            case ASE_XTB:
            case MACE:
            case FENNOL:
            case NONE: throw exc::InputFileException(errorMsg);
        }
    }

    /**
     * @brief validate the inner region center
     *
     * @throw exc::UserInputException if an atom index is out of range
     */
    void HybridSetup::setupInnerRegionCenter()
    {
        const auto numberOfAtoms =
            _engine.get_simulationBox().getNumberOfAtoms();

        for (const auto index :
             _engine.getSettings().hybridSettings.get_innerRegionCenter())
        {
            if (index >= numberOfAtoms)
            {
                throw exc::UserInputException(
                    std::format(
                        "Inner region center atom index {} out of range",
                        index
                    )
                );
            }
        }
    }

    /**
     * @brief setup forced core list
     *
     */
    void HybridSetup::setupForcedCoreList()
    {
        _engine.get_simulationBox().setupForcedCoreMolecules(
            _engine.getSettings().hybridSettings.get_forcedCoreList()
        );
    }

    /**
     * @brief setup forced layer list
     *
     */
    void HybridSetup::setupForcedLayerList()
    {
        _engine.get_simulationBox().setupForcedLayerMolecules(
            _engine.getSettings().hybridSettings.get_forcedLayerList()
        );
    }

    /**
     * @brief setup forced outer list
     *
     */
    void HybridSetup::setupForcedOuterList()
    {
        _engine.get_simulationBox().setupForcedOuterMolecules(
            _engine.getSettings().hybridSettings.get_forcedOuterList()
        );
    }

    /**
     * @brief Validate zone radii configuration for hybrid calculations
     *
     * @throws exc::InputFileException if the core radius is larger than
     * the layer radius
     * @throws exc::InputFileException if the smoothing region is too
     * thick for the chosen combination of core and layer radius
     * @throws exc::InputFileException if the layer radius exceeds one
     quarter of the smallest box dimension (minimum image convention)
     * @throws exc::InputFileException if the sum of layer radius and
     point charge thickness exceeds three quarters of the smallest box dimension
     (includes point charges from beyond immediate neighboring cells)
     */
    void HybridSetup::checkZoneRadii()
    {
        const auto coreRadius =
            _engine.getSettings().hybridSettings.get_coreRadius();
        const auto layerRadius =
            _engine.getSettings().hybridSettings.get_layerRadius();
        const auto smoothingRegionThickness =
            _engine.getSettings().hybridSettings.get_smoothingRegionThickness();
        const auto pointChargeThickness =
            _engine.getSettings().hybridSettings.get_pointChargeThickness();
        const auto minimalBoxDimension =
            _engine.get_simulationBox().getMinimalBoxDimension();

        if (coreRadius > layerRadius)
        {
            throw exc::InputFileException(
                std::format(
                    "Core radius ({}) cannot be larger than layer radius ({})",
                    coreRadius,
                    layerRadius
                )
            );
        }

        if (coreRadius > (layerRadius - smoothingRegionThickness))
        {
            throw exc::InputFileException(
                std::format(
                    "Smoothing region is too thick ({}) for the chosen "
                    "combination of core ({}) and layer radius ({})",
                    smoothingRegionThickness,
                    coreRadius,
                    layerRadius
                )
            );
        }

        if (layerRadius > (minimalBoxDimension / 4))
        {
            throw exc::InputFileException(
                std::format(
                    "Layer radius ({}) exceeds one quarter of the smallest "
                    "box "
                    "dimension ({}). This configuration is not allowed to "
                    "ensure "
                    "compliance with the minimum image convention.",
                    layerRadius,
                    minimalBoxDimension
                )
            );
        }

        if ((layerRadius + pointChargeThickness) >
            (minimalBoxDimension * 3 / 2))
        {
            throw exc::InputFileException(
                std::format(
                    "Layer radius ({}) plus point charge thickness ({}) "
                    "exceeds three halves of the smallest box dimension ({}). "
                    "This configuration is not allowed, as it would include "
                    "point "
                    "charges from beyond the immediate neighboring cells.",
                    layerRadius,
                    pointChargeThickness,
                    minimalBoxDimension
                )
            );
        }
    }

    /**
     * @brief Validate the compatibility between QM charge settings and moltype
     * 0 presence
     *
     * @details This function checks for a configuration conflict in hybrid
     * QM/MM calculations where MM charges are requested (qm_charges = mm) but
     * QM atoms (moltype 0) are present in the system.
     *
     * @throws exc::InputFileException if MM charges are requested but
     * atoms without moltype are present in the simulation box
     */
    void HybridSetup::validateQMChargeSettings()
    {
        const auto mmChargesRequested =
            !_engine.getSettings().hybridSettings.get_useQMCharges();
        const auto qmAtomsPresent =
            _engine.get_simulationBox().moleculeTypeExists(MolType{0});

        if (mmChargesRequested && qmAtomsPresent)
        {
            throw exc::InputFileException(
                "Invalid configuration: MM charges requested (qm_charges = mm) "
                "in "
                "input file but atoms with moltype \"0\" are present in the "
                "system. Either set \"qm_charges = qm\" or ensure all atoms "
                "have a"
                "non-zero moltype."
            );
        }
    }

    /**
     * @brief Validate that no moltype 0 atoms are present when QM reference
     * energies are used
     *
     * @throws exc::InputFileException if a moltype has a QM reference energy
     * and atoms with moltype 0 are present in the simulation box
     */
    void HybridSetup::validateQMReferenceEnergies()
    {
        auto &simBox = _engine.get_simulationBox();

        const auto hasReference = [](const auto &moleculeType)
        { return moleculeType.hasQMReferenceEnergy(); };

        const auto usesReferences =
            std::ranges::any_of(simBox.getMoleculeTypes(), hasReference);

        if (usesReferences && simBox.moleculeTypeExists(MolType{0}))
        {
            throw exc::InputFileException(
                "Atoms with moltype \"0\" are not allowed when QM reference "
                "energies are set in the moldescriptor file."
            );
        }
    }

    /**
     * @brief write info about the hybrid setup
     *
     */
    void HybridSetup::setupWriteInfo() const
    {
        auto &logOutput = _engine.getLogOutput();

        const auto jobtype = _engine.getSettings().generalSettings.getJobtype();
        const auto smoothingMethod =
            _engine.getSettings().hybridSettings.get_smoothing();
        const auto innerRegionCenter =
            _engine.getSettings().hybridSettings.get_innerRegionCenter();
        const auto forcedCoreList =
            _engine.getSettings().hybridSettings.get_forcedCoreList();
        const auto forcedOuterList =
            _engine.getSettings().hybridSettings.get_forcedOuterList();
        const auto useQMCharges =
            _engine.getSettings().hybridSettings.get_useQMCharges();
        const auto coreRadius =
            _engine.getSettings().hybridSettings.get_coreRadius();
        const auto layerRadius =
            _engine.getSettings().hybridSettings.get_layerRadius();
        const auto smoothingRegionThickness =
            _engine.getSettings().hybridSettings.get_smoothingRegionThickness();
        const auto pointChargeThickness =
            _engine.getSettings().hybridSettings.get_pointChargeThickness();

        const auto formatIndexList =
            []<typename T>(const std::vector<T> &indices)
        {
            if (indices.empty())
                return std::string("none");

            auto formatted = std::to_string(indices.front());

            for (size_t i = 1; i < indices.size(); ++i)
                formatted += std::format(", {}", indices[i]);

            return formatted;
        };

        if (jobtype == JobType::QMMM_MD)
        {
            const auto *const jobtypeMsg = "Hybrid type:                 QM/MM";

            logOutput.writeSetupInfo(jobtypeMsg);
        }

        const auto smoothingMethodMsg = std::format(
            "Smoothing method:            {}",
            SmoothingMethodMeta::toString(smoothingMethod)
        );
        const auto innerRegionCenterMsg = std::format(
            "Inner region center atoms:   {}",
            formatIndexList(innerRegionCenter)
        );
        const auto forcedCoreListMsg = std::format(
            "Forced inner molecules:      {}",
            formatIndexList(forcedCoreList)
        );
        const auto forcedOuterListMsg = std::format(
            "Forced outer molecules:      {}",
            formatIndexList(forcedOuterList)
        );
        const auto qmChargesSourceMsg = std::format(
            "QM charge source:            {}",
            useQMCharges ? "qm" : "mm"
        );
        const auto coreRadiusMsg =
            std::format("Core radius:                 {} Å", coreRadius);
        const auto layerRadiusMsg =
            std::format("Layer radius:                {} Å", layerRadius);
        const auto smoothingRegionThicknessMsg = std::format(
            "Smoothing region thickness:  {} Å",
            smoothingRegionThickness
        );
        const auto pointChargeThicknessMsg = std::format(
            "Point charge thickness:      {} Å",
            pointChargeThickness
        );

        logOutput.writeSetupInfo(smoothingMethodMsg);
        logOutput.writeSetupInfo(innerRegionCenterMsg);
        logOutput.writeSetupInfo(forcedCoreListMsg);
        logOutput.writeSetupInfo(forcedOuterListMsg);
        logOutput.writeSetupInfo(qmChargesSourceMsg);
        logOutput.writeEmptyLine();

        logOutput.writeSetupInfo(coreRadiusMsg);
        logOutput.writeSetupInfo(layerRadiusMsg);
        logOutput.writeSetupInfo(smoothingRegionThicknessMsg);
        logOutput.writeSetupInfo(pointChargeThicknessMsg);

        logOutput.writeEmptyLine();
    }

}   // namespace setup
