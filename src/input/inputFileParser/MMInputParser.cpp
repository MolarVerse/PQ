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

#include "MMInputParser.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <utility>

#include "exceptions.hpp"        // for exc::InputFileException, customException
#include "forceFieldClass.hpp"   // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "forceFieldSettings.hpp"     // for settings::ForceFieldSettings
#include "parserUtils.hpp"
#include "potential.hpp"            // for Potential
#include "potentialSettings.hpp"    // for PotentialSettings
#include "stringUtilities.hpp"      // for utilities::toLowerCopy
#include "waterModelSettings.hpp"   // for settings::WaterModelSettings

namespace input
{

    /**
     * @brief Construct a new Input File Parser Force Field:: Input File Parser
     * Force Field object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) force-field
     * "<on/off/bonded>"
     *
     * @param forceField
     * @param potential
     */
    MMInputParser::MMInputParser(
        std::shared_ptr<ff::ForceField> forceField,
        std::shared_ptr<pot::Potential> potential
    )
        : _forceField(std::move(forceField)), _potential(std::move(potential))
    {
        addKeyword(
            std::string("force-field"),
            bindMember(&MMInputParser::parseForceFieldType, this),
            false
        );
        addKeyword(
            std::string("noncoulomb"),
            bindMember(&MMInputParser::parseNonCoulombType, this),
            false
        );
        addKeyword(
            std::string("water_intra"),
            bindMember(&MMInputParser::parseWaterIntraModel, this),
            false
        );
        addKeyword(
            std::string("water_inter"),
            bindMember(&MMInputParser::parseWaterInterModel, this),
            false
        );
    }

    /**
     * @brief Parse the force field type
     *
     * @details Possible options are:
     * 1) "on"  - force-field is activated
     * 2) "off" - force-field is deactivated (default)
     * 3) "bonded" - only bonded interactions are activated
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if force-field is not valid - currently
     * only on, off and bonded are supported
     */
    void MMInputParser::parseForceFieldType(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto forceFieldType = utilities::toLowerCopy(lineElements[2]);

        if (forceFieldType == "on")
        {
            settings::ForceFieldSettings::activate();
            _forceField->activateNonCoulombic();
            _potential->makeNonCoulombPotential(pot::ForceFieldNonCoulomb());
        }
        else if (forceFieldType == "off")
        {
            settings::ForceFieldSettings::deactivate();
            _forceField->deactivateNonCoulombic();
        }
        else if (forceFieldType == "bonded")
        {
            settings::ForceFieldSettings::activate();
            _forceField->deactivateNonCoulombic();
        }
        else
        {
            throw exc::InputFileException(format(
                "Invalid force-field keyword \"{}\" at line {} "
                "in input file\n"
                "Possible options are \"on\", \"off\" or \"bonded\"",
                lineElements[2],
                lineNumber
            ));
        }
    }

    /**
     * @brief Parse the nonCoulombic type of the guff.dat file
     *
     * @details Possible options are:
     * 1) "guff"  - guff.dat file is used (default)
     * 2) "lj"
     * 3) "buck"
     * 4) "morse"
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if invalid nonCoulomb type
     */
    void MMInputParser::parseNonCoulombType(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        checkCommand(lineElements, lineNumber);

        const auto type = utilities::toLowerCopy(lineElements[2]);

        using enum settings::NonCoulombType;

        if (type == "guff")
            settings::PotentialSettings::setNonCoulombType(GUFF);

        else if (type == "lj")
            settings::PotentialSettings::setNonCoulombType(LJ);

        else if (type == "buck")
            settings::PotentialSettings::setNonCoulombType(BUCKINGHAM);

        else if (type == "morse")
            settings::PotentialSettings::setNonCoulombType(MORSE);

        else
        {
            throw exc::InputFileException(format(
                "Invalid nonCoulomb type \"{}\" at line {} in input file.\n"
                "Possible options are: lj, buck, morse and guff",
                lineElements[2],
                lineNumber
            ));
        }
    }

    /**
     * @brief Parse the intramolecular water model type
     *
     * @details Possible options are:
     * 1) "SPC/Fw"
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if invalid water_intra model type
     */
    void MMInputParser::parseWaterIntraModel(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::WaterIntraModel;

        checkCommand(lineElements, lineNumber);

        const auto waterIntraModel =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if (waterIntraModel == "spc")
            settings::WaterModelSettings::setWaterIntraModel(SPC);
        else if (waterIntraModel == "spc_e")
            settings::WaterModelSettings::setWaterIntraModel(SPC_E);
        else if (waterIntraModel == "spc_fw")
            settings::WaterModelSettings::setWaterIntraModel(SPC_FW);
        else if (waterIntraModel == "qspc_fw")
            settings::WaterModelSettings::setWaterIntraModel(QSPC_FW);
        else if (waterIntraModel == "spc_dc")
        {
            settings::WaterModelSettings::setWaterIntraModel(SPC_DC);
        }
        else if (waterIntraModel == "h2o_dc")
        {
            settings::WaterModelSettings::setWaterIntraModel(H2O_DC);
        }
        else if (waterIntraModel == "tip3p")
        {
            settings::WaterModelSettings::setWaterIntraModel(TIP3P);
        }
        else if (waterIntraModel == "opc3")
        {
            settings::WaterModelSettings::setWaterIntraModel(OPC3);
        }
        else if (waterIntraModel == "spc_mtr")
        {
            settings::WaterModelSettings::setWaterIntraModel(SPC_MTR);
        }
        else if (waterIntraModel == "tip3p_mtr")
        {
            settings::WaterModelSettings::setWaterIntraModel(TIP3P_MTR);
        }
        else
        {
            throw exc::InputFileException(format(
                "Invalid water_intra keyword \"{}\" at line {} "
                "in input file\n"
                "Possible options are \"SPC\", \"SPC_E\", \"SPC_Fw\", "
                "\"qSPC_Fw\", "
                "\"SPC_DC\", \"H2O-DC\", \"TIP3P\", \"OPC3\", \"SPC-mTR\" and "
                "\"TIP3P-mTR\"",
                lineElements[2],
                lineNumber
            ));
        }

        settings::WaterModelSettings::setIsWaterModelSet(true);
    }

    /**
     * @brief Parse the intermolecular water model type
     *
     * @details Possible options are:
     * 1) "SPC/Fw"
     *
     * @param lineElements
     * @param lineNumber
     *
     * @throws exc::InputFileException if invalid water_inter model type
     */
    void MMInputParser::parseWaterInterModel(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        using enum settings::WaterInterModel;

        checkCommand(lineElements, lineNumber);

        const auto waterInterModel =
            utilities::toLowerAndReplaceDashesCopy(lineElements[2]);

        if (waterInterModel == "spc")
            settings::WaterModelSettings::setWaterInterModel(SPC);
        else if (waterInterModel == "spc_e")
            settings::WaterModelSettings::setWaterInterModel(SPC_E);
        else if (waterInterModel == "spc_fw")
            settings::WaterModelSettings::setWaterInterModel(SPC_FW);
        else if (waterInterModel == "qspc_fw")
            settings::WaterModelSettings::setWaterInterModel(QSPC_FW);
        else if (waterInterModel == "spc_dc")
            settings::WaterModelSettings::setWaterInterModel(SPC_DC);
        else if (waterInterModel == "h2o_dc")
            settings::WaterModelSettings::setWaterInterModel(H2O_DC);
        else if (waterInterModel == "tip3p")
            settings::WaterModelSettings::setWaterInterModel(TIP3P);
        else if (waterInterModel == "opc3")
            settings::WaterModelSettings::setWaterInterModel(OPC3);
        else if (waterInterModel == "spc_mtr")
            settings::WaterModelSettings::setWaterInterModel(SPC_MTR);
        else if (waterInterModel == "tip3p_mtr")
            settings::WaterModelSettings::setWaterInterModel(TIP3P_MTR);
        else
        {
            throw exc::InputFileException(format(
                "Invalid water_inter keyword \"{}\" at line {} "
                "in input file\n"
                "Possible options are \"SPC\", \"SPC_E\", \"SPC_Fw\", "
                "\"qSPC_Fw\", "
                "\"SPC-DC\", \"H2O-DC\", \"TIP3P\", \"OPC3\", \"SPC-mTR\" and "
                "\"TIP3P-mTR\"",
                lineElements[2],
                lineNumber
            ));
        }

        settings::WaterModelSettings::setIsWaterModelSet(true);
        settings::WaterModelSettings::setIsInterWaterModelSet(true);
    }

}   // namespace input
