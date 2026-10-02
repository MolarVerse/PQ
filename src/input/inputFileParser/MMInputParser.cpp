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

#include <utility>

#include "forceFieldClass.hpp"        // for ForceField
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "forceFieldSettings.hpp"     // for settings::ForceFieldSettings
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "potential.hpp"            // for Potential
#include "potentialSettings.hpp"    // for PotentialSettings
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
        addForceFieldTypeKey();
        addNonCoulombTypeKey();
        addWaterIntraModelKey();
        addWaterInterModelKey();
    }

    /**
     * @brief Add the force field type key to the registry
     *
     * @details This function registers the "force-field" key with the input
     * key registry and associates it with the appropriate metadata and
     * callback function.
     */
    void MMInputParser::addForceFieldTypeKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "force-field",
            .title = "Force Field Type",
            .description =
                "Specifies the type of force field to be used (on, off, "
                "bonded)",
        };

        const auto setValue = [forceField = _forceField,
                               potential  = _potential](ForceFieldType value)
        {
            switch (value)
            {
                case ForceFieldType::ON:
                    settings::ForceFieldSettings::activate();
                    forceField->activateNonCoulombic();
                    potential->makeNonCoulombPotential(
                        pot::ForceFieldNonCoulomb()
                    );
                    break;
                case ForceFieldType::OFF:
                    settings::ForceFieldSettings::deactivate();
                    forceField->deactivateNonCoulombic();
                    break;
                case ForceFieldType::BONDED:
                    settings::ForceFieldSettings::activate();
                    forceField->deactivateNonCoulombic();
                    break;
            }
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<ForceFieldType>{
                .metadata     = metaData,
                .defaultValue = ForceFieldType::OFF,
                .onSet        = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the key for specifying the non-Coulombic interaction type
     */
    void MMInputParser::addNonCoulombTypeKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "noncoulomb",
            .title = "Non-Coulomb Type",
            .description =
                "Specifies the type of non-Coulombic interaction to be used "
                "(guff, lj, buck, morse)",
        };

        const auto setValue = [potential = _potential](NonCoulombType value)
        { settings::PotentialSettings::setNonCoulombType(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<NonCoulombType>{
                .metadata   = metaData,
                .notAllowed = {NonCoulombType::LJ_9_12},
                .onSet      = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the key for specifying the intermolecular water model
     */
    void MMInputParser::addWaterIntraModelKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "water_intra",
            .title = "Intramolecular Water Model",
            .description =
                "Specifies the intramolecular water model to be used "
                "(spc, spc_e, spc_fw, qspc_fw, spc_dc, h2o_dc, tip3p, opc3, "
                "spc_mtr, tip3p_mtr)",
        };

        const auto setValue = [potential = _potential](WaterIntraModel value)
        {
            settings::WaterModelSettings::setWaterIntraModel(value);
            settings::WaterModelSettings::setIsWaterModelSet(true);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<WaterIntraModel>{
                .metadata = metaData,
                .onSet    = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the key for specifying the intermolecular water model
     */
    void MMInputParser::addWaterInterModelKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "water_inter",
            .title = "Intermolecular Water Model",
            .description =
                "Specifies the intermolecular water model to be used "
                "(spc, spc_e, spc_fw, qspc_fw, spc_dc, h2o_dc, tip3p, opc3, "
                "spc_mtr, tip3p_mtr)",
        };

        const auto setValue = [potential = _potential](WaterInterModel value)
        {
            settings::WaterModelSettings::setWaterInterModel(value);
            settings::WaterModelSettings::setIsWaterModelSet(true);
            settings::WaterModelSettings::setIsInterWaterModelSet(true);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<WaterInterModel>{
                .metadata = metaData,
                .onSet    = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
