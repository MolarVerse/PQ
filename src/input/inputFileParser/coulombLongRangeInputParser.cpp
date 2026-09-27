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

#include "coulombLongRangeInputParser.hpp"

#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "potentialSettings.hpp"   // for settings::PotentialSettings
#include "rangeValidator.hpp"

namespace input
{

    /**
     * @brief Construct a new Input File Parser Coulomb Long Range:: Input File
     * Parser Coulomb Long Range object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) long_range "<string>" 2)
     * wolf_param "<double>"
     */
    CoulombLongRangeInputParser::CoulombLongRangeInputParser()
    {
        addCoulombLongRangeKey();
        addWolfParameterKey();
        addReactionFieldEpsilonKey();
    }

    /**
     * @brief Add the "long_range" key to the input file parser
     *
     */
    void CoulombLongRangeInputParser::addCoulombLongRangeKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "long_range",
            .title = "Coulomb long-range correction type",
            .description =
                "Specifies the type of Coulomb long-range correction to use in "
                "the simulation",
        };

        const auto setValue = [](settings::CoulombLongRangeType type)
        { settings::PotentialSettings::setCoulombLongRangeType(type); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<settings::CoulombLongRangeType>{
                .metadata     = metaData,
                .defaultValue = settings::CoulombLongRangeType::SHIFTED,
                .onSet        = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the "wolf_param" key to the input file parser
     *
     */
    void CoulombLongRangeInputParser::addWolfParameterKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "wolf_param",
            .title = "Wolf long-range correction parameter",
            .description =
                "Specifies the parameter for the Wolf long-range correction",
        };

        const auto setValue = [](double wolfParameter)
        { settings::PotentialSettings::setWolfParameter(wolfParameter); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata     = metaData,
                .defaultValue = defaults::WOLF_PARAM_DEFAULT,
                .onSet        = setValue,
                .validator    = makeShared(PositiveGTDoubleValidator),
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the "rf_epsilon" key to the input file parser
     *
     */
    void CoulombLongRangeInputParser::addReactionFieldEpsilonKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "rf_epsilon",
            .title = "Reaction field epsilon",
            .description =
                "Specifies the static relative permittivity for the reaction "
                "field correction",
        };

        const auto setValue = [](double epsilon)
        { settings::PotentialSettings::setReactionFieldEpsilon(epsilon); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata     = metaData,
                .defaultValue = defaults::RF_EPSILON_DEFAULT,
                .onSet        = setValue,
                .validator = makeShared(GEDoubleValidator{1.0, std::nullopt}),
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
