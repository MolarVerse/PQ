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

#include "hessianInputParser.hpp"

#include "hessianSettings.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"

namespace input
{

    /**
     * @brief Construct a new HessianInputParser object
     *
     */
    HessianInputParser::HessianInputParser()
    {
        addHessianFileKey();
        addHessianInfoFileKey();
        addDisplacementKey();
        addOptimizeBeforeHessianKey();
        addBuilderKey();
    }

    void HessianInputParser::addHessianFileKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "hessian_file",
            .title       = "Hessian file",
            .description = "The output Hessian file"
        };

        const auto setValue = [](const std::string &value)
        { settings::HessianSettings::setHessianFile(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HessianInputParser::addHessianInfoFileKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "hessian_info_file",
            .title       = "Hessian info file",
            .description = "The output Hessian info file"
        };

        const auto setValue = [](const std::string &value)
        { settings::HessianSettings::setHessianInfoFile(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<std::string>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HessianInputParser::addDisplacementKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "hessian_displacement",
            .title = "Hessian displacement",
            .description =
                "The displacement used for numerical Hessian calculations"
        };

        const auto setValue = [](double value)
        { settings::HessianSettings::setDisplacement(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HessianInputParser::addOptimizeBeforeHessianKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "optimize_before_hessian",
            .title = "Optimize before Hessian",
            .description =
                "Whether to optimize the structure before calculating the "
                "Hessian"
        };

        const auto setValue = [](bool value)
        { settings::HessianSettings::setOptimizeBeforeHessian(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HessianInputParser::addBuilderKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "hessian_builder",
            .title       = "Hessian builder",
            .description = "The type of Hessian builder to use"
        };

        const auto setValue = [](HessianBuilderType value)
        { settings::HessianSettings::setBuilder(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<HessianBuilderType>{
                .metadata = metaData,
                .onSet    = setValue
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
