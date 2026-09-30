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

#include "convergenceInputParser.hpp"

#include "convergenceSettings.hpp"
#include "enums/convergence.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"

namespace input
{

    /**
     * @brief Constructor
     *
     * @details following keywords are added:
     * - energy-conv-strategy "<string>"
     *
     * - use-energy-conv "<bool>"
     * - use-force-conv "<bool>"
     * - use-max-force-conv "<bool>"
     * - use-rms-force-conv "<bool>"
     *
     * - energy-conv "<double>"
     * - rel-energy-conv "<double>"
     * - abs-energy-conv "<double>"
     *
     * - force-conv "<double>"
     * - max-force-conv "<double>"
     * - rms-force-conv "<double>"
     *
     */
    ConvInputParser::ConvInputParser()
    {
        addEnergyConvergenceStrategyKey();
        addUseEnergyConvergenceKey();
        addUseForceConvergenceKey();
        addUseMaxForceConvergenceKey();
        addUseRMSForceConvergenceKey();
        addEnergyConvergenceKey();
        addRelativeEnergyConvergenceKey();
        addAbsoluteEnergyConvergenceKey();
        addForceConvergenceKey();
        addMaxForceConvergenceKey();
        addRMSForceConvergenceKey();
    }

    void ConvInputParser::addEnergyConvergenceStrategyKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "energy-conv-strategy",
            .title = "Energy Convergence Strategy",
            .description =
                "Specifies the strategy for energy convergence (rigorous, "
                "loose, absolute, relative)"
        };

        const auto setValue = [](ConvStrategy strategy)
        { settings::ConvSettings::setEnergyConvStrategy(strategy); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<ConvStrategy>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addUseEnergyConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "use-energy-conv",
            .title = "Use Energy Convergence",
            .description =
                "Specifies whether to use energy convergence (true or false)"
        };

        const auto setValue = [](bool useEnergyConv)
        { settings::ConvSettings::setUseEnergyConv(useEnergyConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addUseForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "use-force-conv",
            .title = "Use Force Convergence",
            .description =
                "Specifies whether to use force convergence (true or false)"
        };

        const auto setValue = [](bool useForceConv)
        { settings::ConvSettings::setUseForceConv(useForceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addUseMaxForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "use-max-force-conv",
            .title = "Use Max Force Convergence",
            .description =
                "Specifies whether to use max force convergence (true or false)"
        };

        const auto setValue = [](bool useMaxForceConv)
        { settings::ConvSettings::setUseMaxForceConv(useMaxForceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addUseRMSForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "use-rms-force-conv",
            .title = "Use RMS Force Convergence",
            .description =
                "Specifies whether to use RMS force convergence (true or false)"
        };

        const auto setValue = [](bool useRMSForceConv)
        { settings::ConvSettings::setUseRMSForceConv(useRMSForceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<bool>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addEnergyConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "energy-conv",
            .title = "Energy Convergence",
            .description =
                "Specifies the energy convergence threshold (must be greater "
                "than 0.0)"
        };

        const auto setValue = [](double energyConv)
        { settings::ConvSettings::setEnergyConv(energyConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addRelativeEnergyConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "rel-energy-conv",
            .title = "Relative Energy Convergence",
            .description =
                "Specifies the relative energy convergence threshold (must be "
                "greater than 0.0)"
        };

        const auto setValue = [](double relEnergyConv)
        { settings::ConvSettings::setRelEnergyConv(relEnergyConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addAbsoluteEnergyConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "abs-energy-conv",
            .title = "Absolute Energy Convergence",
            .description =
                "Specifies the absolute energy convergence threshold (must be "
                "greater than 0.0)"
        };

        const auto setValue = [](double absEnergyConv)
        { settings::ConvSettings::setAbsEnergyConv(absEnergyConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "force-conv",
            .title = "Force Convergence",
            .description =
                "Specifies the force convergence threshold (must be "
                "greater than 0.0)"
        };

        const auto setValue = [](double forceConv)
        { settings::ConvSettings::setForceConv(forceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addRMSForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "rms-force-conv",
            .title = "RMS Force Convergence",
            .description =
                "Specifies the RMS force convergence threshold (must be "
                "greater than 0.0)"
        };

        const auto setValue = [](double rmsForceConv)
        { settings::ConvSettings::setRMSForceConv(rmsForceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void ConvInputParser::addMaxForceConvergenceKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "max-force-conv",
            .title = "Max Force Convergence",
            .description =
                "Specifies the max force convergence threshold (must be "
                "greater than 0.0)"
        };

        const auto setValue = [](double maxForceConv)
        { settings::ConvSettings::setMaxForceConv(maxForceConv); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

}   // namespace input
