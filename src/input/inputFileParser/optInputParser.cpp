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

#include "optInputParser.hpp"

#include "enums/optimizer.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "keyRegistry.hpp"
#include "keyValidatorBase.hpp"
#include "optimizerSettings.hpp"
#include "rangeValidator.hpp"

namespace input
{

    /**
     * @brief Constructor
     *
     * @details following keywords are added:
     * - optimizer "<string>"
     * - n-iterations "<int>"
     * - learning-rate-strategy "<string>"
     * - initial-learning-rate "<double>"
     */
    OptInputParser::OptInputParser()
    {
        addOptimizerKey();
        addLearningRateStrategyKey();
        addInitialLearningRateKey();
        addLearningRateUpdateFreqKey();
        addMinLearningRateKey();
        addMaxLearningRateKey();
        addLearningRateDecayKey();
    }

    /**
     * @brief Adds the optimizer key to the input parser.
     *
     */
    void OptInputParser::addOptimizerKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "optimizer",
            .title = "Optimizer Type",
            .description =
                "Specifies the optimizer to be used for the optimization "
                "process.",
        };

        const auto setValue = [](OptimizerType value)
        { settings::OptimizerSettings::setOptimizer(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<OptimizerType>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the learning rate strategy key to the input parser.
     *
     */
    void OptInputParser::addLearningRateStrategyKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "learning-rate-strategy",
            .title = "Learning Rate Strategy",
            .description =
                "Specifies the strategy for updating the learning rate during "
                "the optimization process."
        };

        const auto setValue = [](LearningRate value)
        { settings::OptimizerSettings::setLearningRateStrategy(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<LearningRate>{.metadata = metaData, .onSet = setValue}
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the initial learning rate key to the input parser.
     *
     */
    void OptInputParser::addInitialLearningRateKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "initial-learning-rate",
            .title = "Initial Learning Rate",
            .description =
                "Specifies the initial learning rate for the optimization "
                "process."
        };

        const auto setValue = [](double value)
        { settings::OptimizerSettings::setInitialLearningRate(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the learning rate update frequency key to the input parser.
     *
     */
    void OptInputParser::addLearningRateUpdateFreqKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "learning-rate-update-freq",
            .title = "Learning Rate Update Frequency",
            .description =
                "Specifies how frequently the learning rate should be "
                "updated during the optimization process."
        };

        const auto setValue = [](size_t value)
        { settings::OptimizerSettings::setLRUpdateFrequency(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<size_t>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(SizeTValidatorExcludingZero)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the minimum learning rate key to the input parser.
     *
     */
    void OptInputParser::addMinLearningRateKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "min-learning-rate",
            .title = "Minimum Learning Rate",
            .description =
                "Specifies the minimum learning rate for the optimization "
                "process."
        };

        const auto setValue = [](double value)
        { settings::OptimizerSettings::setMinLearningRate(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the maximum learning rate key to the input parser.
     *
     */
    void OptInputParser::addMaxLearningRateKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "max-learning-rate",
            .title = "Maximum Learning Rate",
            .description =
                "Specifies the maximum learning rate for the optimization "
                "process."
        };

        const auto setValue = [](double value)
        { settings::OptimizerSettings::setMaxLearningRate(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the learning rate decay key to the input parser.
     *
     */
    void OptInputParser::addLearningRateDecayKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "learning-rate-decay",
            .title = "Learning Rate Decay",
            .description =
                "Specifies the learning rate decay for the optimization "
                "process."
        };

        const auto setValue = [](double value)
        { settings::OptimizerSettings::setLearningRateDecay(value); };

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
