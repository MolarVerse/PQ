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

#include "optimizerSettings.hpp"

#include <format>

#include "exceptions.hpp"

namespace settings
{

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief sets the optimizer to enum in settings
     *
     * @param optimizer
     */
    void OptimizerSettings::setOptimizer(OptimizerType optimizer)
    {
        _optimizer = optimizer;
    }

    /**
     * @brief sets the optimizer to enum in settings
     *
     * @param method
     */
    void OptimizerSettings::setLearningRateStrategy(LearningRate method)
    {
        _lRStrategy = method;
    }

    /**
     * @brief sets the number of epochs
     *
     * @param nEpochs
     */
    void OptimizerSettings::setNumberOfEpochs(size_t nEpochs)
    {
        _nEpochs = nEpochs;
    }

    /**
     * @brief sets the learning rate update frequency
     *
     * @param frequency
     */
    void OptimizerSettings::setLRUpdateFrequency(size_t frequency)
    {
        _lRupdateFrequency = frequency;
    }

    /**
     * @brief sets the initial learning rate
     *
     * @param learningRate
     */
    void OptimizerSettings::setInitialLearningRate(double learningRate)
    {
        _initialLearningRate = learningRate;
    }

    /**
     * @brief sets the learning rate decay
     *
     * @param decay
     */
    void OptimizerSettings::setLearningRateDecay(double decay)
    {
        _learningRateDecay = decay;
    }

    /**
     * @brief sets the min learning rate
     *
     * @param minLearningRate
     */
    void OptimizerSettings::setMinLearningRate(double minLearningRate)
    {
        _minLearningRate = minLearningRate;
    }

    /**
     * @brief sets the max learning rate
     *
     * @param maxLearningRate
     */
    void OptimizerSettings::setMaxLearningRate(double maxLearningRate)
    {
        _maxLearningRate = maxLearningRate;
    }

    /*****************************
     *                           *
     * validation helper methods *
     *                           *
     *****************************/

    /**
     * @brief validates the selected learning-rate strategy
     */
    void OptimizerSettings::validateLearningRateStrategy()
    {
        const auto strategy = getLearningRateStrategy();

        if (strategy == LearningRate::LINESEARCH_WOLFE)
        {
            throw exc::UserInputException(
                "The Wolfe line search learning rate strategy is not yet "
                "implemented"
            );
        }

        if (strategy == LearningRate::NONE)
        {
            throw exc::UserInputException(
                "In order to run the optimizer, you need to specify a learning "
                "rate strategy."
            );
        }

        const auto needsDecay = strategy == LearningRate::CONSTANT_DECAY ||
                                strategy == LearningRate::EXPONENTIAL_DECAY;

        if (needsDecay && !getLearningRateDecay().has_value())
        {
            throw exc::UserInputException(
                std::format(
                    "The {} learning rate strategy requires "
                    "learning-rate-decay.",
                    strategy == LearningRate::CONSTANT_DECAY
                        ? "constant-decay"
                        : "exponential-decay"
                )
            );
        }
    }

    /**
     * @brief validates the configured learning-rate bounds
     */
    void OptimizerSettings::validateLearningRateBounds()
    {
        const auto minLR = getMinLearningRate();
        const auto maxLR = getMaxLearningRate();

        if (maxLR.has_value() && minLR >= maxLR.value())
        {
            throw exc::UserInputException(
                std::format(
                    "The minimum learning rate {} is greater or equal to the "
                    "maximum learning rate {}, which is not allowed.",
                    minLR,
                    maxLR.value()
                )
            );
        }
    }

    /***************************
     *                         *
     * standard getter methods *
     *                         *
     ***************************/

    /**
     * @brief returns the optimizer as string
     *
     * @return OptimizerType
     */
    OptimizerType OptimizerSettings::getOptimizer() { return _optimizer; }

    /**
     * @brief returns the learning rate strategy as string
     *
     * @return LearningRateStrategy
     */
    LearningRate OptimizerSettings::getLearningRateStrategy()
    {
        return _lRStrategy;
    }

    /**
     * @brief returns the number of epochs
     *
     * @return size_t
     */
    size_t OptimizerSettings::getNumberOfEpochs() { return _nEpochs; }

    /**
     * @brief returns the learning rate update frequency
     *
     * @return size_t
     */
    size_t OptimizerSettings::getLRUpdateFrequency()
    {
        return _lRupdateFrequency;
    }

    /**
     * @brief returns the initial learning rate
     *
     * @return double
     */
    double OptimizerSettings::getInitialLearningRate()
    {
        return _initialLearningRate;
    }

    /**
     * @brief returns the min learning rate
     *
     * @return double
     */
    double OptimizerSettings::getMinLearningRate() { return _minLearningRate; }

    /**
     * @brief returns the learning rate decay
     *
     * @return std::optional<double>
     */
    std::optional<double> OptimizerSettings::getLearningRateDecay()
    {
        return _learningRateDecay;
    }

    /**
     * @brief returns the max learning rate
     *
     * @return std::optional<double>
     */
    std::optional<double> OptimizerSettings::getMaxLearningRate()
    {
        return _maxLearningRate;
    }

}   // namespace settings
