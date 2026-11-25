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

#ifndef SETTINGS__OPTIMIZER_SETTINGS_HPP
#define SETTINGS__OPTIMIZER_SETTINGS_HPP

#include <cstddef>
#include <optional>

#include "enums/optimizer.hpp"
#include "setting.hpp"

/**
 * @brief OptimizerSettings
 *
 * @details stores all information about the optimizer
 *
 */
class OptimizerSettings
{
   private:
    Setting<OptimizerType, "Optimizer Type">        _optimizer;
    Setting<LearningRate, "Learning Rate Strategy"> _lRStrategy;

    Setting<size_t, "Number of Epochs">               _nEpochs;
    Setting<size_t, "Learning Rate Update Frequency"> _lRupdateFrequency;

    Setting<double, "Initial Learning Rate"> _initialLearningRate;
    Setting<double, "Minimum Learning Rate"> _minLearningRate;

    Setting<double, "Learning Rate Decay">   _learningRateDecay;
    Setting<double, "Maximum Learning Rate"> _maxLearningRate;

   public:
    OptimizerSettings();

    /***************************
     * standard setter methods *
     ***************************/

    void setOptimizerByUser(OptimizerType optimizer);
    void setLearningRateStrategyByUser(LearningRate);

    void setNumberOfEpochsByUser(size_t);
    void setLRUpdateFrequencyByUser(size_t);

    void setInitialLearningRateByUser(double);
    void setLearningRateDecayByUser(double);

    void setMaxLearningRateByUser(double);
    void setMinLearningRateByUser(double);

    /******************************
     * validation helper methods *
     ******************************/

    void validateLearningRateStrategy() const;
    void validateLearningRateBounds() const;

    /***************************
     * standard getter methods *
     ***************************/

    [[nodiscard]] OptimizerType getOptimizer() const;
    [[nodiscard]] LearningRate  getLearningRateStrategy() const;

    [[nodiscard]] size_t getNumberOfEpochs() const;
    [[nodiscard]] size_t getLRUpdateFrequency() const;

    [[nodiscard]] double getInitialLearningRate() const;
    [[nodiscard]] double getMinLearningRate() const;

    [[nodiscard]] double                getLearningRateDecay() const;
    [[nodiscard]] std::optional<double> getMaxLearningRate() const;

    [[nodiscard]] bool isLearningRateDecaySetByUser() const;
    [[nodiscard]] bool isMaxLearningRateSetByUser() const;
};

#endif   // SETTINGS__OPTIMIZER_SETTINGS_HPP
