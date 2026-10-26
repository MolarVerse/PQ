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

#ifndef _OPTIMIZER_SETTINGS_HPP_

#define _OPTIMIZER_SETTINGS_HPP_

#include <cstddef>

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
    Setting<OptimizerType> _optimizer;
    Setting<LearningRate>  _lRStrategy;

    Setting<size_t> _nEpochs;
    Setting<size_t> _lRupdateFrequency;

    Setting<double> _initialLearningRate;
    Setting<double> _minLearningRate;

    Setting<double> _learningRateDecay;
    Setting<double> _maxLearningRate;

   public:
    OptimizerSettings();

    /***************************
     * standard setter methods *
     ***************************/

    void setOptimizer(OptimizerType optimizer);
    void setLearningRateStrategy(LearningRate);

    void setNumberOfEpochs(size_t);
    void setLRUpdateFrequency(size_t);

    void setInitialLearningRate(double);
    void setLearningRateDecay(double);

    void setMaxLearningRate(double);
    void setMinLearningRate(double);

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

    [[nodiscard]] double getLearningRateDecay() const;
    [[nodiscard]] double getMaxLearningRate() const;

    [[nodiscard]] bool isLearningRateDecaySet() const;
    [[nodiscard]] bool isMaxLearningRateSet() const;
};

#endif   // _OPTIMIZER_SETTINGS_HPP_
