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

    SETTING_GETTER_AND_SETTER_BY_USER(_optimizer);
    SETTING_GETTER_AND_SETTER_BY_USER(_lRStrategy);
    SETTING_GETTER_AND_SETTER_BY_USER(_nEpochs);
    SETTING_GETTER_AND_SETTER_BY_USER(_lRupdateFrequency);
    SETTING_GETTER_AND_SETTER_BY_USER(_initialLearningRate);
    SETTING_GETTER_AND_SETTER_BY_USER(_minLearningRate);
    SETTING_GETTER_AND_SETTER_BY_USER(_learningRateDecay);
    SETTING_SETTER_BY_USER(_maxLearningRate);

    void validateLearningRateStrategy() const;
    void validateLearningRateBounds() const;

    [[nodiscard]] std::optional<double> getMaxLearningRate() const;

    [[nodiscard]] bool isLearningRateDecaySetByUser() const;
    [[nodiscard]] bool isMaxLearningRateSetByUser() const;
};

#endif   // SETTINGS__OPTIMIZER_SETTINGS_HPP
