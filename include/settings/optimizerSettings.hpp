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

#include <cstddef>       // for size_t
#include <optional>      // for optional
#include <string>        // for string
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _OPTIMIZER_DEFAULT_

namespace settings
{
    /**
     * @class enum Optimizer
     *
     */
    enum class Optimizer : size_t
    {
        NONE,
        STEEPEST_DESCENT,
        ADAM
    };

    /**
     * @class enum LREnum
     *
     */
    enum class LREnum : size_t
    {
        NONE,
        CONSTANT,
        CONSTANT_DECAY,
        EXPONENTIAL_DECAY,
        LINESEARCH_WOLFE
    };

    std::string string(const Optimizer method);
    std::string string(const LREnum method);

    /**
     * @class OptimizerSettings
     *
     * @brief stores all information about the optimizer
     *
     */
    class OptimizerSettings
    {
       private:
        // clang-format off
        static inline Optimizer _optimizer = Optimizer::STEEPEST_DESCENT;
        static inline LREnum _LRStrategy   = LREnum::CONSTANT_DECAY;

        static inline size_t _nEpochs           = defaults::_N_EPOCHS_DEFAULT_;
        static inline size_t _LRupdateFrequency = defaults::_LR_UPDATE_FREQUENCY_DEFAULT_;

        static inline double _initialLearningRate = defaults::_INITIAL_LEARNING_RATE_DEFAULT_;
        static inline double _minLearningRate     = defaults::_MIN_LEARNING_RATE_DEFAULT_;
        // clang-format on

        static inline std::optional<double> _learningRateDecay;
        static inline std::optional<double> _maxLearningRate;

       public:
        /***************************
         * standard setter methods *
         ***************************/

        static void setOptimizer(const std::string_view &optimizer);
        static void setOptimizer(const Optimizer optimizer);

        static void setLearningRateStrategy(const std::string_view &);
        static void setLearningRateStrategy(const LREnum);

        static void setNumberOfEpochs(const size_t);
        static void setLRUpdateFrequency(const size_t);

        static void setInitialLearningRate(const double);
        static void setLearningRateDecay(const double);

        static void setMaxLearningRate(const double);
        static void setMinLearningRate(const double);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static Optimizer getOptimizer();
        [[nodiscard]] static LREnum    getLearningRateStrategy();

        [[nodiscard]] static size_t getNumberOfEpochs();
        [[nodiscard]] static size_t getLRUpdateFrequency();

        [[nodiscard]] static double getInitialLearningRate();
        [[nodiscard]] static double getMinLearningRate();

        [[nodiscard]] static std::optional<double> getLearningRateDecay();
        [[nodiscard]] static std::optional<double> getMaxLearningRate();

    };   // namespace settings
}   // namespace settings

#endif   // _OPTIMIZER_SETTINGS_HPP_
