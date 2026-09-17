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

#include <cstddef>   // for size_t
#include <cstdint>
#include <optional>      // for optional
#include <string>        // for string
#include <string_view>   // for string_view

#include "defaults.hpp"   // for _OPTIMIZER_DEFAULT_

namespace settings
{
    /**
     * @brief enum OptimizerType
     *
     */
    enum class OptimizerType : std::uint8_t
    {
        NONE,
        STEEPEST_DESCENT,
        ADAM
    };

    /**
     * @brief enum LREnum
     *
     */
    enum class LREnum : std::uint8_t
    {
        NONE,
        CONSTANT,
        CONSTANT_DECAY,
        EXPONENTIAL_DECAY,
        LINESEARCH_WOLFE
    };

    std::string string(OptimizerType method);
    std::string string(LREnum method);

    /**
     * @brief OptimizerSettings
     *
     * @details stores all information about the optimizer
     *
     */
    class OptimizerSettings
    {
       private:
        // clang-format off
        static inline OptimizerType _optimizer = OptimizerType::STEEPEST_DESCENT;
        static inline LREnum _lRStrategy   = LREnum::EXPONENTIAL_DECAY;

        static inline size_t _nEpochs           = defaults::N_EPOCHS_DEFAULT;
        static inline size_t _lRupdateFrequency = defaults::LR_UPDATE_FREQUENCY_DEFAULT;

        static inline double _initialLearningRate = defaults::INITIAL_LEARNING_RATE_DEFAULT;
        static inline double _minLearningRate     = defaults::MIN_LEARNING_RATE_DEFAULT;
        // clang-format on

        static inline std::optional<double> _learningRateDecay;
        static inline std::optional<double> _maxLearningRate;

       public:
        /***************************
         * standard setter methods *
         ***************************/

        static void setOptimizer(const std::string_view &optimizer);
        static void setOptimizer(OptimizerType optimizer);

        static void setLearningRateStrategy(const std::string_view &);
        static void setLearningRateStrategy(LREnum);

        static void setNumberOfEpochs(size_t);
        static void setLRUpdateFrequency(size_t);

        static void setInitialLearningRate(double);
        static void setLearningRateDecay(double);

        static void setMaxLearningRate(double);
        static void setMinLearningRate(double);

        /******************************
         * validation helper methods *
         ******************************/

        static void validateLearningRateStrategy();
        static void validateLearningRateBounds();

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static OptimizerType getOptimizer();
        [[nodiscard]] static LREnum        getLearningRateStrategy();

        [[nodiscard]] static size_t getNumberOfEpochs();
        [[nodiscard]] static size_t getLRUpdateFrequency();

        [[nodiscard]] static double getInitialLearningRate();
        [[nodiscard]] static double getMinLearningRate();

        [[nodiscard]] static std::optional<double> getLearningRateDecay();
        [[nodiscard]] static std::optional<double> getMaxLearningRate();

    };   // namespace settings
}   // namespace settings

#endif   // _OPTIMIZER_SETTINGS_HPP_
