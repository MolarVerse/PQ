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
#include <string>    // for string

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
    };

    /**
     * @class enum LearningRateStrategy
     *
     */
    enum class LearningRateStrategy : size_t
    {
        NONE,
        CONSTANT,
        DECAY,
    };

    std::string string(const Optimizer method);
    std::string string(const LearningRateStrategy method);

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
        static inline LearningRateStrategy _learningRateStrategy = LearningRateStrategy::NONE;

        static inline size_t _nEpochs = defaults::_N_EPOCHS_DEFAULT_;

        static inline double _initialLearningRate      = defaults::_INITIAL_LEARNING_RATE_DEFAULT_;
        static inline bool   _isInitialLearningRateSet = false;
        // clang-format on

       public:
        /***************************
         * standard setter methods *
         ***************************/

        static void setOptimizer(const std::string_view &optimizer);
        static void setOptimizer(const Optimizer optimizer);

        static void setLearningRateStrategy(const std::string_view &);
        static void setLearningRateStrategy(const LearningRateStrategy);

        static void setNumberOfEpochs(const size_t);

        static void setInitialLearningRate(const double);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static Optimizer            getOptimizer();
        [[nodiscard]] static LearningRateStrategy getLearningRateStrategy();
        [[nodiscard]] static size_t               getNumberOfEpochs();
        [[nodiscard]] static double               getInitialLearningRate();

    };   // namespace settings
}   // namespace settings

#endif   // _OPTIMIZER_SETTINGS_HPP_
