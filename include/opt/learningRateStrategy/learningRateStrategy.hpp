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

#ifndef _LEARNING_RATE_STRATEGY_HPP_

#define _LEARNING_RATE_STRATEGY_HPP_

#include <cstddef>    // for size_t
#include <memory>     // for shared_ptr
#include <optional>   // for optional
#include <string>     // for string
#include <vector>     // for vector

namespace engine
{
    class OptEngine;   // forward declaration

}   // namespace engine

namespace opt
{
    /**
     * @class LearningRateStrategy
     *
     * @brief Learning Rate Strategy
     *
     */
    class LearningRateStrategy
    {
       protected:
        size_t _frequency = 1;
        size_t _counter   = 0;
        double _initialLearningRate;
        double _learningRate;

        double                _minLearningRate;
        std::optional<double> _maxLearningRate;

        std::vector<std::string> _warningMessages;
        std::vector<std::string> _errorMessages;

       public:
        explicit LearningRateStrategy(const double);
        explicit LearningRateStrategy(const double, const size_t);

        LearningRateStrategy()          = default;
        virtual ~LearningRateStrategy() = default;

        virtual std::shared_ptr<LearningRateStrategy> clone() const = 0;

        virtual void updateLearningRate(const size_t step) = 0;
        void         checkLearningRate();

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] double                   getLearningRate() const;
        [[nodiscard]] std::vector<std::string> getWarningMessages() const;
        [[nodiscard]] std::vector<std::string> getErrorMessages() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setMinLearningRate(const double);
        void setMaxLearningRate(const std::optional<double>);
    };

}   // namespace opt

#endif   // _LEARNING_RATE_STRATEGY_HPP_