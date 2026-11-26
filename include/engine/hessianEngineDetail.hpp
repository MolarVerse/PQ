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

#ifndef ENGINE__HESSIAN_ENGINE_DETAIL_HPP
#define ENGINE__HESSIAN_ENGINE_DETAIL_HPP

#include <memory>

#include "convergenceSettings.hpp"
#include "hessianBuilder.hpp"
#include "hessianSettings.hpp"
#include "learningRateStrategy.hpp"
#include "optimizer.hpp"
#include "optimizerSettings.hpp"
#include "unit_types.hpp"

namespace engine::detail
{
    [[nodiscard]] std::shared_ptr<opt::HessianBuilder> setupHessianBuilder(
        const HessianSettings &hessianSettings
    );

    [[nodiscard]]
    std::shared_ptr<opt::LearningRateStrategy> setupLearningRateStrategy(
        const OptimizerSettings &settings
    );

    void setupMinMaxLearningRate(
        std::shared_ptr<opt::LearningRateStrategy> &learningRate,
        const OptimizerSettings                    &settings
    );

    void setupConvergence(
        std::shared_ptr<opt::Optimizer> &optimizer,
        const ConvSettings              &settings
    );

    void writeHessian(
        const Hessian         &hessian,
        const HessianSettings &hessianSettings
    );

    void writeHessianInfo(
        const Hessian         &hessian,
        const HessianSettings &hessianSettings
    );
}   // namespace engine::detail

#endif   // ENGINE__HESSIAN_ENGINE_DETAIL_HPP
