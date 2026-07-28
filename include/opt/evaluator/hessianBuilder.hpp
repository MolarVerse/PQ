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

#ifndef _HESSIAN_BUILDER_HPP_

#define _HESSIAN_BUILDER_HPP_

#include <memory>
#include <vector>

#include "hessianSettings.hpp"
#include "typeAliases.hpp"

namespace opt
{
    class HessianBuilder
    {
       public:
        HessianBuilder()          = default;
        virtual ~HessianBuilder() = default;

        [[nodiscard]] virtual pq::HessianMatrix build(
            Evaluator          &evaluator,
            pq::SimBox         &simulationBox
        ) const = 0;
    };

    class ForceDifferenceHessianBuilder : public HessianBuilder
    {
       protected:
        double _displacement;

        [[nodiscard]] std::vector<double> evaluateForces(
            Evaluator         &evaluator,
            pq::SimBox        &simulationBox,
            const size_t       coordinateIndex,
            const double       displacement
        ) const;

        static void restorePositions(
            pq::SimBox                  &simulationBox,
            const std::vector<pq::Vec3D> &positions
        );

        static void displaceCoordinate(
            pq::SimBox        &simulationBox,
            const size_t       coordinateIndex,
            const double       displacement
        );

        [[nodiscard]] static std::vector<double> flattenForces(
            const pq::SimBox &simulationBox
        );

       public:
        explicit ForceDifferenceHessianBuilder(const double displacement);

        static void symmetrize(pq::HessianMatrix &hessian);
    };

    class CentralForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]] pq::HessianMatrix build(
            Evaluator          &evaluator,
            pq::SimBox         &simulationBox
        ) const override;
    };

    class ForwardForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]] pq::HessianMatrix build(
            Evaluator          &evaluator,
            pq::SimBox         &simulationBox
        ) const override;
    };

    class FivePointForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]] pq::HessianMatrix build(
            Evaluator          &evaluator,
            pq::SimBox         &simulationBox
        ) const override;
    };

    class AnalyticHessianBuilder : public HessianBuilder
    {
       public:
        [[nodiscard]] pq::HessianMatrix build(
            Evaluator          &evaluator,
            pq::SimBox         &simulationBox
        ) const override;
    };

    [[nodiscard]] std::shared_ptr<HessianBuilder> makeHessianBuilder(
        const settings::HessianBuilderType builder,
        const double                       displacement
    );

}   // namespace opt

#endif   // _HESSIAN_BUILDER_HPP_
