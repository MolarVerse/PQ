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
#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace opt
{
    class Evaluator;   // forward declaration

    using HessianMatrix = std::vector<std::vector<double>>;

    class HessianBuilder
    {
       public:
        HessianBuilder()          = default;
        virtual ~HessianBuilder() = default;

        [[nodiscard]]
        virtual HessianMatrix build(
            Evaluator             &evaluator,
            molsys::SimulationBox &simulationBox
        ) const = 0;
    };

    class ForceDifferenceHessianBuilder : public HessianBuilder
    {
       protected:
        double _displacement;

        static void restorePositions(
            molsys::SimulationBox            &simulationBox,
            const std::vector<linalg::Vec3D> &positions
        );

       public:
        explicit ForceDifferenceHessianBuilder(double displacement);

        static void symmetrize(HessianMatrix &hessian);
    };

    class CentralForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]]
        HessianMatrix build(
            Evaluator             &evaluator,
            molsys::SimulationBox &simulationBox
        ) const override;
    };

    class ForwardForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]]
        HessianMatrix build(
            Evaluator             &evaluator,
            molsys::SimulationBox &simulationBox
        ) const override;
    };

    class FivePointForceDifferenceHessianBuilder
        : public ForceDifferenceHessianBuilder
    {
       public:
        using ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder;

        [[nodiscard]]
        HessianMatrix build(
            Evaluator             &evaluator,
            molsys::SimulationBox &simulationBox
        ) const override;
    };

    class AnalyticHessianBuilder : public HessianBuilder
    {
       public:
        [[nodiscard]]
        HessianMatrix build(
            Evaluator             &evaluator,
            molsys::SimulationBox &simulationBox
        ) const override;
    };

    [[nodiscard]]
    std::shared_ptr<HessianBuilder> makeHessianBuilder(
        settings::HessianBuilderType builder,
        double                       displacement
    );

}   // namespace opt

#endif   // _HESSIAN_BUILDER_HPP_
