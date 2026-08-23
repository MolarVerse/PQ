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

#include <gtest/gtest.h>

#include <memory>
#include <utility>
#include <vector>

#include "atom.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"
#include "hessianBuilder.hpp"
#include "simulationBox.hpp"

using namespace opt;
using molsys::Atom;
using molsys::SimulationBox;

namespace
{
    class HarmonicEvaluator : public Evaluator
    {
       private:
        std::vector<double> _forceConstants;
        bool                _analyticSupported = false;

       public:
        explicit HarmonicEvaluator(
            std::vector<double> forceConstants,
            const bool          analyticSupported
        )
            : _forceConstants(std::move(forceConstants)),
              _analyticSupported(analyticSupported)
        {
        }
        explicit HarmonicEvaluator(std::vector<double> forceConstants)
            : HarmonicEvaluator(std::move(forceConstants), false)
        {
        }

        [[nodiscard]]
        std::shared_ptr<Evaluator> clone() const override
        {
            return std::make_shared<HarmonicEvaluator>(*this);
        }

        void evaluate() override
        {
            for (size_t atomIndex = 0;
                 atomIndex < _simulationBox->getNumberOfAtoms();
                 ++atomIndex)
            {
                const auto position =
                    _simulationBox->getAtom(atomIndex).getPosition();
                const auto offset = 3 * atomIndex;
                _simulationBox->getAtom(atomIndex).setForce(
                    {-_forceConstants[offset] * position[0],
                     -_forceConstants[offset + 1] * position[1],
                     -_forceConstants[offset + 2] * position[2]}
                );
            }
        }

        [[nodiscard]] bool supportsAnalyticHessian() const override
        {
            return _analyticSupported;
        }

        [[nodiscard]] HessianMatrix calculateAnalyticHessian() override
        {
            return {{2.0, 4.0}, {6.0, 8.0}};
        }
    };

    std::shared_ptr<SimulationBox> makeSimulationBox()
    {
        auto box = std::make_shared<SimulationBox>();

        auto atom0 = std::make_shared<Atom>();
        atom0->setPosition({1.0, -2.0, 3.0});

        auto atom1 = std::make_shared<Atom>();
        atom1->setPosition({-4.0, 5.0, -6.0});

        box->addAtom(atom0);
        box->addAtom(atom1);

        return box;
    }

    void expectDiagonalHessian(
        const HessianMatrix       &hessian,
        const std::vector<double> &diagonal
    )
    {
        ASSERT_EQ(hessian.size(), diagonal.size());

        for (size_t row = 0; row < hessian.size(); ++row)
        {
            ASSERT_EQ(hessian[row].size(), diagonal.size());

            for (size_t col = 0; col < hessian[row].size(); ++col)
            {
                const auto expected = row == col ? diagonal[row] : 0.0;
                EXPECT_NEAR(hessian[row][col], expected, 1.0e-10);
            }
        }
    }

    void expectPositionsRestored(SimulationBox &box)
    {
        EXPECT_EQ(
            box.getAtom(0).getPosition(),
            linearAlgebra::Vec3D(1.0, -2.0, 3.0)
        );
        EXPECT_EQ(
            box.getAtom(1).getPosition(),
            linearAlgebra::Vec3D(-4.0, 5.0, -6.0)
        );
    }
}   // namespace

TEST(TestHessianBuilder, forceDifferenceBuildersRecoverHarmonicHessian)
{
    const auto diagonal = std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    for (const auto &builder : {
             std::shared_ptr<HessianBuilder>(
                 std::make_shared<CentralForceDifferenceHessianBuilder>(1.0e-4)
             ),
             std::shared_ptr<HessianBuilder>(
                 std::make_shared<ForwardForceDifferenceHessianBuilder>(1.0e-4)
             ),
             std::shared_ptr<HessianBuilder>(
                 std::make_shared<FivePointForceDifferenceHessianBuilder>(1.0e-4
                 )
             ),
         })
    {
        auto              box = makeSimulationBox();
        HarmonicEvaluator evaluator(diagonal);
        evaluator.setSimulationBox(box);

        const auto hessian = builder->build(evaluator, *box);

        expectDiagonalHessian(hessian, diagonal);
        expectPositionsRestored(*box);
        EXPECT_EQ(
            box->getAtom(0).getForce(),
            linearAlgebra::Vec3D(-1.0, 4.0, -9.0)
        );
        EXPECT_EQ(
            box->getAtom(1).getForce(),
            linearAlgebra::Vec3D(16.0, -25.0, 36.0)
        );
    }
}

TEST(TestHessianBuilder, analyticBuilderRequiresEvaluatorSupport)
{
    auto              box = makeSimulationBox();
    HarmonicEvaluator evaluator({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    evaluator.setSimulationBox(box);
    AnalyticHessianBuilder builder;

    EXPECT_THROW(
        (void) builder.build(evaluator, *box),
        customException::UserInputException
    );
}

TEST(TestHessianBuilder, analyticBuilderSymmetrizesEvaluatorHessian)
{
    auto              box = makeSimulationBox();
    HarmonicEvaluator evaluator({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, true);
    evaluator.setSimulationBox(box);
    AnalyticHessianBuilder builder;

    const auto hessian = builder.build(evaluator, *box);

    ASSERT_EQ(hessian.size(), 2U);
    ASSERT_EQ(hessian[0].size(), 2U);
    EXPECT_DOUBLE_EQ(hessian[0][0], 2.0);
    EXPECT_DOUBLE_EQ(hessian[0][1], 5.0);
    EXPECT_DOUBLE_EQ(hessian[1][0], 5.0);
    EXPECT_DOUBLE_EQ(hessian[1][1], 8.0);
}

TEST(TestHessianBuilder, makeHessianBuilderSelectsConcreteStrategies)
{
    using enum settings::HessianBuilderType;

    EXPECT_NE(
        std::dynamic_pointer_cast<CentralForceDifferenceHessianBuilder>(
            makeHessianBuilder(FINITE_DIFFERENCE_FORCES_CENTRAL, 1.0e-3)
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<ForwardForceDifferenceHessianBuilder>(
            makeHessianBuilder(FINITE_DIFFERENCE_FORCES_FORWARD, 1.0e-3)
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<FivePointForceDifferenceHessianBuilder>(
            makeHessianBuilder(FINITE_DIFFERENCE_FORCES_FIVE_POINT, 1.0e-3)
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<AnalyticHessianBuilder>(
            makeHessianBuilder(ANALYTIC, 1.0e-3)
        ),
        nullptr
    );

    EXPECT_THROW(
        (void) makeHessianBuilder(settings::HessianBuilderType::NONE, 1.0e-3),
        customException::UserInputException
    );
}
