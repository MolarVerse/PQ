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

#include <csignal>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "enums/hessian.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"
#include "hessianBuilder.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"

#ifdef WITH_PYBIND11
#include <pybind11/embed.h>
#endif

namespace
{
    class HarmonicEvaluator : public opt::Evaluator
    {
       private:
        std::vector<ForceConstant> _forceConstants;
        bool                       _analyticSupported = false;

       public:
        explicit HarmonicEvaluator(
            const Settings                   &settings,
            const std::vector<ForceConstant> &forceConstants,
            const bool                        analyticSupported
        )
            : opt::Evaluator(settings),
              _forceConstants(forceConstants),
              _analyticSupported(analyticSupported)
        {
        }
        explicit HarmonicEvaluator(
            const Settings                   &settings,
            const std::vector<ForceConstant> &forceConstants
        )
            : HarmonicEvaluator(settings, forceConstants, false)
        {
        }

        [[nodiscard]]
        std::shared_ptr<Evaluator> clone() const override
        {
            return std::make_shared<HarmonicEvaluator>(*this);
        }

        void evaluate(physicalData::PhysicalData & /*physicalData*/) override
        {
            auto &simulationBox = _getSimulationBox();

            for (size_t atomIndex = 0;
                 atomIndex < simulationBox.getNumberOfAtoms();
                 ++atomIndex)
            {
                auto      &atom     = simulationBox.getAtom(atomIndex);
                const auto position = atom.getPosition();
                const auto offset   = 3 * atomIndex;
                atom.setForce(
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

        [[nodiscard]] Hessian calculateAnalyticHessian() override
        {
            return {
                {2.0 * kcal_per_mol / angstrom / angstrom,
                 4.0 * kcal_per_mol / angstrom / angstrom},
                {6.0 * kcal_per_mol / angstrom / angstrom,
                 8.0 * kcal_per_mol / angstrom / angstrom}
            };
        }
    };

    std::shared_ptr<molsys::SimulationBox> makeSimulationBox()
    {
        auto box = std::make_shared<molsys::SimulationBox>();

        auto atom0 = std::make_shared<molsys::Atom>();
        atom0->setPosition(linalg::Vec3D{1.0, -2.0, 3.0} * angstrom);

        auto atom1 = std::make_shared<molsys::Atom>();
        atom1->setPosition(linalg::Vec3D{-4.0, 5.0, -6.0} * angstrom);

        box->addAtom(atom0);
        box->addAtom(atom1);

        return box;
    }

    void expectDiagonalHessian(
        const Hessian                    &hessian,
        const std::vector<ForceConstant> &diagonal
    )
    {
        ASSERT_EQ(hessian.size(), diagonal.size());

        for (size_t row = 0; row < hessian.size(); ++row)
        {
            ASSERT_EQ(hessian[row].size(), diagonal.size());

            for (size_t col = 0; col < hessian[row].size(); ++col)
            {
                const auto expected =
                    row == col ? diagonal[row]
                               : 0.0 * kcal_per_mol / angstrom / angstrom;
                EXPECT_NEAR(
                    hessian[row][col].numerical_value_in(
                        kcal_per_mol / angstrom / angstrom
                    ),
                    expected.numerical_value_in(
                        kcal_per_mol / angstrom / angstrom
                    ),
                    1.0e-10
                );
            }
        }
    }

    void expectPositionsRestored(molsys::SimulationBox &simulationBox)
    {
        EXPECT_EQ(
            simulationBox.getAtom(0).getPosition(),
            linalg::Vec3D(1.0, -2.0, 3.0) * angstrom
        );
        EXPECT_EQ(
            simulationBox.getAtom(1).getPosition(),
            linalg::Vec3D(-4.0, 5.0, -6.0) * angstrom
        );
    }
}   // namespace

TEST(TestHessianBuilder, forceDifferenceBuildersRecoverHarmonicHessian)
{
    const auto diagonal = std::vector<ForceConstant>{
        1.0 * kcal_per_mol / angstrom / angstrom,
        2.0 * kcal_per_mol / angstrom / angstrom,
        3.0 * kcal_per_mol / angstrom / angstrom,
        4.0 * kcal_per_mol / angstrom / angstrom,
        5.0 * kcal_per_mol / angstrom / angstrom,
        6.0 * kcal_per_mol / angstrom / angstrom
    };

    const auto displacement = 1.0e-4 * angstrom;
    for (const auto &builder : {
             std::shared_ptr<opt::HessianBuilder>(
                 std::make_shared<opt::CentralForceDifferenceHessianBuilder>(
                     displacement
                 )
             ),
             std::shared_ptr<opt::HessianBuilder>(
                 std::make_shared<opt::ForwardForceDifferenceHessianBuilder>(
                     displacement
                 )
             ),
             std::shared_ptr<opt::HessianBuilder>(
                 std::make_shared<opt::FivePointForceDifferenceHessianBuilder>(
                     displacement
                 )
             ),
         })
    {
        auto              box = makeSimulationBox();
        Settings          settings;
        HarmonicEvaluator evaluator(settings, diagonal);
        evaluator.setSimulationBox(box);

        physicalData::PhysicalData physicalData;
        const auto hessian = builder->build(evaluator, *box, physicalData);

        expectDiagonalHessian(hessian, diagonal);
        expectPositionsRestored(*box);
        EXPECT_EQ(
            box->getAtom(0).getForce(),
            linalg::Vec3D(-1.0, 4.0, -9.0) * kcal_per_mol / angstrom
        );
        EXPECT_EQ(
            box->getAtom(1).getForce(),
            linalg::Vec3D(16.0, -25.0, 36.0) * kcal_per_mol / angstrom
        );
    }
}

#ifdef WITH_PYBIND11
TEST(TestHessianBuilder, pendingSigintStopsForceDifferenceBuilder)
{
    const pybind11::scoped_interpreter guard{};

    auto                       box = makeSimulationBox();
    Settings                   settings;
    physicalData::PhysicalData physicalData;
    HarmonicEvaluator          evaluator(
        settings,
        {1.0 * kcal_per_mol / angstrom / angstrom,
                  2.0 * kcal_per_mol / angstrom / angstrom,
                  3.0 * kcal_per_mol / angstrom / angstrom,
                  4.0 * kcal_per_mol / angstrom / angstrom,
                  5.0 * kcal_per_mol / angstrom / angstrom,
                  6.0 * kcal_per_mol / angstrom / angstrom}
    );
    evaluator.setSimulationBox(box);

    const auto builder =
        opt::CentralForceDifferenceHessianBuilder(1.0e-4 * angstrom);

    ASSERT_EQ(std::raise(SIGINT), 0);

    try
    {
        static_cast<void>(builder.build(evaluator, *box, physicalData));
        FAIL() << "pending SIGINT was not reported";
    }
    catch (const pybind11::error_already_set &error)
    {
        EXPECT_TRUE(error.matches(PyExc_KeyboardInterrupt));
    }

    expectPositionsRestored(*box);
}
#endif

TEST(TestHessianBuilder, analyticBuilderRequiresEvaluatorSupport)
{
    auto                       box = makeSimulationBox();
    Settings                   settings;
    physicalData::PhysicalData physicalData;
    HarmonicEvaluator          evaluator(
        settings,
        {1.0 * kcal_per_mol / angstrom / angstrom,
                  2.0 * kcal_per_mol / angstrom / angstrom,
                  3.0 * kcal_per_mol / angstrom / angstrom,
                  4.0 * kcal_per_mol / angstrom / angstrom,
                  5.0 * kcal_per_mol / angstrom / angstrom,
                  6.0 * kcal_per_mol / angstrom / angstrom}
    );
    evaluator.setSimulationBox(box);
    opt::AnalyticHessianBuilder builder;

    EXPECT_THROW_MSG(
        (void) builder.build(evaluator, *box, physicalData),
        exc::UserInputException,
        "The selected evaluator does not support analytic Hessians."
    );
}

TEST(TestHessianBuilder, analyticBuilderSymmetrizesEvaluatorHessian)
{
    auto                       box = makeSimulationBox();
    Settings                   settings;
    physicalData::PhysicalData physicalData;
    HarmonicEvaluator          evaluator(
        settings,
        {1.0 * kcal_per_mol / angstrom / angstrom,
                  2.0 * kcal_per_mol / angstrom / angstrom,
                  3.0 * kcal_per_mol / angstrom / angstrom,
                  4.0 * kcal_per_mol / angstrom / angstrom,
                  5.0 * kcal_per_mol / angstrom / angstrom,
                  6.0 * kcal_per_mol / angstrom / angstrom},
        true
    );
    evaluator.setSimulationBox(box);
    opt::AnalyticHessianBuilder builder;

    const auto hessian = builder.build(evaluator, *box, physicalData);

    ASSERT_EQ(hessian.size(), 2U);
    ASSERT_EQ(hessian[0].size(), 2U);
    EXPECT_DOUBLE_EQ(
        hessian[0][0].numerical_value_in(kcal_per_mol / angstrom / angstrom),
        2.0
    );
    EXPECT_DOUBLE_EQ(
        hessian[0][1].numerical_value_in(kcal_per_mol / angstrom / angstrom),
        5.0
    );
    EXPECT_DOUBLE_EQ(
        hessian[1][0].numerical_value_in(kcal_per_mol / angstrom / angstrom),
        5.0
    );
    EXPECT_DOUBLE_EQ(
        hessian[1][1].numerical_value_in(kcal_per_mol / angstrom / angstrom),
        8.0
    );
}

TEST(TestHessianBuilder, makeHessianBuilderSelectsConcreteStrategies)
{
    EXPECT_NE(
        std::dynamic_pointer_cast<opt::CentralForceDifferenceHessianBuilder>(
            opt::makeHessianBuilder(
                HessianBuilderType::CENTRAL,
                1.0e-3 * angstrom
            )
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<opt::ForwardForceDifferenceHessianBuilder>(
            opt::makeHessianBuilder(
                HessianBuilderType::FORWARD,
                1.0e-3 * angstrom
            )
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<opt::FivePointForceDifferenceHessianBuilder>(
            opt::makeHessianBuilder(
                HessianBuilderType::FIVE_POINT,
                1.0e-3 * angstrom
            )
        ),
        nullptr
    );
    EXPECT_NE(
        std::dynamic_pointer_cast<opt::AnalyticHessianBuilder>(
            opt::makeHessianBuilder(
                HessianBuilderType::ANALYTIC,
                1.0e-3 * angstrom
            )
        ),
        nullptr
    );
}
