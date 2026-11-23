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

#include <gmock/gmock.h>

#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include "atom.hpp"
#include "bondConstraint.hpp"
#include "constraintSettings.hpp"
#include "constraints.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"
#include "molecule.hpp"
#include "optEngine.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "settings.hpp"
#include "simulationBox.hpp"
#include "testSettings.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace
{
    class ConstantForceEvaluator : public opt::Evaluator
    {
       public:
        using opt::Evaluator::Evaluator;

        [[nodiscard]] std::shared_ptr<Evaluator> clone() const override;
        void evaluate(physicalData::PhysicalData &physicalData) override;
    };

    std::shared_ptr<opt::Evaluator> ConstantForceEvaluator::clone() const
    {
        return std::make_shared<ConstantForceEvaluator>(*this);
    }

    void ConstantForceEvaluator::evaluate(
        physicalData::PhysicalData &physicalData
    )
    {
        auto &atoms = _getSimulationBox().getAtoms();
        atoms[0]->setForce(linalg::Vec3D(4.0, 2.0, 0.0) * kcal_per_mol / angstrom);
        atoms[1]->setForce(Force3D::zero());
        physicalData.addBondEnergy(1.0 * kcal_per_mol);
    }

    // two atoms (masses 1 and 3) 1.1 Å apart along x, bonded by SHAKE
    std::shared_ptr<molsys::SimulationBox> makeShakeDimer(
        constraints::Constraints &constraints
    )
    {
        auto molecule = molsys::Molecule();
        auto atom1    = std::make_shared<molsys::Atom>();
        auto atom2    = std::make_shared<molsys::Atom>();

        atom1->setPosition(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom);
        atom2->setPosition(linalg::Vec3D(2.1, 1.0, 1.0) * angstrom);
        atom1->setVelocity(Vel3D::zero());
        atom2->setVelocity(Vel3D::zero());
        atom1->setMass(1.0 * g_per_mol);
        atom2->setMass(3.0 * g_per_mol);
        molecule.addAtom(atom1);
        molecule.addAtom(atom2);

        auto simulationBox = std::make_shared<molsys::SimulationBox>();
        simulationBox->addMolecule(molecule);
        simulationBox->addAtom(atom1);
        simulationBox->addAtom(atom2);
        simulationBox->setBox(
            molsys::OrthorhombicBox{
                {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
            },
            false
        );

        auto *moleculePtr = simulationBox->getMolecules().data();
        constraints.addBondConstraint(
            constraints::BondConstraint(
                moleculePtr,
                moleculePtr,
                AtomIndex{0},
                AtomIndex{1},
                1.1 * angstrom
            )
        );

        return simulationBox;
    }
}   // namespace

TEST(TestOptEngine, aConvergedRunDoesNotThrow)
{
    EXPECT_NO_THROW(engine::OptEngine::throwOnFailure(true, false, 4, 10, {}));
}

TEST(TestOptEngine, aRunWithoutConvergenceThrows)
{
    EXPECT_THROW_MSG(
        engine::OptEngine::throwOnFailure(false, false, 10, 10, {}),
        exc::OptException,
        "Optimizer did not converge after 10 epochs."
    );
}

TEST(TestOptEngine, aStoppedRunReportsTheLearningRateErrors)
{
    // the engine only sets "stopped" for a run that has not converged
    EXPECT_THROW_MSG(
        engine::OptEngine::throwOnFailure(
            false,
            true,
            4,
            10,
            std::vector<std::string>{"learning rate too small", "stalled"}
        ),
        exc::OptException,
        "Optimizer stopped after 4 epochs out of 10. The following error "
        "messages were raised:\n1) learning rate too small\n2) stalled\n"
    );
}

TEST(TestOptEngine, aRunThatHitsTheEpochLimitIsNotReportedAsStopped)
{
    try
    {
        engine::OptEngine::throwOnFailure(false, false, 10, 10, {"unused"});
        FAIL() << "expected an OptException";
    }
    catch (const exc::OptException &exception)
    {
        EXPECT_EQ(
            std::string(exception.what()).find("stopped"),
            std::string::npos
        );
    }
}

TEST(TestOptEngine, aRunThatIsNotStoppedDoesNotThrow)
{
    EXPECT_NO_THROW(engine::OptEngine::throwIfStopped(false, 10, 10, {}));
}

TEST(TestOptEngine, aStoppedRunThrowsWithoutTheConvergenceCheck)
{
    EXPECT_THROW_MSG(
        engine::OptEngine::throwIfStopped(true, 3, 10, {"stalled"}),
        exc::OptException,
        "Optimizer stopped after 3 epochs out of 10. The following error "
        "messages were raised:\n1) stalled\n"
    );
}

TEST(TestOptEngine, theNotConvergedMessageNamesTheEpochLimit)
{
    EXPECT_EQ(
        engine::OptEngine::notConvergedMessage(25),
        "Optimizer did not converge after 25 epochs."
    );
}

TEST(TestOptEngine, theOptimizedEnergyAddsTheDistanceConstraintPenalties)
{
    physicalData::PhysicalData physicalData;
    physicalData.addBondEnergy(1.0 * kcal_per_mol);
    physicalData.set_lowerDistanceConstraints(0.25 * kcal_per_mol);
    physicalData.set_upperDistanceConstraints(0.5 * kcal_per_mol);

    EXPECT_NEAR(
        engine::OptEngine::optimizedEnergy(physicalData)
            .numerical_value_in(kcal_per_mol),
        1.75,
        1e-12
    );
}

TEST(TestOptEngine, evaluateConstrainedProjectsTheForcesOntoShake)
{
    const auto constraintSettings =
        testSettings::make<ConstraintSettings>()
            .apply([](auto &config) { config.activateShakeByUser(); })
            .build();
    constraints::Constraints constraints(constraintSettings);
    auto simulationBox = makeShakeDimer(constraints);

    const Settings         settings;
    ConstantForceEvaluator evaluator(settings);
    evaluator.setSimulationBox(simulationBox);

    physicalData::PhysicalData physicalData;
    const auto                 energy = engine::OptEngine::evaluateConstrained(
        evaluator,
        constraints,
        *simulationBox,
        physicalData
    );

    EXPECT_NEAR(energy.numerical_value_in(kcal_per_mol), 1.0, 1e-12);

    const auto &atoms = simulationBox->getAtoms();
    EXPECT_THAT(
        atoms[0]->getForce().numerical_value_in(kcal_per_mol / angstrom),
        testing::ElementsAre(
            testing::DoubleNear(1.0, 1e-9),
            testing::DoubleNear(2.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9)
        )
    );
    EXPECT_THAT(
        atoms[1]->getForce().numerical_value_in(kcal_per_mol / angstrom),
        testing::ElementsAre(
            testing::DoubleNear(3.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9),
            testing::DoubleNear(0.0, 1e-9)
        )
    );
}

TEST(TestOptEngine, evaluateConstrainedWithoutConstraintsKeepsTheForces)
{
    const auto constraintSettings =
        testSettings::make<ConstraintSettings>().build();
    constraints::Constraints constraints(constraintSettings);
    auto simulationBox = makeShakeDimer(constraints);

    const Settings         settings;
    ConstantForceEvaluator evaluator(settings);
    evaluator.setSimulationBox(simulationBox);

    physicalData::PhysicalData physicalData;
    std::ignore = engine::OptEngine::evaluateConstrained(
        evaluator,
        constraints,
        *simulationBox,
        physicalData
    );

    EXPECT_THAT(
        simulationBox->getAtoms()[0]->getForce().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        testing::ElementsAre(4.0, 2.0, 0.0)
    );
}
