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

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "exceptions.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "resetKinetics.hpp"
#include "resetKineticsSettings.hpp"
#include "simulationBox.hpp"
#include "testSettings.hpp"
#include "throwWithMessage.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d.hpp"
#include "vectorNear.hpp"

namespace
{
    // Build a fresh 3-atom molsys::SimulationBox the tests can share: 2
    // atoms in mol1, 1 atom in mol2, all mass 1, deterministic velocities.
    molsys::SimulationBox *makeBox()
    {
        auto *box      = new molsys::SimulationBox();
        auto  molecule = molsys::Molecule();

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();
        atom1->setMass(1.0 * g_per_mol);
        atom2->setMass(1.0 * g_per_mol);
        atom1->setPosition(linalg::Vec3D(0.0, 0.0, 0.0) * angstrom);
        atom2->setPosition(linalg::Vec3D(1.0, 0.0, 0.0) * angstrom);
        atom1->setVelocity(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s);
        atom2->setVelocity(linalg::Vec3D(1.0, 2.0, 3.0) * angstrom / s);
        molecule.addAtom(atom1);
        molecule.addAtom(atom2);

        auto molecule2 = molsys::Molecule();
        auto atom3     = std::make_shared<molsys::Atom>();
        atom3->setMass(1.0 * g_per_mol);
        atom3->setPosition(linalg::Vec3D(0.0, 1.0, 0.0) * angstrom);
        atom3->setVelocity(linalg::Vec3D(1.0, 1.0, 1.0) * angstrom / s);
        molecule2.addAtom(atom3);

        box->addMolecule(molecule);
        box->addMolecule(molecule2);
        box->addAtom(atom1);
        box->addAtom(atom2);
        box->addAtom(atom3);
        box->setTotalMass(3.0 * g_per_mol);
        box->calculateDegreesOfFreedom();

        return box;
    }

    // A frequency that never fires for the (small) steps used in the tests:
    // step % never != 0 for 0 < step < never.
    constexpr size_t never = 1'000'000U;

    std::vector<Vel3D> velocitiesOf(molsys::SimulationBox &simulationBox)
    {
        std::vector<Vel3D> velocities;
        for (const auto &atom : simulationBox.getAtoms())
            velocities.push_back(atom->getVelocity());
        return velocities;
    }

    std::vector<Pos3D> positionsOf(molsys::SimulationBox &simulationBox)
    {
        std::vector<Pos3D> positions;
        for (const auto &atom : simulationBox.getAtoms())
            positions.push_back(atom->getPosition());
        return positions;
    }

    // velocities can become huge when scaling to a target temperature (the
    // fixture starts at a tiny temperature) -> compare relative to that scale
    Vel velocityTolerance(molsys::SimulationBox &simulationBox)
    {
        Vel scale = 1.0 * angstrom / s;
        for (const auto &velocity : velocitiesOf(simulationBox))
            for (size_t i = 0; i < 3; ++i)
                scale = mstd::max(scale, mstd::abs(velocity[i]));

        return 1e-12 * scale;
    }

    // total angular momentum about the centre of mass, as reset() sees it
    AngularMomentum3D angularMomentumOf(molsys::SimulationBox &simulationBox)
    {
        simulationBox.calculateCenterOfMass();
        return simulationBox.calculateAngularMomentum(
            simulationBox.calculateMomentum()
        );
    }

    // fill PhysicalData the same way the MD loop does before reset() is called
    physicalData::PhysicalData makeData(molsys::SimulationBox &simulationBox)
    {
        auto data = physicalData::PhysicalData();
        simulationBox.calculateCenterOfMass();
        data.calculateKinetics(simulationBox);
        data.calculateTemperature(simulationBox);
        return data;
    }

    // reset() has to leave PhysicalData consistent with the SimulationBox
    // (momenta are stored in SI units in PhysicalData, internal units in
    // the box)
    void expectDataMatchesBox(
        physicalData::PhysicalData &physicalData,
        molsys::SimulationBox      &simulationBox
    )
    {
        const auto tolerance =
            velocityTolerance(simulationBox) *
            mstd::max(1.0 * g_per_mol, simulationBox.getTotalMass());

        const auto momentum        = simulationBox.calculateMomentum();
        const auto angularMomentum = angularMomentumOf(simulationBox);

        EXPECT_NEAR(
            physicalData.getTemperature()
                .quantity_from_zero()
                .numerical_value_in(K),
            simulationBox.calculateTemperature()
                .quantity_from_zero()
                .numerical_value_in(K),
            1e-9
        );
        EXPECT_VECTOR_NEAR(
            physicalData.getMomentum().numerical_value_in(
                g_per_mol * angstrom / s
            ),
            momentum.numerical_value_in(g_per_mol * angstrom / s),
            tolerance.numerical_value_in(angstrom * g_per_mol / s)
        );
        EXPECT_VECTOR_NEAR(
            physicalData.getAngularMomentum().numerical_value_in(
                g_per_mol * angstrom * angstrom / s
            ),
            angularMomentum.numerical_value_in(
                g_per_mol * angstrom * angstrom / s
            ),
            tolerance.numerical_value_in(angstrom * g_per_mol / s)
        );
    }

    // true if reset() at the given step modified the velocities of a fresh box
    bool resetModifiesVelocities(
        const kinetics::ResetKinetics &reset,
        size_t                         step
    )
    {
        auto *box    = makeBox();
        auto  data   = makeData(*box);
        auto  before = velocitiesOf(*box);

        reset.reset(step, data, *box, toTemperature(300.0));

        const auto modified = (before != velocitiesOf(*box));
        delete box;
        return modified;
    }

    // masses 1, 2, 4 - total mass 7
    void makeMassesNonUniform(molsys::SimulationBox &simulationBox)
    {
        const std::vector<double> masses = {1.0, 2.0, 4.0};

        for (size_t i = 0; i < masses.size(); ++i)
            simulationBox.getAtoms()[i]->setMass(masses[i] * g_per_mol);

        simulationBox.setTotalMass(7.0 * g_per_mol);
    }

    ResetKineticsSettings makeSettings(
        size_t nScale,
        size_t fScale,
        size_t nReset,
        size_t fReset,
        size_t nResetAngular,
        size_t fResetAngular,
        size_t fResetForces,
        size_t numberOfSteps
    )
    {
        return testSettings::make<ResetKineticsSettings>()
            .with(&ResetKineticsSettings::setNScaleByUser, nScale)
            .with(&ResetKineticsSettings::setFScaleByUser, fScale)
            .with(&ResetKineticsSettings::setNResetByUser, nReset)
            .with(&ResetKineticsSettings::setFResetByUser, fReset)
            .with(&ResetKineticsSettings::setNResetAngularByUser, nResetAngular)
            .with(&ResetKineticsSettings::setFResetAngularByUser, fResetAngular)
            .with(&ResetKineticsSettings::setFResetForcesByUser, fResetForces)
            .apply([numberOfSteps](auto &config)
                   { config.finalize(numberOfSteps); })
            .build();
    }
}   // namespace

TEST(TestResetKinetics, resetTemperatureRescalesVelocitiesAndStaysFinite)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();

    // resetTemperature scales by sqrt(target / temperature) with the
    // temperature handed in by the caller (reset() takes it from PhysicalData)
    data.calculateTemperature(*box);

    kinetics::ResetKinetics::resetTemperature(
        *box,
        data.getTemperature(),
        toTemperature(300.0)
    );

    data.calculateTemperature(*box);
    const auto T_after =
        data.getTemperature().quantity_from_zero().numerical_value_in(K);

    EXPECT_FALSE(std::isnan(T_after));
    EXPECT_FALSE(std::isinf(T_after));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureScalesFiniteTemperatureToZero)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();
    data.calculateTemperature(*box);
    kinetics::ResetKinetics::resetTemperature(
        *box,
        data.getTemperature(),
        toTemperature(0.0)
    );

    data.calculateTemperature(*box);
    EXPECT_DOUBLE_EQ(
        data.getTemperature().quantity_from_zero().numerical_value_in(K),
        0.0
    );
    for (const auto &atom : box->getAtoms())
    {
        EXPECT_EQ(
            atom->getVelocity(),
            linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
        );
    }

    delete box;
}

TEST(TestResetKinetics, rejectsZeroTargetFromZeroTemperature)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    EXPECT_THROW_MSG(
        kinetics::ResetKinetics::resetTemperature(
            *box,
            toTemperature(0.0),
            toTemperature(0.0)
        ),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    delete box;
}

TEST(TestResetKinetics, rejectsPositiveTargetFromZeroTemperature)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    EXPECT_THROW_MSG(
        kinetics::ResetKinetics::resetTemperature(
            *box,
            toTemperature(0.0),
            toTemperature(300.0)
        ),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    delete box;
}

TEST(TestResetKinetics, resetMomentumZerosTotalLinearMomentum)
{
    auto *box = makeBox();

    // resetMomentum subtracts (momentum / totalMass) from every atom's
    // velocity; for the total to land at zero the momentum handed in has to
    // be the current total p = sum m_i v_i (reset() takes it from
    // data.getMomentum()).
    Momentum3D totalP = Momentum::zero();
    for (const auto &atom : box->getAtoms())
        totalP += atom->getMass() * atom->getVelocity();

    kinetics::ResetKinetics::resetMomentum(*box, totalP);

    Momentum3D totalPAfter = Momentum::zero();
    for (const auto &atom : box->getAtoms())
        totalPAfter += atom->getMass() * atom->getVelocity();

    EXPECT_NEAR(
        totalPAfter[0].numerical_value_in(g_per_mol * angstrom / s),
        0.0,
        1e-12
    );
    EXPECT_NEAR(
        totalPAfter[1].numerical_value_in(g_per_mol * angstrom / s),
        0.0,
        1e-12
    );
    EXPECT_NEAR(
        totalPAfter[2].numerical_value_in(g_per_mol * angstrom / s),
        0.0,
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumLeavesVelocitiesFinite)
{
    auto *box = makeBox();

    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        AngularMomentum3D{AngularMomentum::zero()}
    );

    for (const auto &atom : box->getAtoms())
    {
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(isinf(atom->getVelocity()[i]));
        }
    }

    delete box;
}

TEST(TestResetKinetics, resetForcesZerosForcesEachStep)
{
    auto                 *box = makeBox();
    ResetKineticsSettings settings =
        makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 1U, 100U);
    kinetics::ResetKinetics reset{settings};

    // Seed atom forces with non-zero values.
    for (auto &atom : box->getAtoms())
        atom->setForce(linalg::Vec3D(1.0, 2.0, 3.0) * kcal_per_mol / angstrom);

    reset.resetForces(0U, *box);

    for (const auto &atom : box->getAtoms())
    {
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_DOUBLE_EQ(
                atom->getForce()[i].numerical_value_in(kcal_per_mol / angstrom),
                0.0
            );
        }
    }

    delete box;
}

/*****************************
 *                           *
 * resetTemperature (static) *
 *                           *
 ****************************/

TEST(TestResetKinetics, resetTemperatureReachesTargetTemperature)
{
    for (const auto target : {50.0, 300.0, 1500.0})
    {
        SCOPED_TRACE("target temperature " + std::to_string(target));

        auto *box  = makeBox();
        auto  data = makeData(*box);

        kinetics::ResetKinetics::resetTemperature(
            *box,
            data.getTemperature(),
            toTemperature(target)
        );

        EXPECT_NEAR(
            box->calculateTemperature().quantity_from_zero().numerical_value_in(
                K
            ),
            target,
            1e-10 * target
        );

        delete box;
    }
}

TEST(TestResetKinetics, resetTemperatureScalesAllVelocitiesByCommonFactor)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto before    = velocitiesOf(*box);
    const auto positions = positionsOf(*box);
    const auto lambda    = ::sqrt(
        300.0 / data.getTemperature().quantity_from_zero().numerical_value_in(K)
    );

    kinetics::ResetKinetics::resetTemperature(
        *box,
        data.getTemperature(),
        toTemperature(300.0)
    );

    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_VECTOR_NEAR(
            after[i].numerical_value_in(angstrom / s),
            (before[i] * lambda).numerical_value_in(angstrom / s),
            velocityTolerance(*box).numerical_value_in(angstrom / s)
        );
    }

    // only velocities are touched
    EXPECT_EQ(positionsOf(*box), positions);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureUsesPassedTemperatureNotBoxTemperature)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    // claim the system is 4 times hotter than it is - the scaling factor has
    // to follow the argument: sqrt(target / (4 T)) = 0.5 * sqrt(target / T)
    const auto before = velocitiesOf(*box);
    const auto lambda =
        0.5 *
        ::sqrt(
            300.0 /
            data.getTemperature().quantity_from_zero().numerical_value_in(K)
        );

    kinetics::ResetKinetics::resetTemperature(
        *box,
        scaleAbsTemp(data.getTemperature(), 4.0),
        toTemperature(300.0)
    );

    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_VECTOR_NEAR(
            after[i].numerical_value_in(angstrom / s),
            (before[i] * lambda).numerical_value_in(angstrom / s),
            velocityTolerance(*box).numerical_value_in(angstrom / s)
        );
    }

    delete box;
}

TEST(TestResetKinetics, resetTemperatureIsIdempotent)
{
    auto *box = makeBox();

    kinetics::ResetKinetics::resetTemperature(
        *box,
        toTemperature(box->calculateTemperature()
                          .quantity_from_zero()
                          .numerical_value_in(K)),
        toTemperature(box->calculateTemperature()
                          .quantity_from_zero()
                          .numerical_value_in(K))
    );
    const auto once = velocitiesOf(*box);

    // already at the target -> lambda = 1
    kinetics::ResetKinetics::resetTemperature(
        *box,
        toTemperature(box->calculateTemperature()
                          .quantity_from_zero()
                          .numerical_value_in(K)),
        toTemperature(box->calculateTemperature()
                          .quantity_from_zero()
                          .numerical_value_in(K))
    );

    EXPECT_VELOCITIES_NEAR(*box, once, velocityTolerance(*box));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureCanHeatAndCool)
{
    auto      *box   = makeBox();
    auto       data  = makeData(*box);
    const auto temp0 = data.getTemperature();

    kinetics::ResetKinetics::resetTemperature(
        *box,
        temp0,
        scaleAbsTemp(temp0, 2.0)
    );
    EXPECT_NEAR(
        box->calculateTemperature().quantity_from_zero().numerical_value_in(K),
        scaleAbsTemp(temp0, 2.0).quantity_from_zero().numerical_value_in(K),
        1e-10 * temp0.quantity_from_zero().numerical_value_in(K)
    );

    kinetics::ResetKinetics::resetTemperature(
        *box,
        scaleAbsTemp(temp0, 2.0),
        scaleAbsTemp(temp0, 0.25)
    );
    EXPECT_NEAR(
        box->calculateTemperature().quantity_from_zero().numerical_value_in(K),
        scaleAbsTemp(temp0, 0.25).quantity_from_zero().numerical_value_in(K),
        1e-10 * temp0.quantity_from_zero().numerical_value_in(K)
    );

    delete box;
}

TEST(TestResetKinetics, resetTemperatureThrowsBeforeModifyingVelocities)
{
    auto *box = makeBox();

    const auto before = velocitiesOf(*box);

    // the box itself is hot, but the temperature handed in is zero - the
    // decision to throw is based on the argument
    EXPECT_THROW_MSG(
        kinetics::ResetKinetics::resetTemperature(
            *box,
            toTemperature(0.0),
            toTemperature(300.0)
        ),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    EXPECT_EQ(velocitiesOf(*box), before);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureOfColdBoxWithNonZeroArgumentStaysCold)
{
    auto *box = makeBox();
    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    EXPECT_NO_THROW(
        kinetics::ResetKinetics::resetTemperature(
            *box,
            toTemperature(10.0),
            toTemperature(300.0)
        )
    );

    for (const auto &atom : box->getAtoms())
    {
        EXPECT_EQ(
            atom->getVelocity(),
            linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
        );
    }

    delete box;
}

/***************************
 *                         *
 * resetMomentum (static)  *
 *                         *
 **************************/

TEST(TestResetKinetics, resetMomentumZerosMomentumForNonUniformMasses)
{
    auto *box = makeBox();
    makeMassesNonUniform(*box);

    const auto momentum = box->calculateMomentum();
    ASSERT_GT(norm(momentum), 1.0 * momentum[0].unit);

    kinetics::ResetKinetics::resetMomentum(*box, momentum);

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetMomentumShiftsAllVelocitiesByCentreOfMassVelocity)
{
    auto *box = makeBox();

    const auto before    = velocitiesOf(*box);
    const auto positions = positionsOf(*box);
    const auto momentum  = box->calculateMomentum();
    const auto vCom      = momentum / box->getTotalMass();

    kinetics::ResetKinetics::resetMomentum(*box, momentum);

    // velocities relative to each other stay untouched
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_VECTOR_NEAR(
            after[i].numerical_value_in(angstrom / s),
            (before[i] - vCom).numerical_value_in(angstrom / s),
            1e-12
        );
    }

    EXPECT_EQ(positionsOf(*box), positions);

    delete box;
}

TEST(TestResetKinetics, resetMomentumSubtractsExactlyThePassedMomentum)
{
    const std::vector<Momentum3D> corrections = {
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom * g_per_mol / s,
        linalg::Vec3D(1.0, -2.0, 3.0) * angstrom * g_per_mol / s,
        linalg::Vec3D(-10.0, 0.5, 0.25) * angstrom * g_per_mol / s,
    };

    for (const auto &correction : corrections)
    {
        auto *box = makeBox();
        makeMassesNonUniform(*box);

        const auto expected = (box->calculateMomentum() - correction)
                                  .numerical_value_in(g_per_mol * angstrom / s);

        kinetics::ResetKinetics::resetMomentum(*box, correction);

        EXPECT_VECTOR_NEAR(
            box->calculateMomentum().numerical_value_in(
                g_per_mol * angstrom / s
            ),
            expected,
            1e-12
        );

        delete box;
    }
}

TEST(TestResetKinetics, resetMomentumWithZeroMomentumLeavesVelocitiesUnchanged)
{
    auto *box = makeBox();

    const auto before = velocitiesOf(*box);

    kinetics::ResetKinetics::resetMomentum(
        *box,
        linalg::Vec3D(0.0, 0.0, 0.0) * angstrom * g_per_mol / s
    );

    EXPECT_EQ(velocitiesOf(*box), before);

    delete box;
}

TEST(TestResetKinetics, resetMomentumIsIdempotent)
{
    auto *box = makeBox();

    kinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());
    const auto once = velocitiesOf(*box);

    kinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());

    EXPECT_VELOCITIES_NEAR(*box, once, (1e-12 * angstrom / s));

    delete box;
}

TEST(TestResetKinetics, resetMomentumRemovesPureTranslationCompletely)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D(0.3, -0.7, 1.1) * angstrom / s);

    kinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());

    for (const auto &atom : box->getAtoms())
    {
        EXPECT_VECTOR_NEAR(
            atom->getVelocity().numerical_value_in(angstrom / s),
            linalg::Vec3D(0.0, 0.0, 0.0),
            1e-12
        );
    }

    delete box;
}

/**********************************
 *                                *
 * resetAngularMomentum (static)  *
 *                                *
 *********************************/

TEST(TestResetKinetics, resetAngularMomentumZerosAngularMomentum)
{
    auto *box = makeBox();

    const auto angularMomentum = angularMomentumOf(*box);
    ASSERT_GT(norm(angularMomentum), 1e-3 * angularMomentum[0].unit);

    kinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(
    TestResetKinetics,
    resetAngularMomentumZerosAngularMomentumNonUniformMasses
)
{
    auto *box = makeBox();
    makeMassesNonUniform(*box);
    box->calculateCenterOfMass();

    const auto angularMomentum = angularMomentumOf(*box);
    ASSERT_GT(norm(angularMomentum), 1e-3 * angularMomentum[0].unit);

    kinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumPreservesLinearMomentum)
{
    auto *box = makeBox();
    makeMassesNonUniform(*box);

    const auto momentum        = box->calculateMomentum();
    const auto angularMomentum = angularMomentumOf(*box);

    kinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        momentum.numerical_value_in(g_per_mol * angstrom / s),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumSubtractsExactlyThePassedValue)
{
    const auto scaled = [](const AngularMomentum3D &vec, double factor)
    { return vec * factor; };

    auto      *reference        = makeBox();
    const auto angularMomentum0 = angularMomentumOf(*reference);
    delete reference;

    const std::vector<AngularMomentum3D> corrections = {
        AngularMomentum::zero(),
        scaled(angularMomentum0, 0.5),
        linalg::Vec3D(0.3, -0.2, 0.5) * g_per_mol * angstrom * angstrom / s,
    };

    for (const auto &correction : corrections)
    {
        auto *box = makeBox();

        const auto expected =
            (angularMomentumOf(*box) - correction)
                .numerical_value_in(g_per_mol * angstrom * angstrom / s);

        kinetics::ResetKinetics::resetAngularMomentum(*box, correction);

        EXPECT_VECTOR_NEAR(
            angularMomentumOf(*box).numerical_value_in(
                g_per_mol * angstrom * angstrom / s
            ),
            expected,
            1e-12
        );

        delete box;
    }
}

TEST(TestResetKinetics, resetAngularMomentumWithZeroLeavesVelocitiesUnchanged)
{
    auto *box = makeBox();

    const auto before = velocitiesOf(*box);

    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        linalg::Vec3D(0.0, 0.0, 0.0) * g_per_mol * angstrom * angstrom / s
    );

    EXPECT_EQ(velocitiesOf(*box), before);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumRemovesRigidRotationCompletely)
{
    auto *box = makeBox();
    makeMassesNonUniform(*box);
    box->calculateCenterOfMass();

    // pure rigid rotation about the centre of mass: v_i = omega x (r_i - R)
    const linalg::Vector3D<mstd::quantity<one / s>> omega =
        linalg::Vec3D(0.1, 0.2, -0.3) / s;
    const auto centerOfMass = box->getCenterOfMass();

    for (const auto &atom : box->getAtoms())
    {
        atom->setVelocity(cross(omega, atom->getPosition() - centerOfMass));
    }

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );
    const auto angularMomentum = angularMomentumOf(*box);
    ASSERT_GT(norm(angularMomentum), 1e-3 * angularMomentum[0].unit);

    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    for (const auto &atom : box->getAtoms())
    {
        EXPECT_VECTOR_NEAR(
            atom->getVelocity().numerical_value_in(angstrom / s),
            linalg::Vec3D(0.0, 0.0, 0.0),
            1e-12
        );
    }

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumLeavesPureTranslationUntouched)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D(0.3, -0.7, 1.1) * angstrom / s);

    const auto before = velocitiesOf(*box);

    // a translating body has no angular momentum about its centre of mass
    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    EXPECT_VELOCITIES_NEAR(*box, before, (1e-12 * angstrom / s));

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumRefreshesCentreOfMass)
{
    auto *box = makeBox();

    // makeBox never calculated the centre of mass -> stale value
    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        linalg::Vec3D(0.0, 0.0, 0.0) * g_per_mol * angstrom * angstrom / s
    );

    // positions (0,0,0), (1,0,0), (0,1,0), equal masses
    EXPECT_VECTOR_NEAR(
        box->getCenterOfMass().numerical_value_in(angstrom),
        linalg::Vec3D(1.0 / 3.0, 1.0 / 3.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumDoesNotIncreaseKineticEnergy)
{
    auto *box = makeBox();

    const auto T_before = box->calculateTemperature();

    kinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    // the removed rotational energy is 1/2 omega . L > 0
    EXPECT_LT(box->calculateTemperature(), T_before);
    EXPECT_GT(box->calculateTemperature(), Temperature::zero());

    delete box;
}

/*****************************
 *                           *
 * reset - dispatch by step  *
 *                           *
 ****************************/

TEST(TestResetKinetics, resetWithoutScheduledResetLeavesBoxUntouched)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    // reset() must be callable on a const object
    const auto settings =
        makeSettings(0U, never, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    const auto velocities      = velocitiesOf(*box);
    const auto temp            = data.getTemperature();
    const auto momentum        = data.getMomentum();
    const auto angularMomentum = data.getAngularMomentum();

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_EQ(velocitiesOf(*box), velocities);
    EXPECT_EQ(data.getTemperature(), temp);
    EXPECT_VECTOR_NEAR(
        data.getMomentum().numerical_value_in(g_per_mol * angstrom / s),
        momentum.numerical_value_in(g_per_mol * angstrom / s),
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        data.getAngularMomentum().numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        angularMomentum.numerical_value_in(g_per_mol * angstrom * angstrom / s),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetWithoutScheduledResetTakesDataAsSourceOfTruth)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();

    const auto settings =
        makeSettings(0U, never, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    // values in PhysicalData deliberately do not match the box
    const linalg::Vec3D momentum(1.0, 2.0, 3.0);
    const linalg::Vec3D angularMomentum(-4.0, 5.0, -6.0);
    data.setTemperature(toTemperature(123.0));
    data.setMomentum(momentum * g_per_mol * angstrom / s);
    data.setAngularMomentum(
        angularMomentum * g_per_mol * angstrom * angstrom / s
    );

    const auto velocities = velocitiesOf(*box);

    reset.reset(7U, data, *box, toTemperature(300.0));

    // nothing recalculated - the unit conversions S_TO_FS / FS_TO_S cancel
    EXPECT_EQ(velocitiesOf(*box), velocities);
    EXPECT_EQ(data.getTemperature(), toTemperature(123.0));
    EXPECT_VECTOR_NEAR(
        data.getMomentum().numerical_value_in(g_per_mol * angstrom / s),
        momentum,
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        data.getAngularMomentum().numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        angularMomentum,
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetTemperatureBranchScalesAndRemovesMomentum)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, 7U, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    const auto before = velocitiesOf(*box);
    const auto vCom   = box->calculateMomentum() / box->getTotalMass();
    const auto lambda = ::sqrt(
        300.0 / data.getTemperature().quantity_from_zero().numerical_value_in(K)
    );

    reset.reset(7U, data, *box, toTemperature(300.0));

    // hard scaling first, then removal of the (scaled) centre of mass motion
    // -> v_i' = lambda * (v_i - v_com)
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_VECTOR_NEAR(
            after[i].numerical_value_in(angstrom / s),
            ((before[i] - vCom) * lambda).numerical_value_in(angstrom / s),
            velocityTolerance(*box).numerical_value_in(angstrom / s)
        );
    }

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );

    // the removed centre of mass motion carried kinetic energy
    EXPECT_LT(
        data.getTemperature().quantity_from_zero().numerical_value_in(K),
        300.0
    );
    EXPECT_GT(
        data.getTemperature().quantity_from_zero().numerical_value_in(K),
        0.0
    );

    // angular momentum was not scheduled
    const auto angularMomentumNorm = norm(angularMomentumOf(*box));
    EXPECT_GT(angularMomentumNorm, 1e-3 * angularMomentumNorm.unit);

    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetMomentumBranchOnlyShiftsVelocities)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, never, 0U, 7U, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    const auto before = velocitiesOf(*box);
    const auto vCom   = box->calculateMomentum() / box->getTotalMass();

    reset.reset(7U, data, *box, toTemperature(300.0));

    // no temperature rescaling, only v_i' = v_i - v_com
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        EXPECT_VECTOR_NEAR(
            after[i].numerical_value_in(angstrom / s),
            (before[i] - vCom).numerical_value_in(angstrom / s),
            1e-12
        );
    }

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );
    EXPECT_NEAR(
        data.getMomentum().numerical_value_in(g_per_mol * angstrom / s)[0],
        0.0,
        1e-12
    );

    // the temperature in data follows the (unscaled) box
    EXPECT_GT(
        std::abs(
            data.getTemperature().quantity_from_zero().numerical_value_in(K) -
            300.0
        ),
        1e-3
    );
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetAngularBranchOnlyRemovesRotation)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, never, 0U, never, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    const auto momentum = box->calculateMomentum();
    const auto T_before = data.getTemperature();

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    // linear momentum is a conserved quantity of the rotation correction, and
    // has to be reported unchanged (in SI units) in PhysicalData
    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        momentum.numerical_value_in(g_per_mol * angstrom / s),
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        data.getMomentum().numerical_value_in(g_per_mol * angstrom / s),
        momentum.numerical_value_in(g_per_mol * angstrom / s),
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        data.getAngularMomentum().numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    // the rotational energy is gone -> data has to carry the new temperature
    EXPECT_LT(data.getTemperature(), T_before);
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureAndMomentumDueTogetherActOnlyOnce)
{
    auto *boxTemperature  = makeBox();
    auto  dataTemperature = makeData(*boxTemperature);
    auto *boxBoth         = makeBox();
    auto  dataBoth        = makeData(*boxBoth);

    const auto settingsOnlyTemperature =
        makeSettings(0U, 7U, 0U, never, 0U, never, 1U, 100U);
    const auto settingsTemperatureAndMomentum =
        makeSettings(0U, 7U, 0U, 7U, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics onlyTemperature{settingsOnlyTemperature};
    const kinetics::ResetKinetics temperatureAndMomentum{
        settingsTemperatureAndMomentum
    };

    onlyTemperature
        .reset(7U, dataTemperature, *boxTemperature, toTemperature(300.0));
    temperatureAndMomentum.reset(7U, dataBoth, *boxBoth, toTemperature(300.0));

    // the temperature reset already includes the momentum reset - a second
    // correction with the stale momentum would shift the velocities again
    EXPECT_EQ(velocitiesOf(*boxBoth), velocitiesOf(*boxTemperature));
    EXPECT_DOUBLE_EQ(
        dataBoth.getTemperature().quantity_from_zero().numerical_value_in(K),
        dataTemperature.getTemperature()
            .quantity_from_zero()
            .numerical_value_in(K)
    );

    delete boxTemperature;
    delete boxBoth;
}

TEST(TestResetKinetics, resetAllBranchesMatchSequentialStaticCalls)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, 7U, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    // reference: temperature -> momentum -> angular momentum
    auto *expectedBox = makeBox();
    expectedBox->calculateCenterOfMass();
    kinetics::ResetKinetics::resetTemperature(
        *expectedBox,
        expectedBox->calculateTemperature(),
        toTemperature(300.0)
    );
    kinetics::ResetKinetics::resetMomentum(
        *expectedBox,
        expectedBox->calculateMomentum()
    );
    kinetics::ResetKinetics::resetAngularMomentum(
        *expectedBox,
        angularMomentumOf(*expectedBox)
    );

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VELOCITIES_NEAR(
        *box,
        velocitiesOf(*expectedBox),
        velocityTolerance(*box)
    );
    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    expectDataMatchesBox(data, *box);

    // data reports the final state: no momentum, no angular momentum
    EXPECT_VECTOR_NEAR(
        data.getMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    EXPECT_VECTOR_NEAR(
        data.getAngularMomentum().numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );

    delete box;
    delete expectedBox;
}

TEST(TestResetKinetics, resetTemperatureAndAngularDueTogetherKeepMomentumZero)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, 7U, 0U, never, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetMomentumAndAngularDueTogetherZeroBoth)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, never, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );
    expectDataMatchesBox(data, *box);

    delete box;
}

/*****************************
 *                           *
 * reset - step conditions   *
 *                           *
 ****************************/

TEST(TestResetKinetics, resetTemperatureIsScheduledByStepsAndFrequency)
{
    const auto settings =
        makeSettings(5U, 1000U, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    EXPECT_TRUE(resetModifiesVelocities(reset, 1U));      // within nSteps
    EXPECT_TRUE(resetModifiesVelocities(reset, 5U));      // nSteps inclusive
    EXPECT_FALSE(resetModifiesVelocities(reset, 6U));     // first step after
    EXPECT_FALSE(resetModifiesVelocities(reset, 999U));   // before frequency
    EXPECT_TRUE(resetModifiesVelocities(reset, 1000U));   // frequency
    EXPECT_FALSE(resetModifiesVelocities(reset, 1001U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 3000U));   // multiple
}

TEST(TestResetKinetics, resetMomentumIsScheduledByStepsAndFrequency)
{
    const auto settings =
        makeSettings(0U, never, 5U, 1000U, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    EXPECT_TRUE(resetModifiesVelocities(reset, 1U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 5U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 6U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 999U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 1000U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 1001U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 3000U));
}

TEST(TestResetKinetics, resetAngularMomentumIsScheduledByStepsAndFrequency)
{
    const auto settings =
        makeSettings(0U, never, 0U, never, 5U, 1000U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    EXPECT_TRUE(resetModifiesVelocities(reset, 1U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 5U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 6U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 999U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 1000U));
    EXPECT_FALSE(resetModifiesVelocities(reset, 1001U));
    EXPECT_TRUE(resetModifiesVelocities(reset, 3000U));
}

TEST(TestResetKinetics, stepZeroTriggersEveryResetIndependentOfSchedule)
{
    // step 0: 0 % frequency == 0 for every non-zero frequency
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings =
        makeSettings(0U, never, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    reset.reset(0U, data, *box, toTemperature(300.0));

    EXPECT_VECTOR_NEAR(
        box->calculateMomentum().numerical_value_in(g_per_mol * angstrom / s),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box).numerical_value_in(angstrom / s)
    );
    expectDataMatchesBox(data, *box);

    delete box;
}

/*******************************
 *                             *
 * reset - errors and reuse    *
 *                             *
 ******************************/

TEST(TestResetKinetics, resetThrowsForZeroTemperatureInDataWhenScheduled)
{
    auto *box = makeBox();

    // PhysicalData has not been filled -> temperature 0.0
    auto data = physicalData::PhysicalData();

    const auto settings = makeSettings(0U, 7U, 0U, never, 0U, never, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    const auto velocities = velocitiesOf(*box);

    EXPECT_THROW_MSG(
        reset.reset(7U, data, *box, toTemperature(300.0)),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    EXPECT_EQ(velocitiesOf(*box), velocities);

    delete box;
}

TEST(TestResetKinetics, resetOfMomentumAndAngularDoesNotNeedTemperature)
{
    auto *box = makeBox();
    for (const auto &atom : box->getAtoms())
        atom->setVelocity(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom / s);

    auto data = makeData(*box);
    ASSERT_DOUBLE_EQ(
        data.getTemperature().quantity_from_zero().numerical_value_in(K),
        0.0
    );

    const auto settings = makeSettings(0U, never, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    EXPECT_NO_THROW(reset.reset(7U, data, *box, toTemperature(300.0)));

    for (const auto &atom : box->getAtoms())
    {
        EXPECT_EQ(
            atom->getVelocity(),
            linalg::Vec3D(0.0, 0.0, 0.0) * angstrom / s
        );
    }

    EXPECT_DOUBLE_EQ(
        data.getTemperature().quantity_from_zero().numerical_value_in(K),
        0.0
    );

    delete box;
}

TEST(TestResetKinetics, resetIsIdempotentForMomentumAndAngularMomentum)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, never, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    reset.reset(7U, data, *box, toTemperature(300.0));
    const auto once = velocitiesOf(*box);

    data = makeData(*box);
    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VELOCITIES_NEAR(*box, once, (1e-12 * angstrom / s));

    delete box;
}

TEST(TestResetKinetics, resetDoesNotCarryStateBetweenCalls)
{
    const auto settings = makeSettings(0U, 7U, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reused{settings};

    // first call with a very different system
    auto *boxA = makeBox();
    for (const auto &atom : boxA->getAtoms()) atom->scaleVelocity(5.0);
    auto dataA = makeData(*boxA);
    reused.reset(7U, dataA, *boxA, toTemperature(300.0));

    // second call with the same object must behave like a fresh object
    auto *boxB  = makeBox();
    auto  dataB = makeData(*boxB);
    reused.reset(7U, dataB, *boxB, toTemperature(300.0));

    const auto freshSettings = makeSettings(0U, 7U, 0U, 7U, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics fresh{freshSettings};
    auto                         *boxC  = makeBox();
    auto                          dataC = makeData(*boxC);
    fresh.reset(7U, dataC, *boxC, toTemperature(300.0));

    EXPECT_EQ(velocitiesOf(*boxB), velocitiesOf(*boxC));
    EXPECT_DOUBLE_EQ(
        dataB.getTemperature().quantity_from_zero().numerical_value_in(K),
        dataC.getTemperature().quantity_from_zero().numerical_value_in(K)
    );
    EXPECT_EQ(dataB.getMomentum(), dataC.getMomentum());
    EXPECT_EQ(dataB.getAngularMomentum(), dataC.getAngularMomentum());

    delete boxA;
    delete boxB;
    delete boxC;
}

TEST(TestResetKinetics, resetForcesRespectsStepFrequency)
{
    auto      *box      = makeBox();
    const auto settings = makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 3U, 100U);
    const kinetics::ResetKinetics reset{settings};

    for (auto &atom : box->getAtoms())
        atom->setForce(linalg::Vec3D(1.0, 2.0, 3.0) * kcal_per_mol / angstrom);

    // step 1 and 2 are not multiples of 3
    reset.resetForces(1U, *box);
    reset.resetForces(2U, *box);
    for (const auto &atom : box->getAtoms())
    {
        EXPECT_EQ(
            atom->getForce(),
            linalg::Vec3D(1.0, 2.0, 3.0) * kcal_per_mol / angstrom
        );
    }

    reset.resetForces(3U, *box);
    for (const auto &atom : box->getAtoms())
    {
        EXPECT_VECTOR_NEAR(
            atom->getForce().numerical_value_in(kcal_per_mol / angstrom),
            linalg::Vec3D(0.0, 0.0, 0.0),
            1e-12
        );
    }

    delete box;
}

TEST(TestResetKinetics, resetAngularBranchUsesFreshCentreOfMass)
{
    // In the MD loop PhysicalData::calculateKinetics runs before reset() with
    // whatever centre of mass the box has cached - it is only refreshed on
    // demand. Here it was never calculated (still the origin), so the angular
    // momentum stored in PhysicalData is taken about the wrong point. reset()
    // has to use the current centre of mass, or the angular momentum is not
    // removed.
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();
    data.calculateKinetics(*box);
    data.calculateTemperature(*box);

    const auto settings = makeSettings(0U, never, 0U, never, 0U, 7U, 1U, 100U);
    const kinetics::ResetKinetics reset{settings};

    reset.reset(7U, data, *box, toTemperature(300.0));

    EXPECT_VECTOR_NEAR(
        angularMomentumOf(*box).numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );
    EXPECT_VECTOR_NEAR(
        data.getAngularMomentum().numerical_value_in(
            g_per_mol * angstrom * angstrom / s
        ),
        linalg::Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, momentumDriftWarnsOnceWithoutReset)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const auto settings = makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 0U, 100U);
    kinetics::ResetKinetics reset{settings};

    const auto warning = reset.checkMomentumDrift(5U, data, *box);

    ASSERT_TRUE(warning.has_value());
    EXPECT_THAT(*warning, ::testing::HasSubstr("at step 5"));
    EXPECT_THAT(*warning, ::testing::HasSubstr("no momentum reset is set"));
    EXPECT_THAT(*warning, ::testing::HasSubstr("Set freset"));
    EXPECT_FALSE(reset.checkMomentumDrift(6U, data, *box).has_value());

    delete box;
}

TEST(TestResetKinetics, momentumDriftWarnsAfterInitialReset)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const std::vector<ResetKineticsSettings> initialOnly = {
        makeSettings(0U, 0U, 10U, 0U, 0U, 0U, 0U, 100U),
        makeSettings(10U, 0U, 0U, 0U, 0U, 0U, 0U, 100U),
    };

    for (const auto &settings : initialOnly)
    {
        kinetics::ResetKinetics reset{settings};
        EXPECT_FALSE(reset.checkMomentumDrift(10U, data, *box).has_value());

        const auto warning = reset.checkMomentumDrift(11U, data, *box);

        ASSERT_TRUE(warning.has_value());
        EXPECT_THAT(
            *warning,
            ::testing::HasSubstr("only removed in the first 10 steps")
        );
        EXPECT_THAT(*warning, ::testing::HasSubstr("Set freset"));
    }

    delete box;
}

TEST(TestResetKinetics, momentumDriftWarnsBetweenPeriodicResets)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const std::vector<ResetKineticsSettings> periodic = {
        makeSettings(0U, 0U, 5U, 10U, 0U, 0U, 0U, 100U),
        makeSettings(0U, 20U, 0U, 10U, 0U, 0U, 0U, 100U),
    };

    for (const auto &settings : periodic)
    {
        kinetics::ResetKinetics reset{settings};
        const auto warning = reset.checkMomentumDrift(6U, data, *box);

        ASSERT_TRUE(warning.has_value());
        EXPECT_THAT(*warning, ::testing::HasSubstr("every 10 steps"));
        EXPECT_THAT(*warning, ::testing::HasSubstr("smaller freset"));
    }

    delete box;
}

TEST(TestResetKinetics, momentumDriftSilentWhenResetEveryStep)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const std::vector<ResetKineticsSettings> everyStep = {
        makeSettings(0U, 0U, 0U, 1U, 0U, 0U, 0U, 100U),
        makeSettings(0U, 1U, 0U, 0U, 0U, 0U, 0U, 100U),
    };

    for (const auto &settings : everyStep)
    {
        kinetics::ResetKinetics reset{settings};
        EXPECT_FALSE(reset.checkMomentumDrift(5U, data, *box).has_value());
    }

    delete box;
}

TEST(TestResetKinetics, momentumDriftSilentWithZeroThreshold)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    auto settings = makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 0U, 100U);
    settings.setMomentumWarnThresholdByUser(0.0 * percent);
    kinetics::ResetKinetics reset{settings};

    EXPECT_FALSE(reset.checkMomentumDrift(5U, data, *box).has_value());

    delete box;
}

TEST(TestResetKinetics, momentumDriftSilentBelowThreshold)
{
    auto *box = makeBox();
    kinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());
    box->getAtoms()[0]->addVelocity(
        linalg::Vec3D(1.0e-4, 0.0, 0.0) * angstrom / s
    );
    auto data = makeData(*box);

    const auto settings = makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 0U, 100U);
    kinetics::ResetKinetics reset{settings};

    EXPECT_FALSE(reset.checkMomentumDrift(5U, data, *box).has_value());

    delete box;
}

TEST(TestResetKinetics, momentumResetStepsAndInterval)
{
    const auto none = makeSettings(0U, 0U, 0U, 0U, 0U, 0U, 0U, 100U);
    EXPECT_EQ(none.getInitialMomentumResetSteps(), 0U);
    EXPECT_FALSE(none.getMomentumResetInterval().has_value());

    const auto both = makeSettings(7U, 20U, 3U, 10U, 0U, 0U, 0U, 100U);
    EXPECT_EQ(both.getInitialMomentumResetSteps(), 7U);
    EXPECT_EQ(both.getMomentumResetInterval(), 10U);

    const auto scaleOnly = makeSettings(0U, 4U, 0U, 0U, 0U, 0U, 0U, 100U);
    EXPECT_EQ(scaleOnly.getMomentumResetInterval(), 4U);
}
