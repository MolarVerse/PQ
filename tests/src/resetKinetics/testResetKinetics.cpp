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

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

#include "atom.hpp"
#include "constants/conversionFactors.hpp"
#include "exceptions.hpp"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "resetKinetics.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "throwWithMessage.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

namespace
{
    // Build a fresh 3-atom molsys::SimulationBox the tests can share: 2
    // atoms in mol1, 1 atom in mol2, all mass 1, deterministic velocities.
    molsys::SimulationBox *makeBox()
    {
        auto *box      = new molsys::SimulationBox();
        auto  molecule = molsys::Molecule();
        molecule.setNumberOfAtoms(2);

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();
        atom1->setMass(1.0);
        atom2->setMass(1.0);
        atom1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        atom2->setPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        atom1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        atom2->setVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule.setMolMass(2.0);
        molecule.addAtom(atom1);
        molecule.addAtom(atom2);

        auto molecule2 = molsys::Molecule();
        molecule2.setNumberOfAtoms(1);
        auto atom3 = std::make_shared<molsys::Atom>();
        atom3->setMass(1.0);
        atom3->setPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));
        atom3->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setMolMass(1.0);
        molecule2.addAtom(atom3);

        box->addMolecule(molecule);
        box->addMolecule(molecule2);
        box->addAtom(atom1);
        box->addAtom(atom2);
        box->addAtom(atom3);
        box->setTotalMass(3.0);
        box->calculateDegreesOfFreedom();

        return box;
    }

    using linearAlgebra::Vec3D;

    // A frequency that never fires for the (small) steps used in the tests:
    // step % never != 0 for 0 < step < never.
    constexpr size_t never = 1'000'000U;

    std::vector<Vec3D> velocitiesOf(molsys::SimulationBox &box)
    {
        std::vector<Vec3D> velocities;
        for (const auto &atom : box.getAtoms())
            velocities.push_back(atom->getVelocity());
        return velocities;
    }

    std::vector<Vec3D> positionsOf(molsys::SimulationBox &box)
    {
        std::vector<Vec3D> positions;
        for (const auto &atom : box.getAtoms())
            positions.push_back(atom->getPosition());
        return positions;
    }

    // velocities can become huge when scaling to a target temperature (the
    // fixture starts at a tiny temperature) -> compare relative to that scale
    double velocityTolerance(molsys::SimulationBox &box)
    {
        auto scale = 1.0;
        for (const auto &velocity : velocitiesOf(box))
            for (size_t i = 0; i < 3; ++i)
                scale = std::max(scale, std::abs(velocity[i]));

        return 1e-12 * scale;
    }

    // total angular momentum about the centre of mass, as reset() sees it
    Vec3D angularMomentumOf(molsys::SimulationBox &box)
    {
        box.calculateCenterOfMass();
        return box.calculateAngularMomentum(box.calculateMomentum());
    }

    // fill PhysicalData the same way the MD loop does before reset() is called
    physicalData::PhysicalData makeData(molsys::SimulationBox &box)
    {
        auto data = physicalData::PhysicalData();
        box.calculateCenterOfMass();
        data.calculateKinetics(box);
        data.calculateTemperature(box);
        return data;
    }

    void expectVec3DNear(
        const Vec3D &actual,
        const Vec3D &expected,
        double       tolerance
    )
    {
        for (size_t i = 0; i < 3; ++i)
            EXPECT_NEAR(actual[i], expected[i], tolerance) << "component " << i;
    }

    void expectVelocitiesNear(
        molsys::SimulationBox    &box,
        const std::vector<Vec3D> &expected,
        double                    tolerance
    )
    {
        const auto actual = velocitiesOf(box);
        ASSERT_EQ(actual.size(), expected.size());

        for (size_t i = 0; i < actual.size(); ++i)
        {
            SCOPED_TRACE("atom " + std::to_string(i));
            expectVec3DNear(actual[i], expected[i], tolerance);
        }
    }

    // reset() has to leave PhysicalData consistent with the SimulationBox
    // (momenta are stored in SI units in PhysicalData, internal units in
    // the box)
    void expectDataMatchesBox(
        physicalData::PhysicalData &data,
        molsys::SimulationBox      &box
    )
    {
        const auto tolerance =
            velocityTolerance(box) * std::max(1.0, box.getTotalMass());

        const auto momentum        = box.calculateMomentum();
        const auto angularMomentum = angularMomentumOf(box);

        EXPECT_NEAR(data.getTemperature(), box.calculateTemperature(), 1e-9);
        expectVec3DNear(
            data.getMomentum() * constants::S_TO_FS,
            momentum,
            tolerance
        );
        expectVec3DNear(
            data.getAngularMomentum() * constants::S_TO_FS,
            angularMomentum,
            tolerance
        );
    }

    // true if reset() at the given step modified the velocities of a fresh box
    bool resetModifiesVelocities(
        const resetKinetics::ResetKinetics &reset,
        size_t                              step
    )
    {
        auto *box    = makeBox();
        auto  data   = makeData(*box);
        auto  before = velocitiesOf(*box);

        reset.reset(step, data, *box);

        const auto modified = (before != velocitiesOf(*box));
        delete box;
        return modified;
    }

    // masses 1, 2, 4 - total mass 7
    void makeMassesNonUniform(molsys::SimulationBox &box)
    {
        const std::vector<double> masses = {1.0, 2.0, 4.0};

        for (size_t i = 0; i < masses.size(); ++i)
            box.getAtoms()[i]->setMass(masses[i]);

        box.setTotalMass(7.0);
    }
}   // namespace

TEST(TestResetKinetics, constructorStoresStepAndFrequencyParameters)
{
    resetKinetics::ResetKinetics reset(1U, 2U, 3U, 4U, 50U, 100U, 11U);
    EXPECT_EQ(reset.getNStepsTemperatureReset(), 1U);
    EXPECT_EQ(reset.getFrequencyTemperatureReset(), 2U);
    EXPECT_EQ(reset.getNStepsMomentumReset(), 3U);
    EXPECT_EQ(reset.getFrequencyMomentumReset(), 4U);
    EXPECT_EQ(reset.getNStepsForcesReset(), 11U);
}

TEST(TestResetKinetics, resetTemperatureRescalesVelocitiesAndStaysFinite)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();

    settings::ThermostatSettings::setTargetTemperature(300.0);

    // resetTemperature scales by sqrt(target / temperature) with the
    // temperature handed in by the caller (reset() takes it from PhysicalData)
    data.calculateTemperature(*box);

    resetKinetics::ResetKinetics::resetTemperature(*box, data.getTemperature());

    data.calculateTemperature(*box);
    const auto T_after = data.getTemperature();

    EXPECT_FALSE(std::isnan(T_after));
    EXPECT_FALSE(std::isinf(T_after));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureScalesFiniteTemperatureToZero)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();
    data.calculateTemperature(*box);
    settings::ThermostatSettings::setTargetTemperature(0.0);
    resetKinetics::ResetKinetics::resetTemperature(*box, data.getTemperature());

    data.calculateTemperature(*box);
    EXPECT_DOUBLE_EQ(data.getTemperature(), 0.0);
    for (const auto &atom : box->getAtoms())
        EXPECT_EQ(atom->getVelocity(), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    delete box;
}

TEST(TestResetKinetics, rejectsZeroTargetFromZeroTemperature)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms()) atom->setVelocity({0.0, 0.0, 0.0});

    settings::ThermostatSettings::setTargetTemperature(0.0);

    EXPECT_THROW_MSG(
        resetKinetics::ResetKinetics::resetTemperature(*box, 0.0),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    delete box;
}

TEST(TestResetKinetics, rejectsPositiveTargetFromZeroTemperature)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms()) atom->setVelocity({0.0, 0.0, 0.0});

    settings::ThermostatSettings::setTargetTemperature(300.0);

    EXPECT_THROW_MSG(
        resetKinetics::ResetKinetics::resetTemperature(*box, 0.0),
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
    linearAlgebra::Vec3D totalP{0.0, 0.0, 0.0};
    for (const auto &atom : box->getAtoms())
        totalP += atom->getMass() * atom->getVelocity();

    resetKinetics::ResetKinetics::resetMomentum(*box, totalP);

    linearAlgebra::Vec3D totalPAfter{0.0, 0.0, 0.0};
    for (const auto &atom : box->getAtoms())
        totalPAfter += atom->getMass() * atom->getVelocity();

    EXPECT_NEAR(totalPAfter[0], 0.0, 1e-12);
    EXPECT_NEAR(totalPAfter[1], 0.0, 1e-12);
    EXPECT_NEAR(totalPAfter[2], 0.0, 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumLeavesVelocitiesFinite)
{
    auto *box = makeBox();

    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        linearAlgebra::Vec3D(0.0, 0.0, 0.0)
    );

    for (const auto &atom : box->getAtoms())
    {
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(std::isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(std::isinf(atom->getVelocity()[i]));
        }
    }

    delete box;
}

TEST(TestResetKinetics, resetForcesZerosForcesEachStep)
{
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics reset(0U, 0U, 0U, 0U, 0U, 0U, 1U);

    // Seed atom forces with non-zero values.
    for (auto &atom : box->getAtoms())
        atom->setForce(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    reset.resetForces(0U, *box);

    for (const auto &atom : box->getAtoms())
        for (size_t i = 0; i < 3; ++i)
            EXPECT_DOUBLE_EQ(atom->getForce()[i], 0.0);

    delete box;
}

/*********************************************
 *                                           *
 * default construction (value-initialised)  *
 *                                           *
 ********************************************/

TEST(TestResetKinetics, valueInitialisedObjectHasZeroedParameters)
{
    // MDEngine holds the object as `_resetKinetics{}`; value-initialisation
    // has to zero the members instead of leaving them indeterminate
    const resetKinetics::ResetKinetics reset{};

    EXPECT_EQ(reset.getNStepsTemperatureReset(), 0U);
    EXPECT_EQ(reset.getFrequencyTemperatureReset(), 0U);
    EXPECT_EQ(reset.getNStepsMomentumReset(), 0U);
    EXPECT_EQ(reset.getFrequencyMomentumReset(), 0U);
    EXPECT_EQ(reset.getNStepsForcesReset(), 0U);
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

        settings::ThermostatSettings::setTargetTemperature(target);

        resetKinetics::ResetKinetics::resetTemperature(
            *box,
            data.getTemperature()
        );

        EXPECT_NEAR(box->calculateTemperature(), target, 1e-10 * target);

        delete box;
    }
}

TEST(TestResetKinetics, resetTemperatureScalesAllVelocitiesByCommonFactor)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const auto before    = velocitiesOf(*box);
    const auto positions = positionsOf(*box);
    const auto lambda    = ::sqrt(300.0 / data.getTemperature());

    resetKinetics::ResetKinetics::resetTemperature(*box, data.getTemperature());

    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
        expectVec3DNear(after[i], before[i] * lambda, velocityTolerance(*box));

    // only velocities are touched
    EXPECT_EQ(positionsOf(*box), positions);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureUsesPassedTemperatureNotBoxTemperature)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    // claim the system is 4 times hotter than it is - the scaling factor has
    // to follow the argument: sqrt(target / (4 T)) = 0.5 * sqrt(target / T)
    const auto before = velocitiesOf(*box);
    const auto lambda = 0.5 * ::sqrt(300.0 / data.getTemperature());

    resetKinetics::ResetKinetics::resetTemperature(
        *box,
        4.0 * data.getTemperature()
    );

    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
        expectVec3DNear(after[i], before[i] * lambda, velocityTolerance(*box));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureIsIdempotent)
{
    auto *box = makeBox();

    settings::ThermostatSettings::setTargetTemperature(300.0);

    resetKinetics::ResetKinetics::resetTemperature(
        *box,
        box->calculateTemperature()
    );
    const auto once = velocitiesOf(*box);

    // already at the target -> lambda = 1
    resetKinetics::ResetKinetics::resetTemperature(
        *box,
        box->calculateTemperature()
    );

    expectVelocitiesNear(*box, once, velocityTolerance(*box));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureCanHeatAndCool)
{
    auto      *box   = makeBox();
    auto       data  = makeData(*box);
    const auto temp0 = data.getTemperature();

    settings::ThermostatSettings::setTargetTemperature(2.0 * temp0);
    resetKinetics::ResetKinetics::resetTemperature(*box, temp0);
    EXPECT_NEAR(box->calculateTemperature(), 2.0 * temp0, 1e-10 * temp0);

    settings::ThermostatSettings::setTargetTemperature(0.25 * temp0);
    resetKinetics::ResetKinetics::resetTemperature(*box, 2.0 * temp0);
    EXPECT_NEAR(box->calculateTemperature(), 0.25 * temp0, 1e-10 * temp0);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureThrowsBeforeModifyingVelocities)
{
    auto *box = makeBox();

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const auto before = velocitiesOf(*box);

    // the box itself is hot, but the temperature handed in is zero - the
    // decision to throw is based on the argument
    EXPECT_THROW_MSG(
        resetKinetics::ResetKinetics::resetTemperature(*box, 0.0),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    EXPECT_EQ(velocitiesOf(*box), before);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureOfColdBoxWithNonZeroArgumentStaysCold)
{
    auto *box = makeBox();
    for (const auto &atom : box->getAtoms()) atom->setVelocity({0.0, 0.0, 0.0});

    settings::ThermostatSettings::setTargetTemperature(300.0);

    EXPECT_NO_THROW(resetKinetics::ResetKinetics::resetTemperature(*box, 10.0));

    for (const auto &atom : box->getAtoms())
        EXPECT_EQ(atom->getVelocity(), Vec3D(0.0, 0.0, 0.0));

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
    ASSERT_GT(norm(momentum), 1.0);

    resetKinetics::ResetKinetics::resetMomentum(*box, momentum);

    expectVec3DNear(box->calculateMomentum(), Vec3D(0.0, 0.0, 0.0), 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetMomentumShiftsAllVelocitiesByCentreOfMassVelocity)
{
    auto *box = makeBox();

    const auto before    = velocitiesOf(*box);
    const auto positions = positionsOf(*box);
    const auto momentum  = box->calculateMomentum();
    const auto vCom      = momentum / box->getTotalMass();

    resetKinetics::ResetKinetics::resetMomentum(*box, momentum);

    // velocities relative to each other stay untouched
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
        expectVec3DNear(after[i], before[i] - vCom, 1e-12);

    EXPECT_EQ(positionsOf(*box), positions);

    delete box;
}

TEST(TestResetKinetics, resetMomentumSubtractsExactlyThePassedMomentum)
{
    const std::vector<Vec3D> corrections = {
        Vec3D(0.0, 0.0, 0.0),
        Vec3D(1.0, -2.0, 3.0),
        Vec3D(-10.0, 0.5, 0.25),
    };

    for (const auto &correction : corrections)
    {
        auto *box = makeBox();
        makeMassesNonUniform(*box);

        const auto expected = box->calculateMomentum() - correction;

        resetKinetics::ResetKinetics::resetMomentum(*box, correction);

        expectVec3DNear(box->calculateMomentum(), expected, 1e-12);

        delete box;
    }
}

TEST(TestResetKinetics, resetMomentumWithZeroMomentumLeavesVelocitiesUnchanged)
{
    auto *box = makeBox();

    const auto before = velocitiesOf(*box);

    resetKinetics::ResetKinetics::resetMomentum(*box, Vec3D(0.0, 0.0, 0.0));

    EXPECT_EQ(velocitiesOf(*box), before);

    delete box;
}

TEST(TestResetKinetics, resetMomentumIsIdempotent)
{
    auto *box = makeBox();

    resetKinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());
    const auto once = velocitiesOf(*box);

    resetKinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());

    expectVelocitiesNear(*box, once, 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetMomentumRemovesPureTranslationCompletely)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(Vec3D(0.3, -0.7, 1.1));

    resetKinetics::ResetKinetics::resetMomentum(*box, box->calculateMomentum());

    for (const auto &atom : box->getAtoms())
        expectVec3DNear(atom->getVelocity(), Vec3D(0.0, 0.0, 0.0), 1e-12);

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
    ASSERT_GT(norm(angularMomentum), 1e-3);

    resetKinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    expectVec3DNear(angularMomentumOf(*box), Vec3D(0.0, 0.0, 0.0), 1e-12);

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
    ASSERT_GT(norm(angularMomentum), 1e-3);

    resetKinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    expectVec3DNear(angularMomentumOf(*box), Vec3D(0.0, 0.0, 0.0), 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumPreservesLinearMomentum)
{
    auto *box = makeBox();
    makeMassesNonUniform(*box);

    const auto momentum        = box->calculateMomentum();
    const auto angularMomentum = angularMomentumOf(*box);

    resetKinetics::ResetKinetics::resetAngularMomentum(*box, angularMomentum);

    expectVec3DNear(box->calculateMomentum(), momentum, 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumSubtractsExactlyThePassedValue)
{
    const auto scaled = [](const Vec3D &vec, double factor)
    { return vec * factor; };

    auto      *reference        = makeBox();
    const auto angularMomentum0 = angularMomentumOf(*reference);
    delete reference;

    const std::vector<Vec3D> corrections = {
        Vec3D(0.0, 0.0, 0.0),
        scaled(angularMomentum0, 0.5),
        Vec3D(0.3, -0.2, 0.5),
    };

    for (const auto &correction : corrections)
    {
        auto *box = makeBox();

        const auto expected = angularMomentumOf(*box) - correction;

        resetKinetics::ResetKinetics::resetAngularMomentum(*box, correction);

        expectVec3DNear(angularMomentumOf(*box), expected, 1e-12);

        delete box;
    }
}

TEST(TestResetKinetics, resetAngularMomentumWithZeroLeavesVelocitiesUnchanged)
{
    auto *box = makeBox();

    const auto before = velocitiesOf(*box);

    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        Vec3D(0.0, 0.0, 0.0)
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
    const Vec3D omega(0.1, 0.2, -0.3);
    const auto  centerOfMass = box->getCenterOfMass();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(cross(omega, atom->getPosition() - centerOfMass));

    expectVec3DNear(box->calculateMomentum(), Vec3D(0.0, 0.0, 0.0), 1e-12);
    ASSERT_GT(norm(angularMomentumOf(*box)), 1e-3);

    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    for (const auto &atom : box->getAtoms())
        expectVec3DNear(atom->getVelocity(), Vec3D(0.0, 0.0, 0.0), 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumLeavesPureTranslationUntouched)
{
    auto *box = makeBox();

    for (const auto &atom : box->getAtoms())
        atom->setVelocity(Vec3D(0.3, -0.7, 1.1));

    const auto before = velocitiesOf(*box);

    // a translating body has no angular momentum about its centre of mass
    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    expectVelocitiesNear(*box, before, 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumRefreshesCentreOfMass)
{
    auto *box = makeBox();

    // makeBox never calculated the centre of mass -> stale value
    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        Vec3D(0.0, 0.0, 0.0)
    );

    // positions (0,0,0), (1,0,0), (0,1,0), equal masses
    expectVec3DNear(
        box->getCenterOfMass(),
        Vec3D(1.0 / 3.0, 1.0 / 3.0, 0.0),
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetAngularMomentumDoesNotIncreaseKineticEnergy)
{
    auto *box = makeBox();

    const auto T_before = box->calculateTemperature();

    resetKinetics::ResetKinetics::resetAngularMomentum(
        *box,
        angularMomentumOf(*box)
    );

    // the removed rotational energy is 1/2 omega . L > 0
    EXPECT_LT(box->calculateTemperature(), T_before);
    EXPECT_GT(box->calculateTemperature(), 0.0);

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

    settings::ThermostatSettings::setTargetTemperature(300.0);

    // reset() must be callable on a const object
    const resetKinetics::ResetKinetics
        reset(0U, never, 0U, never, 0U, never, 1U);

    const auto velocities      = velocitiesOf(*box);
    const auto temp            = data.getTemperature();
    const auto momentum        = data.getMomentum();
    const auto angularMomentum = data.getAngularMomentum();

    reset.reset(7U, data, *box);

    EXPECT_EQ(velocitiesOf(*box), velocities);
    EXPECT_DOUBLE_EQ(data.getTemperature(), temp);
    expectVec3DNear(
        data.getMomentum() * constants::S_TO_FS,
        momentum * constants::S_TO_FS,
        1e-12
    );
    expectVec3DNear(
        data.getAngularMomentum() * constants::S_TO_FS,
        angularMomentum * constants::S_TO_FS,
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetWithoutScheduledResetTakesDataAsSourceOfTruth)
{
    auto *box  = makeBox();
    auto  data = physicalData::PhysicalData();

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics
        reset(0U, never, 0U, never, 0U, never, 1U);

    // values in PhysicalData deliberately do not match the box
    const Vec3D momentum(1.0, 2.0, 3.0);
    const Vec3D angularMomentum(-4.0, 5.0, -6.0);
    data.setTemperature(123.0);
    data.setMomentum(momentum * constants::FS_TO_S);
    data.setAngularMomentum(angularMomentum * constants::FS_TO_S);

    const auto velocities = velocitiesOf(*box);

    reset.reset(7U, data, *box);

    // nothing recalculated - the unit conversions S_TO_FS / FS_TO_S cancel
    EXPECT_EQ(velocitiesOf(*box), velocities);
    EXPECT_DOUBLE_EQ(data.getTemperature(), 123.0);
    expectVec3DNear(data.getMomentum() * constants::S_TO_FS, momentum, 1e-12);
    expectVec3DNear(
        data.getAngularMomentum() * constants::S_TO_FS,
        angularMomentum,
        1e-12
    );

    delete box;
}

TEST(TestResetKinetics, resetTemperatureBranchScalesAndRemovesMomentum)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reset(0U, 7U, 0U, never, 0U, never, 1U);

    const auto before = velocitiesOf(*box);
    const auto vCom   = box->calculateMomentum() / box->getTotalMass();
    const auto lambda = ::sqrt(300.0 / data.getTemperature());

    reset.reset(7U, data, *box);

    // hard scaling first, then removal of the (scaled) centre of mass motion
    // -> v_i' = lambda * (v_i - v_com)
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
    {
        expectVec3DNear(
            after[i],
            (before[i] - vCom) * lambda,
            velocityTolerance(*box)
        );
    }

    expectVec3DNear(
        box->calculateMomentum(),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );

    // the removed centre of mass motion carried kinetic energy
    EXPECT_LT(data.getTemperature(), 300.0);
    EXPECT_GT(data.getTemperature(), 0.0);

    // angular momentum was not scheduled
    EXPECT_GT(norm(angularMomentumOf(*box)), 1e-3);

    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetMomentumBranchOnlyShiftsVelocities)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reset(0U, never, 0U, 7U, 0U, never, 1U);

    const auto before = velocitiesOf(*box);
    const auto vCom   = box->calculateMomentum() / box->getTotalMass();

    reset.reset(7U, data, *box);

    // no temperature rescaling, only v_i' = v_i - v_com
    const auto after = velocitiesOf(*box);
    for (size_t i = 0; i < before.size(); ++i)
        expectVec3DNear(after[i], before[i] - vCom, 1e-12);

    expectVec3DNear(box->calculateMomentum(), Vec3D(0.0, 0.0, 0.0), 1e-12);
    EXPECT_NEAR(data.getMomentum()[0] * constants::S_TO_FS, 0.0, 1e-12);

    // the temperature in data follows the (unscaled) box
    EXPECT_GT(std::abs(data.getTemperature() - 300.0), 1e-3);
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetAngularBranchOnlyRemovesRotation)
{
    auto *box  = makeBox();
    auto  data = makeData(*box);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reset(0U, never, 0U, never, 0U, 7U, 1U);

    const auto momentum = box->calculateMomentum();
    const auto T_before = data.getTemperature();

    reset.reset(7U, data, *box);

    expectVec3DNear(angularMomentumOf(*box), Vec3D(0.0, 0.0, 0.0), 1e-12);

    // linear momentum is a conserved quantity of the rotation correction, and
    // has to be reported unchanged (in SI units) in PhysicalData
    expectVec3DNear(box->calculateMomentum(), momentum, 1e-12);
    expectVec3DNear(data.getMomentum() * constants::S_TO_FS, momentum, 1e-12);
    expectVec3DNear(
        data.getAngularMomentum() * constants::S_TO_FS,
        Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    // the rotational energy is gone -> data has to carry the new temperature
    EXPECT_LT(data.getTemperature(), T_before);
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetTemperatureAndMomentumDueTogetherActOnlyOnce)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *boxTemperature  = makeBox();
    auto  dataTemperature = makeData(*boxTemperature);
    auto *boxBoth         = makeBox();
    auto  dataBoth        = makeData(*boxBoth);

    const resetKinetics::ResetKinetics
        onlyTemperature(0U, 7U, 0U, never, 0U, never, 1U);
    const resetKinetics::ResetKinetics
        temperatureAndMomentum(0U, 7U, 0U, 7U, 0U, never, 1U);

    onlyTemperature.reset(7U, dataTemperature, *boxTemperature);
    temperatureAndMomentum.reset(7U, dataBoth, *boxBoth);

    // the temperature reset already includes the momentum reset - a second
    // correction with the stale momentum would shift the velocities again
    EXPECT_EQ(velocitiesOf(*boxBoth), velocitiesOf(*boxTemperature));
    EXPECT_DOUBLE_EQ(
        dataBoth.getTemperature(),
        dataTemperature.getTemperature()
    );

    delete boxTemperature;
    delete boxBoth;
}

TEST(TestResetKinetics, resetAllBranchesMatchSequentialStaticCalls)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box  = makeBox();
    auto  data = makeData(*box);

    const resetKinetics::ResetKinetics reset(0U, 7U, 0U, 7U, 0U, 7U, 1U);

    // reference: temperature -> momentum -> angular momentum
    auto *expectedBox = makeBox();
    expectedBox->calculateCenterOfMass();
    resetKinetics::ResetKinetics::resetTemperature(
        *expectedBox,
        expectedBox->calculateTemperature()
    );
    resetKinetics::ResetKinetics::resetMomentum(
        *expectedBox,
        expectedBox->calculateMomentum()
    );
    resetKinetics::ResetKinetics::resetAngularMomentum(
        *expectedBox,
        angularMomentumOf(*expectedBox)
    );

    reset.reset(7U, data, *box);

    expectVelocitiesNear(
        *box,
        velocitiesOf(*expectedBox),
        velocityTolerance(*box)
    );
    expectVec3DNear(
        box->calculateMomentum(),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectVec3DNear(
        angularMomentumOf(*box),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectDataMatchesBox(data, *box);

    // data reports the final state: no momentum, no angular momentum
    expectVec3DNear(
        data.getMomentum() * constants::S_TO_FS,
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectVec3DNear(
        data.getAngularMomentum() * constants::S_TO_FS,
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );

    delete box;
    delete expectedBox;
}

TEST(TestResetKinetics, resetTemperatureAndAngularDueTogetherKeepMomentumZero)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box  = makeBox();
    auto  data = makeData(*box);

    const resetKinetics::ResetKinetics reset(0U, 7U, 0U, never, 0U, 7U, 1U);

    reset.reset(7U, data, *box);

    expectVec3DNear(
        box->calculateMomentum(),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectVec3DNear(
        angularMomentumOf(*box),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectDataMatchesBox(data, *box);

    delete box;
}

TEST(TestResetKinetics, resetMomentumAndAngularDueTogetherZeroBoth)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box  = makeBox();
    auto  data = makeData(*box);

    const resetKinetics::ResetKinetics reset(0U, never, 0U, 7U, 0U, 7U, 1U);

    reset.reset(7U, data, *box);

    expectVec3DNear(box->calculateMomentum(), Vec3D(0.0, 0.0, 0.0), 1e-12);
    expectVec3DNear(angularMomentumOf(*box), Vec3D(0.0, 0.0, 0.0), 1e-12);
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
    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics
        reset(5U, 1000U, 0U, never, 0U, never, 1U);

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
    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics
        reset(0U, never, 5U, 1000U, 0U, never, 1U);

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
    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics
        reset(0U, never, 0U, never, 5U, 1000U, 1U);

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
    settings::ThermostatSettings::setTargetTemperature(300.0);

    // step 0: 0 % frequency == 0 for every non-zero frequency
    auto *box  = makeBox();
    auto  data = makeData(*box);

    const resetKinetics::ResetKinetics
        reset(0U, never, 0U, never, 0U, never, 1U);

    reset.reset(0U, data, *box);

    expectVec3DNear(
        box->calculateMomentum(),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
    );
    expectVec3DNear(
        angularMomentumOf(*box),
        Vec3D(0.0, 0.0, 0.0),
        velocityTolerance(*box)
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

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reset(0U, 7U, 0U, never, 0U, never, 1U);

    const auto velocities = velocitiesOf(*box);

    EXPECT_THROW_MSG(
        reset.reset(7U, data, *box),
        exc::UserInputException,
        "Cannot rescale a zero-temperature system. Initialize velocities first."
    );

    EXPECT_EQ(velocitiesOf(*box), velocities);

    delete box;
}

TEST(TestResetKinetics, resetOfMomentumAndAngularDoesNotNeedTemperature)
{
    auto *box = makeBox();
    for (const auto &atom : box->getAtoms()) atom->setVelocity({0.0, 0.0, 0.0});

    auto data = makeData(*box);
    ASSERT_DOUBLE_EQ(data.getTemperature(), 0.0);

    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reset(0U, never, 0U, 7U, 0U, 7U, 1U);

    EXPECT_NO_THROW(reset.reset(7U, data, *box));

    for (const auto &atom : box->getAtoms())
        EXPECT_EQ(atom->getVelocity(), Vec3D(0.0, 0.0, 0.0));
    EXPECT_DOUBLE_EQ(data.getTemperature(), 0.0);

    delete box;
}

TEST(TestResetKinetics, resetIsIdempotentForMomentumAndAngularMomentum)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    auto *box  = makeBox();
    auto  data = makeData(*box);

    const resetKinetics::ResetKinetics reset(0U, never, 0U, 7U, 0U, 7U, 1U);

    reset.reset(7U, data, *box);
    const auto once = velocitiesOf(*box);

    data = makeData(*box);
    reset.reset(7U, data, *box);

    expectVelocitiesNear(*box, once, 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetDoesNotCarryStateBetweenCalls)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

    const resetKinetics::ResetKinetics reused(0U, 7U, 0U, 7U, 0U, 7U, 1U);

    // first call with a very different system
    auto *boxA = makeBox();
    for (const auto &atom : boxA->getAtoms()) atom->scaleVelocity(5.0);
    auto dataA = makeData(*boxA);
    reused.reset(7U, dataA, *boxA);

    // second call with the same object must behave like a fresh object
    auto *boxB  = makeBox();
    auto  dataB = makeData(*boxB);
    reused.reset(7U, dataB, *boxB);

    const resetKinetics::ResetKinetics fresh(0U, 7U, 0U, 7U, 0U, 7U, 1U);
    auto                              *boxC  = makeBox();
    auto                               dataC = makeData(*boxC);
    fresh.reset(7U, dataC, *boxC);

    EXPECT_EQ(velocitiesOf(*boxB), velocitiesOf(*boxC));
    EXPECT_DOUBLE_EQ(dataB.getTemperature(), dataC.getTemperature());
    EXPECT_EQ(dataB.getMomentum(), dataC.getMomentum());
    EXPECT_EQ(dataB.getAngularMomentum(), dataC.getAngularMomentum());

    delete boxA;
    delete boxB;
    delete boxC;
}

TEST(TestResetKinetics, resetForcesRespectsStepFrequency)
{
    auto                              *box = makeBox();
    const resetKinetics::ResetKinetics reset(0U, 0U, 0U, 0U, 0U, 0U, 3U);

    for (auto &atom : box->getAtoms()) atom->setForce(Vec3D(1.0, 2.0, 3.0));

    // step 1 and 2 are not multiples of 3
    reset.resetForces(1U, *box);
    reset.resetForces(2U, *box);
    for (const auto &atom : box->getAtoms())
        EXPECT_EQ(atom->getForce(), Vec3D(1.0, 2.0, 3.0));

    reset.resetForces(3U, *box);
    for (const auto &atom : box->getAtoms())
        expectVec3DNear(atom->getForce(), Vec3D(0.0, 0.0, 0.0), 1e-12);

    delete box;
}

TEST(TestResetKinetics, resetAngularBranchUsesFreshCentreOfMass)
{
    settings::ThermostatSettings::setTargetTemperature(300.0);

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

    const resetKinetics::ResetKinetics reset(0U, never, 0U, never, 0U, 7U, 1U);

    reset.reset(7U, data, *box);

    expectVec3DNear(angularMomentumOf(*box), Vec3D(0.0, 0.0, 0.0), 1e-12);
    expectVec3DNear(
        data.getAngularMomentum() * constants::S_TO_FS,
        Vec3D(0.0, 0.0, 0.0),
        1e-12
    );

    delete box;
}
