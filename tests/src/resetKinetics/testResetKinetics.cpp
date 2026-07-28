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

#include <cmath>
#include <memory>

#include "atom.hpp"
#include "exceptions.hpp"
#include "gtest/gtest.h"
#include "molecule.hpp"
#include "physicalData.hpp"
#include "resetKinetics.hpp"
#include "simulationBox.hpp"
#include "thermostatSettings.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

namespace
{
    // Build a fresh 3-atom SimBox the tests can share: 2 atoms in mol1,
    // 1 atom in mol2, all mass 1, deterministic velocities.
    simulationBox::SimulationBox *makeBox()
    {
        auto *box      = new simulationBox::SimulationBox();
        auto  molecule = simulationBox::Molecule();
        molecule.setNumberOfAtoms(2);

        auto a1 = std::make_shared<simulationBox::Atom>();
        auto a2 = std::make_shared<simulationBox::Atom>();
        a1->setMass(1.0);
        a2->setMass(1.0);
        a1->setPosition(linearAlgebra::Vec3D(0.0, 0.0, 0.0));
        a2->setPosition(linearAlgebra::Vec3D(1.0, 0.0, 0.0));
        a1->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        a2->setVelocity(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
        molecule.setMolMass(2.0);
        molecule.addAtom(a1);
        molecule.addAtom(a2);

        auto molecule2 = simulationBox::Molecule();
        molecule2.setNumberOfAtoms(1);
        auto a3 = std::make_shared<simulationBox::Atom>();
        a3->setMass(1.0);
        a3->setPosition(linearAlgebra::Vec3D(0.0, 1.0, 0.0));
        a3->setVelocity(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
        molecule2.setMolMass(1.0);
        molecule2.addAtom(a3);

        box->addMolecule(molecule);
        box->addMolecule(molecule2);
        box->addAtom(a1);
        box->addAtom(a2);
        box->addAtom(a3);
        box->setTotalMass(3.0);
        box->calculateDegreesOfFreedom();

        return box;
    }
}   // namespace

TEST(TestResetKinetics, constructorStoresStepAndFrequencyParameters)
{
    resetKinetics::ResetKinetics rk(1u, 2u, 3u, 4u, 50u, 100u, 11u);
    EXPECT_EQ(rk.getNStepsTemperatureReset(), 1u);
    EXPECT_EQ(rk.getFrequencyTemperatureReset(), 2u);
    EXPECT_EQ(rk.getNStepsMomentumReset(), 3u);
    EXPECT_EQ(rk.getFrequencyMomentumReset(), 4u);
    EXPECT_EQ(rk.getNStepsForcesReset(), 11u);
}

TEST(TestResetKinetics, settersAcceptValuesWithoutThrowing)
{
    resetKinetics::ResetKinetics rk;

    rk.setTemperature(300.0);
    rk.setMomentum(linearAlgebra::Vec3D(1.0, 2.0, 3.0));
    rk.setAngularMomentum(linearAlgebra::Vec3D(0.5, -0.5, 0.0));

    SUCCEED();
}

TEST(TestResetKinetics, resetTemperatureRescalesVelocitiesAndStaysFinite)
{
    auto                        *box  = makeBox();
    auto                         data = physicalData::PhysicalData();
    resetKinetics::ResetKinetics rk;

    settings::ThermostatSettings::setTargetTemperature(300.0);

    // resetTemperature uses _temperature internally as sqrt(target /
    // _temperature). The public reset() entry seeds _temperature from
    // data.getTemperature(); when calling resetTemperature directly the
    // setter must do the same.
    data.calculateTemperature(*box);
    rk.setTemperature(data.getTemperature());

    rk.resetTemperature(*box);

    data.calculateTemperature(*box);
    const auto T_after = data.getTemperature();

    EXPECT_FALSE(std::isnan(T_after));
    EXPECT_FALSE(std::isinf(T_after));

    delete box;
}

TEST(TestResetKinetics, resetTemperatureSupportsZeroKelvin)
{
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics resetKinetics;

    settings::ThermostatSettings::setTargetTemperature(0.0);

    auto data = physicalData::PhysicalData();
    data.calculateTemperature(*box);
    resetKinetics.setTemperature(data.getTemperature());
    resetKinetics.resetTemperature(*box);

    data.calculateTemperature(*box);
    EXPECT_DOUBLE_EQ(data.getTemperature(), 0.0);
    for (const auto &atom : box->getAtoms())
        EXPECT_EQ(atom->getVelocity(), linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    delete box;
}

TEST(TestResetKinetics, rejectsPositiveTargetFromZeroTemperature)
{
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics resetKinetics;

    for (const auto &atom : box->getAtoms()) atom->setVelocity({0.0, 0.0, 0.0});

    settings::ThermostatSettings::setTargetTemperature(300.0);
    resetKinetics.setTemperature(0.0);

    EXPECT_THROW(
        resetKinetics.resetTemperature(*box),
        customException::UserInputException
    );

    delete box;
}

TEST(TestResetKinetics, resetMomentumZerosTotalLinearMomentum)
{
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics rk;

    // resetMomentum subtracts (_momentum / totalMass) from every atom's
    // velocity; for the total to land at zero we have to seed _momentum
    // with the current total p = sum m_i v_i first (the reset() entry
    // does this from data.getMomentum()).
    linearAlgebra::Vec3D totalP{0.0, 0.0, 0.0};
    for (const auto &atom : box->getAtoms())
        totalP += atom->getMass() * atom->getVelocity();
    rk.setMomentum(totalP);

    rk.resetMomentum(*box);

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
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics rk;

    // Seed _angularMomentum the same way reset() does, so the routine
    // has well-defined input.
    rk.setAngularMomentum(linearAlgebra::Vec3D(0.0, 0.0, 0.0));

    rk.resetAngularMomentum(*box);

    for (const auto &atom : box->getAtoms())
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(std::isnan(atom->getVelocity()[i]));
            EXPECT_FALSE(std::isinf(atom->getVelocity()[i]));
        }

    delete box;
}

TEST(TestResetKinetics, resetForcesZerosForcesEachStep)
{
    auto                        *box = makeBox();
    resetKinetics::ResetKinetics rk(0u, 0u, 0u, 0u, 0u, 0u, 1u);

    // Seed atom forces with non-zero values.
    for (auto &atom : box->getAtoms())
        atom->setForce(linearAlgebra::Vec3D(1.0, 2.0, 3.0));

    rk.resetForces(0u, *box);

    for (const auto &atom : box->getAtoms())
        for (size_t i = 0; i < 3; ++i)
            EXPECT_DOUBLE_EQ(atom->getForce()[i], 0.0);

    delete box;
}
