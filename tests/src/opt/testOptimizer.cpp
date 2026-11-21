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

#include "atom.hpp"
#include "convergence.hpp"
#include "exceptions.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "steepestDescent.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace
{
    struct Sample
    {
        EnergyPerMol energy;
        Force3D      force0;
        Force3D      force1;
        Pos3D        pos0;
        Pos3D        pos1;
    };

    // Build a fresh box+physData pair, set the sample state on them, and call
    // updateHistory(). Used to seed the optimizer's deques deterministically.
    void pushSample(opt::SteepestDescent &opt, const Sample &sample)
    {
        auto box     = std::make_shared<molsys::SimulationBox>();
        auto physDat = std::make_shared<physicalData::PhysicalData>();

        auto atom1 = std::make_shared<molsys::Atom>();
        auto atom2 = std::make_shared<molsys::Atom>();

        atom1->setPosition(sample.pos0);
        atom2->setPosition(sample.pos1);
        atom1->setForce(sample.force0);
        atom2->setForce(sample.force1);

        box->addAtom(atom1);
        box->addAtom(atom2);

        physDat->set_kineticEnergy(sample.energy);

        opt.setSimulationBox(box);
        opt.setPhysicalData(physDat);
        opt.updateHistory();
    }
}   // namespace

/* ---------- constructor ---------- */

TEST(TestOptimizer, maxHistoryLengthIsTwoForSteepestDescent)
{
    const opt::SteepestDescent opt;
    EXPECT_EQ(opt.maxHistoryLength(), 2U);
}

/* ---------- getHistoryIndex ---------- */

TEST(TestOptimizer, getHistoryIndexThrowsOnNonNegativeOffset)
{
    const opt::SteepestDescent opt;
    EXPECT_THROW_MSG(
        (void) opt.getHistoryIndex(0),
        exc::OptException,
        "Offset must be negative to access history in the past"
    );
    EXPECT_THROW_MSG(
        (void) opt.getHistoryIndex(1),
        exc::OptException,
        "Offset must be negative to access history in the past"
    );
}

/* ---------- updateHistory + getters ---------- */

TEST(TestOptimizer, updateHistoryAppendsAndGettersReturnLast)
{
    opt::SteepestDescent opt;

    pushSample(
        opt,
        {.energy = 1.0 * kcal_per_mol,
         .force0 = linalg::Vector3D{1.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .force1 = linalg::Vector3D{0.0, 2.0, 0.0} * kcal_per_mol / angstrom,
         .pos0   = linalg::Vector3D{0.1, 0.2, 0.3} * angstrom,
         .pos1   = linalg::Vector3D{0.4, 0.5, 0.6} * angstrom}
    );

    EXPECT_DOUBLE_EQ(opt.getEnergy().numerical_value_in(kcal_per_mol), 1.0);
    EXPECT_DOUBLE_EQ(opt.getEnergy(-1).numerical_value_in(kcal_per_mol), 1.0);
    EXPECT_DOUBLE_EQ(
        opt.getMaxForce().numerical_value_in(kcal_per_mol / angstrom),
        2.0
    );
    EXPECT_DOUBLE_EQ(
        opt.getMaxForce(-1).numerical_value_in(kcal_per_mol / angstrom),
        2.0
    );
    EXPECT_GT(
        opt.getRMSForce().numerical_value_in(kcal_per_mol / angstrom),
        0.0
    );
    EXPECT_DOUBLE_EQ(
        opt.getRMSForce().numerical_value_in(kcal_per_mol / angstrom),
        opt.getRMSForce(-1).numerical_value_in(kcal_per_mol / angstrom)
    );

    const auto forces    = opt.getForces();
    const auto positions = opt.getPositions();
    ASSERT_EQ(forces.size(), 2U);
    ASSERT_EQ(positions.size(), 2U);
    EXPECT_EQ(
        forces[0],
        linalg::Vec3D(1.0, 0.0, 0.0) * kcal_per_mol / angstrom
    );
    EXPECT_EQ(positions[1], linalg::Vec3D(0.4, 0.5, 0.6) * angstrom);

    const auto forcesOff    = opt.getForces(-1);
    const auto positionsOff = opt.getPositions(-1);
    EXPECT_EQ(forcesOff, forces);
    EXPECT_EQ(positionsOff, positions);
}

TEST(TestOptimizer, updateHistoryTrimsToMaxHistoryLength)
{
    opt::SteepestDescent opt;
    const auto           maxLen = opt.maxHistoryLength();

    for (size_t i = 0; i < maxLen + 3; ++i)
    {
        pushSample(
            opt,
            {.energy = static_cast<double>(i) * kcal_per_mol,
             .force0 = linalg::Vector3D{static_cast<double>(i), 0.0, 0.0} *
                       kcal_per_mol / angstrom,
             .force1 = linalg::Vector3D{0.0, static_cast<double>(i), 0.0} *
                       kcal_per_mol / angstrom,
             .pos0 =
                 linalg::Vector3D{static_cast<double>(i), 0.0, 0.0} * angstrom,
             .pos1 =
                 linalg::Vector3D{0.0, static_cast<double>(i), 0.0} * angstrom}
        );
    }

    // After trimming, the oldest entry accessible via -maxLen is just behind
    // the most recent. Accessing past that with index -(maxLen+1) underflows
    // size_t and reads garbage, so we don't reach for it; instead we verify
    // that the front entries match the most recently pushed window.
    const auto latest = maxLen + 3 - 1;
    EXPECT_DOUBLE_EQ(
        opt.getEnergy().numerical_value_in(kcal_per_mol),
        double(latest)
    );
    EXPECT_DOUBLE_EQ(
        opt.getEnergy(-1).numerical_value_in(kcal_per_mol),
        double(latest)
    );
    EXPECT_DOUBLE_EQ(
        opt.getEnergy(-int(maxLen)).numerical_value_in(kcal_per_mol),
        double(latest - (maxLen - 1))
    );
}

/* ---------- convergence ---------- */

TEST(TestOptimizer, setAndGetConvergenceRoundTrips)
{
    opt::SteepestDescent opt;
    opt::Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4 * kcal_per_mol,
        1.0e-3 * kcal_per_mol / angstrom,
        1.0e-3 * kcal_per_mol / angstrom,
        ConvStrategy::RIGOROUS
    );

    opt.setConvergence(conv);
    EXPECT_EQ(opt.getConvergence().getEnConvStrategy(), ConvStrategy::RIGOROUS);

    // Non-const getConvergence() returns a reference — mutating via it should
    // be observable on subsequent reads.
    auto &ref = opt.getConvergence();
    ref.calcForceConvergence(
        1.0 * kcal_per_mol / angstrom,
        1.0 * kcal_per_mol / angstrom
    );
    EXPECT_DOUBLE_EQ(
        opt.getConvergence().getAbsMaxForce().numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.0
    );
}

TEST(TestOptimizer, hasConvergedReturnsTrueForFlatEnergyAndZeroForces)
{
    opt::SteepestDescent opt;
    opt::Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4 * kcal_per_mol,
        1.0e-3 * kcal_per_mol / angstrom,
        1.0e-3 * kcal_per_mol / angstrom,
        ConvStrategy::RIGOROUS
    );
    opt.setConvergence(conv);

    // Two history entries with identical energy and zero forces → converged.
    pushSample(
        opt,
        {.energy = 1.0 * kcal_per_mol,
         .force0 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .force1 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .pos0   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom,
         .pos1   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom}
    );
    pushSample(
        opt,
        {.energy = 1.0 * kcal_per_mol,
         .force0 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .force1 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .pos0   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom,
         .pos1   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom}
    );

    EXPECT_TRUE(opt.hasConverged());
}

TEST(TestOptimizer, hasConvergedReturnsFalseForLargeForce)
{
    opt::SteepestDescent opt;
    opt::Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4 * kcal_per_mol,
        1.0e-3 * kcal_per_mol / angstrom,
        1.0e-3 * kcal_per_mol / angstrom,
        ConvStrategy::RIGOROUS
    );
    opt.setConvergence(conv);

    pushSample(
        opt,
        {.energy = 1.0 * kcal_per_mol,
         .force0 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .force1 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .pos0   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom,
         .pos1   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom}
    );
    pushSample(
        opt,
        {.energy = 1.0 * kcal_per_mol,
         .force0 = linalg::Vector3D{1.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .force1 = linalg::Vector3D{0.0, 0.0, 0.0} * kcal_per_mol / angstrom,
         .pos0   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom,
         .pos1   = linalg::Vector3D{0.0, 0.0, 0.0} * angstrom}
    );

    EXPECT_FALSE(opt.hasConverged());
}

/* ---------- old physical data setter ---------- */

TEST(TestOptimizer, setPhysicalDataOldStoresPointer)
{
    opt::SteepestDescent opt;
    auto                 phys = std::make_shared<physicalData::PhysicalData>();
    phys->set_kineticEnergy(0.42 * kcal_per_mol);
    EXPECT_NO_THROW(opt.setPhysicalDataOld(phys));
}
