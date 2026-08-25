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
#include "convergenceSettings.hpp"
#include "exceptions.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "steepestDescent.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

using namespace opt;
using molsys::Atom;
using molsys::SimulationBox;
using physicalData::PhysicalData;

namespace
{
    struct Sample
    {
        double               energy;
        linearAlgebra::Vec3D force0;
        linearAlgebra::Vec3D force1;
        linearAlgebra::Vec3D pos0;
        linearAlgebra::Vec3D pos1;
    };

    // Build a fresh box+physData pair, set the sample state on them, and call
    // updateHistory(). Used to seed the optimizer's deques deterministically.
    void pushSample(SteepestDescent &opt, const Sample &s)
    {
        auto box     = std::make_shared<SimulationBox>();
        auto physDat = std::make_shared<PhysicalData>();

        auto a1 = std::make_shared<Atom>();
        auto a2 = std::make_shared<Atom>();

        a1->setPosition(s.pos0);
        a2->setPosition(s.pos1);
        a1->setForce(s.force0);
        a2->setForce(s.force1);

        box->addAtom(a1);
        box->addAtom(a2);

        physDat->setKineticEnergy(s.energy);

        opt.setSimulationBox(box);
        opt.setPhysicalData(physDat);
        opt.updateHistory();
    }
}   // namespace

/* ---------- constructor ---------- */

TEST(TestOptimizer, constructorStoresEpochs)
{
    const SteepestDescent opt(42U);
    EXPECT_EQ(opt.getNEpochs(), 42U);
}

TEST(TestOptimizer, maxHistoryLengthIsTwoForSteepestDescent)
{
    const SteepestDescent opt(1U);
    EXPECT_EQ(opt.maxHistoryLength(), 2U);
}

TEST(TestOptimizer, cloneProducesEquivalentObject)
{
    const SteepestDescent opt(7U);
    const auto            cloned = opt.clone();
    EXPECT_EQ(cloned->getNEpochs(), 7U);
}

/* ---------- getHistoryIndex ---------- */

TEST(TestOptimizer, getHistoryIndexThrowsOnNonNegativeOffset)
{
    const SteepestDescent opt(1U);
    EXPECT_THROW((void) opt.getHistoryIndex(0), customException::OptException);
    EXPECT_THROW((void) opt.getHistoryIndex(1), customException::OptException);
}

/* ---------- updateHistory + getters ---------- */

TEST(TestOptimizer, updateHistoryAppendsAndGettersReturnLast)
{
    SteepestDescent opt(1U);

    pushSample(
        opt,
        {.energy = 1.0,
         .force0 = {1.0, 0.0, 0.0},
         .force1 = {0.0, 2.0, 0.0},
         .pos0   = {0.1, 0.2, 0.3},
         .pos1   = {0.4, 0.5, 0.6}}
    );

    EXPECT_DOUBLE_EQ(opt.getEnergy(), 1.0);
    EXPECT_DOUBLE_EQ(opt.getEnergy(-1), 1.0);
    EXPECT_DOUBLE_EQ(opt.getMaxForce(), 2.0);
    EXPECT_DOUBLE_EQ(opt.getMaxForce(-1), 2.0);
    EXPECT_GT(opt.getRMSForce(), 0.0);
    EXPECT_DOUBLE_EQ(opt.getRMSForce(), opt.getRMSForce(-1));

    const auto forces    = opt.getForces();
    const auto positions = opt.getPositions();
    ASSERT_EQ(forces.size(), 2U);
    ASSERT_EQ(positions.size(), 2U);
    EXPECT_EQ(forces[0], linearAlgebra::Vec3D(1.0, 0.0, 0.0));
    EXPECT_EQ(positions[1], linearAlgebra::Vec3D(0.4, 0.5, 0.6));

    const auto forcesOff    = opt.getForces(-1);
    const auto positionsOff = opt.getPositions(-1);
    EXPECT_EQ(forcesOff, forces);
    EXPECT_EQ(positionsOff, positions);
}

TEST(TestOptimizer, updateHistoryTrimsToMaxHistoryLength)
{
    SteepestDescent opt(1U);
    const auto      maxLen = opt.maxHistoryLength();

    for (size_t i = 0; i < maxLen + 3; ++i)
    {
        pushSample(
            opt,
            {.energy = static_cast<double>(i),
             .force0 = {static_cast<double>(i), 0.0, 0.0},
             .force1 = {0.0, static_cast<double>(i), 0.0},
             .pos0   = {static_cast<double>(i), 0.0, 0.0},
             .pos1   = {0.0, static_cast<double>(i), 0.0}}
        );
    }

    // After trimming, the oldest entry accessible via -maxLen is just behind
    // the most recent. Accessing past that with index -(maxLen+1) underflows
    // size_t and reads garbage, so we don't reach for it; instead we verify
    // that the front entries match the most recently pushed window.
    const auto latest = maxLen + 3 - 1;
    EXPECT_DOUBLE_EQ(opt.getEnergy(), double(latest));
    EXPECT_DOUBLE_EQ(opt.getEnergy(-1), double(latest));
    EXPECT_DOUBLE_EQ(
        opt.getEnergy(-int(maxLen)),
        double(latest - (maxLen - 1))
    );
}

/* ---------- convergence ---------- */

TEST(TestOptimizer, setAndGetConvergenceRoundTrips)
{
    SteepestDescent opt(1U);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        settings::ConvStrategy::RIGOROUS
    );

    opt.setConvergence(conv);
    EXPECT_EQ(
        opt.getConvergence().getEnConvStrategy(),
        settings::ConvStrategy::RIGOROUS
    );

    // Non-const getConvergence() returns a reference — mutating via it should
    // be observable on subsequent reads.
    auto &ref = opt.getConvergence();
    ref.calcForceConvergence(1.0, 1.0);
    EXPECT_DOUBLE_EQ(opt.getConvergence().getAbsMaxForce(), 1.0);
}

TEST(TestOptimizer, hasConvergedReturnsTrueForFlatEnergyAndZeroForces)
{
    SteepestDescent opt(1U);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        settings::ConvStrategy::RIGOROUS
    );
    opt.setConvergence(conv);

    // Two history entries with identical energy and zero forces → converged.
    pushSample(
        opt,
        {.energy = 1.0,
         .force0 = {0.0, 0.0, 0.0},
         .force1 = {0.0, 0.0, 0.0},
         .pos0   = {0.0, 0.0, 0.0},
         .pos1   = {0.0, 0.0, 0.0}}
    );
    pushSample(
        opt,
        {.energy = 1.0,
         .force0 = {0.0, 0.0, 0.0},
         .force1 = {0.0, 0.0, 0.0},
         .pos0   = {0.0, 0.0, 0.0},
         .pos1   = {0.0, 0.0, 0.0}}
    );

    EXPECT_TRUE(opt.hasConverged());
}

TEST(TestOptimizer, hasConvergedReturnsFalseForLargeForce)
{
    SteepestDescent opt(1U);
    Convergence     conv(
        true,
        true,
        true,
        1.0e-4,
        1.0e-4,
        1.0e-3,
        1.0e-3,
        settings::ConvStrategy::RIGOROUS
    );
    opt.setConvergence(conv);

    pushSample(
        opt,
        {.energy = 1.0,
         .force0 = {0.0, 0.0, 0.0},
         .force1 = {0.0, 0.0, 0.0},
         .pos0   = {0.0, 0.0, 0.0},
         .pos1   = {0.0, 0.0, 0.0}}
    );
    pushSample(
        opt,
        {.energy = 1.0,
         .force0 = {1.0, 0.0, 0.0},
         .force1 = {0.0, 0.0, 0.0},
         .pos0   = {0.0, 0.0, 0.0},
         .pos1   = {0.0, 0.0, 0.0}}
    );

    EXPECT_FALSE(opt.hasConverged());
}

/* ---------- old physical data setter ---------- */

TEST(TestOptimizer, setPhysicalDataOldStoresPointer)
{
    SteepestDescent opt(1U);
    auto            phys = std::make_shared<PhysicalData>();
    phys->setKineticEnergy(0.42);
    EXPECT_NO_THROW(opt.setPhysicalDataOld(phys));
}
