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

#include "adam.hpp"
#include "atom.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

using namespace opt;
using simulationBox::Atom;
using simulationBox::SimulationBox;

namespace
{
    std::shared_ptr<SimulationBox> makeBoxWithOneAtom(
        const linearAlgebra::Vec3D &pos,
        const linearAlgebra::Vec3D &force,
        const linearAlgebra::Vec3D &boxDims
    )
    {
        auto box = std::make_shared<SimulationBox>();
        box->setBoxDimensions(boxDims);

        auto a = std::make_shared<Atom>();
        a->setPosition(pos);
        a->setForce(force);
        box->addAtom(a);

        return box;
    }

    std::shared_ptr<SimulationBox> makeBoxWithOneAtom(
        const linearAlgebra::Vec3D &pos,
        const linearAlgebra::Vec3D &force
    )
    {
        return makeBoxWithOneAtom(pos, force, {100.0, 100.0, 100.0});
    }
}   // namespace

/* ---------- constructors ---------- */

TEST(TestAdam, defaultBetasConstructorAcceptsNAtoms)
{
    EXPECT_NO_THROW(Adam(10U, /*nAtoms=*/4U));
}

TEST(TestAdam, customBetasConstructorAcceptsBeta1AndBeta2)
{
    EXPECT_NO_THROW(Adam(10U, /*beta1=*/0.5, /*beta2=*/0.5, /*nAtoms=*/4U));
}

TEST(TestAdam, cloneProducesAdamInstance)
{
    const Adam src(10U, 4U);
    const auto cloned = src.clone();
    EXPECT_NE(std::dynamic_pointer_cast<Adam>(cloned), nullptr);
}

TEST(TestAdam, maxHistoryLengthIsTwo)
{
    const Adam adam(10U, 4U);
    EXPECT_EQ(adam.maxHistoryLength(), 2U);
}

/* ---------- single update step ---------- */

TEST(TestAdam, updateAtStepOneReducesToLearningRateTimesSignOfForce)
{
    // Adam-step-1 with momentum1=0, momentum2=0 simplifies analytically to
    //   m1_hat = -force, m2_hat = force²
    //   pos_new ≈ pos + lr * force / sqrt(force² + eps²)
    // For force >> eps, sqrt(force² + eps²) ≈ |force|, so pos_new ≈ pos + lr *
    // sign(force).
    auto box = makeBoxWithOneAtom({0.0, 0.0, 0.0}, {2.0, -3.0, 0.5});

    Adam adam(1U, /*nAtoms=*/1U);
    adam.setSimulationBox(box);

    const auto lr = 0.01;
    adam.update(lr, 1U);

    // Per-component direction is preserved (no aliasing across xyz in Adam).
    const auto pos = box->getAtoms()[0]->getPosition();
    EXPECT_GT(pos[0], 0.0);   // positive force component
    EXPECT_LT(pos[1], 0.0);   // negative force component
    EXPECT_GT(pos[2], 0.0);   // positive force component

    // For large forces relative to eps, each component magnitude ≈ lr.
    EXPECT_NEAR(std::abs(pos[0]), lr, 1.0e-6);
    EXPECT_NEAR(std::abs(pos[1]), lr, 1.0e-6);
    EXPECT_NEAR(std::abs(pos[2]), lr, 1.0e-6);
}

TEST(TestAdam, updateStoresOldPosition)
{
    auto box = makeBoxWithOneAtom({3.0, 4.0, 5.0}, {1.0, 1.0, 1.0});

    Adam adam(1U, 1U);
    adam.setSimulationBox(box);
    adam.update(0.01, 1U);

    EXPECT_EQ(
        box->getAtoms()[0]->getPositionOld(),
        linearAlgebra::Vec3D(3.0, 4.0, 5.0)
    );
}

TEST(TestAdam, updateAppliesPBCToNewPosition)
{
    // pos + ~lr*sign(force) on a 10x10x10 box with start near the boundary
    // and a positive force should wrap.
    auto box = makeBoxWithOneAtom(
        {9.99, 0.0, 0.0},
        {1.0, 0.0, 0.0},
        {10.0, 10.0, 10.0}
    );

    Adam adam(1U, 1U);
    adam.setSimulationBox(box);
    adam.update(/*learningRate=*/0.5, /*step=*/1U);

    // After step pos[0] ≈ 9.99 + 0.5 ≈ 10.49 → wraps to ≈ 0.49.
    EXPECT_LT(box->getAtoms()[0]->getPosition()[0], 1.0);
    EXPECT_GE(box->getAtoms()[0]->getPosition()[0], 0.0);
}

TEST(TestAdam, updateLeavesPositionUnchangedWhenForceIsZero)
{
    auto box = makeBoxWithOneAtom({1.0, 2.0, 3.0}, {0.0, 0.0, 0.0});

    Adam adam(1U, 1U);
    adam.setSimulationBox(box);
    adam.update(0.1, 1U);

    EXPECT_EQ(
        box->getAtoms()[0]->getPosition(),
        linearAlgebra::Vec3D(1.0, 2.0, 3.0)
    );
}
