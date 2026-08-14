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
#include "simulationBox.hpp"
#include "steepestDescent.hpp"
#include "vector3d.hpp"   // IWYU pragma: keep

using namespace opt;
using simulationBox::Atom;
using simulationBox::SimulationBox;

namespace
{
    std::shared_ptr<SimulationBox> makeBoxWithTwoAtoms(
        const linearAlgebra::Vec3D &pos0,
        const linearAlgebra::Vec3D &pos1,
        const linearAlgebra::Vec3D &force0,
        const linearAlgebra::Vec3D &force1,
        const linearAlgebra::Vec3D &boxDims
    )
    {
        auto box = std::make_shared<SimulationBox>();
        box->setBoxDimensions(boxDims);

        auto a1 = std::make_shared<Atom>();
        auto a2 = std::make_shared<Atom>();
        a1->setPosition(pos0);
        a2->setPosition(pos1);
        a1->setForce(force0);
        a2->setForce(force1);

        box->addAtom(a1);
        box->addAtom(a2);

        return box;
    }

    std::shared_ptr<SimulationBox> makeBoxWithTwoAtoms(
        const linearAlgebra::Vec3D &pos0,
        const linearAlgebra::Vec3D &pos1,
        const linearAlgebra::Vec3D &force0,
        const linearAlgebra::Vec3D &force1
    )
    {
        return makeBoxWithTwoAtoms(
            pos0,
            pos1,
            force0,
            force1,
            {100.0, 100.0, 100.0}
        );
    }
}   // namespace

/* ---------- single update step ---------- */

TEST(TestSteepestDescent, updateMovesAtomsByLearningRateTimesForce)
{
    auto box = makeBoxWithTwoAtoms(
        {0.0, 0.0, 0.0},
        {1.0, 2.0, 3.0},
        {0.5, 1.0, -1.0},
        {-0.2, 0.3, 0.7}
    );

    SteepestDescent opt(1u);
    opt.setSimulationBox(box);

    const auto lr = 0.1;
    opt.update(lr, 1u);

    EXPECT_DOUBLE_EQ(box->getAtoms()[0]->getPosition()[0], 0.0 + lr * 0.5);
    EXPECT_DOUBLE_EQ(box->getAtoms()[0]->getPosition()[1], 0.0 + lr * 1.0);
    EXPECT_DOUBLE_EQ(box->getAtoms()[0]->getPosition()[2], 0.0 + lr * -1.0);

    EXPECT_DOUBLE_EQ(box->getAtoms()[1]->getPosition()[0], 1.0 + lr * -0.2);
    EXPECT_DOUBLE_EQ(box->getAtoms()[1]->getPosition()[1], 2.0 + lr * 0.3);
    EXPECT_DOUBLE_EQ(box->getAtoms()[1]->getPosition()[2], 3.0 + lr * 0.7);
}

TEST(TestSteepestDescent, updateStoresOldPosition)
{
    auto box = makeBoxWithTwoAtoms(
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {0.1, 0.1, 0.1},
        {0.2, 0.2, 0.2}
    );

    SteepestDescent opt(1u);
    opt.setSimulationBox(box);
    opt.update(0.05, 1u);

    EXPECT_EQ(
        box->getAtoms()[0]->getPositionOld(),
        linearAlgebra::Vec3D(1.0, 2.0, 3.0)
    );
    EXPECT_EQ(
        box->getAtoms()[1]->getPositionOld(),
        linearAlgebra::Vec3D(4.0, 5.0, 6.0)
    );
}

TEST(TestSteepestDescent, updateAppliesPBCToNewPosition)
{
    // pos + lr * force = 9.0 + 1.0 * 2.0 = 11.0. With a 10.0 box the
    // minimum-image wrap should pull it back to 1.0.
    auto box = makeBoxWithTwoAtoms(
        {9.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {2.0, 0.0, 0.0},
        {0.0, 0.0, 0.0},
        {10.0, 10.0, 10.0}
    );

    SteepestDescent opt(1u);
    opt.setSimulationBox(box);
    opt.update(1.0, 1u);

    EXPECT_DOUBLE_EQ(box->getAtoms()[0]->getPosition()[0], 1.0);
}

TEST(TestSteepestDescent, updateIsNoOpWithZeroLearningRate)
{
    auto box = makeBoxWithTwoAtoms(
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {1.0, 1.0, 1.0},
        {1.0, 1.0, 1.0}
    );

    SteepestDescent opt(1u);
    opt.setSimulationBox(box);
    opt.update(0.0, 1u);

    EXPECT_EQ(
        box->getAtoms()[0]->getPosition(),
        linearAlgebra::Vec3D(1.0, 2.0, 3.0)
    );
    EXPECT_EQ(
        box->getAtoms()[1]->getPosition(),
        linearAlgebra::Vec3D(4.0, 5.0, 6.0)
    );
}
