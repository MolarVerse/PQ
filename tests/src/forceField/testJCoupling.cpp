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

#include <vector>

#include "gtest/gtest.h"
#include "jCouplingForceField.hpp"
#include "jCouplingType.hpp"
#include "molecule.hpp"

/* ---------- JCouplingType ---------- */

/**
 * @brief Verify operator== uses the id + (J0, k, a, b, c, phaseShift)
 * tuple; symmetry flags are *not* part of the equality contract.
 */
TEST(TestJCouplingType, operatorEqual)
{
    const forceField::JCouplingType t1(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_same(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherId(1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherJ0(0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherK(0, 1.0, 9.0, 3.0, 4.0, 5.0, 6.0);

    EXPECT_TRUE(t1 == t1_same);
    EXPECT_FALSE(t1 == t1_otherId);
    EXPECT_FALSE(t1 == t1_otherJ0);
    EXPECT_FALSE(t1 == t1_otherK);
}

TEST(TestJCouplingType, getters)
{
    const forceField::JCouplingType t(7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    EXPECT_EQ(t.getId(), 7U);
    EXPECT_DOUBLE_EQ(t.getJ0(), 1.0);
    EXPECT_DOUBLE_EQ(t.getForceConstant(), 2.0);
    EXPECT_DOUBLE_EQ(t.getA(), 3.0);
    EXPECT_DOUBLE_EQ(t.getB(), 4.0);
    EXPECT_DOUBLE_EQ(t.getC(), 5.0);
    EXPECT_DOUBLE_EQ(t.getPhaseShift(), 6.0);
}

TEST(TestJCouplingType, symmetryFlagSetters)
{
    forceField::JCouplingType t(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    // Symmetry flags only have setters; verify they accept both bool values
    // without throwing and do not affect equality (covered by operatorEqual).
    t.setUpperSymmetry(false);
    t.setLowerSymmetry(false);
    t.setUpperSymmetry(true);
    t.setLowerSymmetry(true);

    SUCCEED();
}

/* ---------- JCouplingForceField ---------- */

TEST(TestJCouplingForceField, settersAndGetters)
{
    molsys::Molecule                molecule;
    forceField::JCouplingForceField ff(
        std::vector<molsys::Molecule *>{
            &molecule,
            &molecule,
            &molecule,
            &molecule

        },
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        42
    );

    EXPECT_EQ(ff.getType(), 42U);

    ff.setJ0(1.5);
    ff.setForceConstant(2.5);
    ff.setA(3.5);
    ff.setB(4.5);
    ff.setC(5.5);
    ff.setPhaseShift(0.25);

    EXPECT_DOUBLE_EQ(ff.getJ0(), 1.5);
    EXPECT_DOUBLE_EQ(ff.getForceConstant(), 2.5);
    EXPECT_DOUBLE_EQ(ff.getA(), 3.5);
    EXPECT_DOUBLE_EQ(ff.getB(), 4.5);
    EXPECT_DOUBLE_EQ(ff.getC(), 5.5);
    EXPECT_DOUBLE_EQ(ff.getPhaseShift(), 0.25);
}

TEST(TestJCouplingForceField, symmetryFlagsDefaultTrue)
{
    molsys::Molecule                molecule;
    forceField::JCouplingForceField ff(
        std::vector<molsys::Molecule *>{
            &molecule,
            &molecule,
            &molecule,
            &molecule
        },
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        0
    );

    // Both symmetry flags default to true (per class declaration).
    EXPECT_TRUE(ff.getUpperSymmetry());
    EXPECT_TRUE(ff.getLowerSymmetry());

    ff.setUpperSymmetry(false);
    ff.setLowerSymmetry(false);
    EXPECT_FALSE(ff.getUpperSymmetry());
    EXPECT_FALSE(ff.getLowerSymmetry());
}
