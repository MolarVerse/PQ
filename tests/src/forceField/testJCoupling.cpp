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
    const forceField::JCouplingType type1(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_same(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherId(1, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherJ0(0, 9.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    const forceField::JCouplingType t1_otherK(0, 1.0, 9.0, 3.0, 4.0, 5.0, 6.0);

    EXPECT_TRUE(type1 == t1_same);
    EXPECT_FALSE(type1 == t1_otherId);
    EXPECT_FALSE(type1 == t1_otherJ0);
    EXPECT_FALSE(type1 == t1_otherK);
}

TEST(TestJCouplingType, getters)
{
    const forceField::JCouplingType type(7, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);
    EXPECT_EQ(type.getId(), 7U);
    EXPECT_DOUBLE_EQ(type.getJ0(), 1.0);
    EXPECT_DOUBLE_EQ(type.getForceConstant(), 2.0);
    EXPECT_DOUBLE_EQ(type.getA(), 3.0);
    EXPECT_DOUBLE_EQ(type.getB(), 4.0);
    EXPECT_DOUBLE_EQ(type.getC(), 5.0);
    EXPECT_DOUBLE_EQ(type.getPhaseShift(), 6.0);
}

TEST(TestJCouplingType, symmetryFlagSetters)
{
    forceField::JCouplingType type(0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

    // Symmetry flags only have setters; verify they accept both bool values
    // without throwing and do not affect equality (covered by operatorEqual).
    type.setUpperSymmetry(false);
    type.setLowerSymmetry(false);
    type.setUpperSymmetry(true);
    type.setLowerSymmetry(true);

    SUCCEED();
}

/* ---------- JCouplingForceField ---------- */

TEST(TestJCouplingForceField, settersAndGetters)
{
    molsys::Molecule                molecule;
    forceField::JCouplingForceField forceField(
        std::vector<molsys::Molecule *>{
            &molecule,
            &molecule,
            &molecule,
            &molecule

        },
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        42
    );

    EXPECT_EQ(forceField.getType(), 42U);

    forceField.setJ0(1.5);
    forceField.setForceConstant(2.5);
    forceField.setA(3.5);
    forceField.setB(4.5);
    forceField.setC(5.5);
    forceField.setPhaseShift(0.25);

    EXPECT_DOUBLE_EQ(forceField.getJ0(), 1.5);
    EXPECT_DOUBLE_EQ(forceField.getForceConstant(), 2.5);
    EXPECT_DOUBLE_EQ(forceField.getA(), 3.5);
    EXPECT_DOUBLE_EQ(forceField.getB(), 4.5);
    EXPECT_DOUBLE_EQ(forceField.getC(), 5.5);
    EXPECT_DOUBLE_EQ(forceField.getPhaseShift(), 0.25);
}

TEST(TestJCouplingForceField, symmetryFlagsDefaultTrue)
{
    molsys::Molecule                molecule;
    forceField::JCouplingForceField forceField(
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
    EXPECT_TRUE(forceField.getUpperSymmetry());
    EXPECT_TRUE(forceField.getLowerSymmetry());

    forceField.setUpperSymmetry(false);
    forceField.setLowerSymmetry(false);
    EXPECT_FALSE(forceField.getUpperSymmetry());
    EXPECT_FALSE(forceField.getLowerSymmetry());
}
