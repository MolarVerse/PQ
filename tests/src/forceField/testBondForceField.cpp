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

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "atom.hpp"
#include "bondForceField.hpp"
#include "coulombShiftedPotential.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "gmock/gmock.h"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "units.hpp"

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

class TestBondForceField : public TestNonCoulombPotentialFF
{
};

TEST_F(TestBondForceField, calculateEnergyAndForces)
{
    auto box = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    auto physicalData     = physicalData::PhysicalData();
    auto coulombPotential = pot::CoulombShiftedPotential(10.0 * angstrom);

    auto nonCoulombPair = pot::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        5.0 * angstrom,
        LJParams{
            .c6  = 2.0 * kcal_per_mol * angstrom6,
            .c12 = 4.0 * kcal_per_mol * angstrom12
        }
    );
    _setNonCoulombPairsMatrix(
        linalg::Matrix<std::shared_ptr<pot::NonCoulombPair>>(2, 2)
    );
    _setNonCoulombPairsMatrix(0, 1, nonCoulombPair);

    auto molecule = molsys::Molecule();

    molecule.setMoltype(MolType{0});

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
    atom2->setPosition(linalg::Vec3D{1.0, 2.0, 3.0} * angstrom);
    atom1->setForce(Force3D::zero());
    atom2->setForce(Force3D::zero());
    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom1->setAtomType(AtomType{0});
    atom2->setAtomType(AtomType{1});
    atom1->setPartialCharge(1.0 * elementary_charge);
    atom2->setPartialCharge(-0.5 * elementary_charge);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    auto bondForceField = ff::BondForceField(
        &molecule,
        &molecule,
        AtomIndex{0},
        AtomIndex{1},
        BondId{0}
    );
    bondForceField.setParams(
        BondParams{
            .equilibrium   = 1.2 * angstrom,
            .forceConstant = 3.0 * kcal_per_mol / angstrom / angstrom
        }
    );
    bondForceField.setIsLinker(false);

    bondForceField.calculateEnergyAndForces(
        box,
        physicalData,
        coulombPotential,
        *_nonCoulombPotential
    );

    auto force =
        (::sqrt(14) - 1.2) * 3.0 / ::sqrt(14) * linalg::Vec3D(1.0, 2.0, 3.0);

    EXPECT_NEAR(
        physicalData.getBondEnergy().numerical_value_in(kcal_per_mol),
        (::sqrt(14) - 1.2) * (::sqrt(14) - 1.2) * 3.0 / 2.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[0],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[1],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[2],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[0],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[1],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[2],
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );

    auto virial = force * linalg::Vec3D{-1.0, -2.0, -3.0};
    EXPECT_THAT(
        diagonal(physicalData.getVirial()).numerical_value_in(kcal_per_mol),
        testing::ElementsAre(
            testing::DoubleNear(virial[0], 1e-6),
            testing::DoubleNear(virial[1], 1e-6),
            testing::DoubleNear(virial[2], 1e-6)
        )
    );

    molecule.setAtomForce(AtomIndex{0}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{1}, Force3D::zero());
    physicalData.reset();

    bondForceField.setIsLinker(true);

    bondForceField.calculateEnergyAndForces(
        box,
        physicalData,
        coulombPotential,
        *_nonCoulombPotential
    );

    force = {-0.68765171383567736, -1.3753034276713547, -2.062955141507032};

    EXPECT_NEAR(
        physicalData.getBondEnergy().numerical_value_in(kcal_per_mol),
        (::sqrt(14) - 1.2) * (::sqrt(14) - 1.2) * 3.0 / 2.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[0],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[1],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        force[2],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[0],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[1],
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -force[2],
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        17.379852093794977,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        -0.00072939421499545261,
        1e-6
    );

    virial = force * linalg::Vec3D{-1.0, -2.0, -3.0};
    EXPECT_THAT(
        diagonal(physicalData.getVirial()).numerical_value_in(kcal_per_mol),
        testing::ElementsAre(
            testing::DoubleNear(virial[0], 1e-6),
            testing::DoubleNear(virial[1], 1e-6),
            testing::DoubleNear(virial[2], 1e-6)
        )
    );
}
