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

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "dihedralForceField.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "hybridSettings.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "units.hpp"
#include "vector3d.hpp"

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

class TestDihedralForceField : public TestNonCoulombPotentialFF
{
};

TEST_F(TestDihedralForceField, calculateEnergyAndForces)
{
    const HybridSettings hybridSettings;
    auto                 box = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    auto physicalData     = physicalData::PhysicalData();
    auto coulombPotential = pot::CoulombShiftedPotential(20.0 * angstrom);

    auto nonCoulombPair = pot::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        15.0 * angstrom,
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
    auto atom3 = std::make_shared<molsys::Atom>();
    auto atom4 = std::make_shared<molsys::Atom>();

    atom1->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
    atom2->setPosition(linalg::Vec3D{1.0, 1.0, 1.0} * angstrom);
    atom3->setPosition(linalg::Vec3D{1.0, 2.0, 3.0} * angstrom);
    atom4->setPosition(linalg::Vec3D{4.0, 2.0, 3.0} * angstrom);

    atom1->setForce(Force3D::zero());
    atom2->setForce(Force3D::zero());
    atom3->setForce(Force3D::zero());
    atom4->setForce(Force3D::zero());

    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom3->setInternalGlobalVDWType(VdwType{0});
    atom4->setInternalGlobalVDWType(VdwType{1});

    atom1->setAtomType(AtomType{0});
    atom2->setAtomType(AtomType{1});
    atom3->setAtomType(AtomType{0});
    atom4->setAtomType(AtomType{1});

    atom1->setPartialCharge(1.0 * elementary_charge);
    atom2->setPartialCharge(-0.5 * elementary_charge);
    atom3->setPartialCharge(1.0 * elementary_charge);
    atom4->setPartialCharge(-0.5 * elementary_charge);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);
    molecule.addAtom(atom4);

    auto bondForceField = ff::DihedralForceField(
        {&molecule, &molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}, AtomIndex{3}},
        DihedralId{0}
    );
    bondForceField.setParams(
        DihedralParams{
            .forceConstant = 3.0 * kcal_per_mol,
            .frequency     = 3,
            .phaseShift    = 180.0 * M_PI / 180.0 * rad
        }
    );
    bondForceField.setIsLinker(false);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.5);

    bondForceField.calculateEnergyAndForces(
        box,
        physicalData,
        false,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.getSmoothingMethod()
    );

    EXPECT_NEAR(
        physicalData.getDihedralEnergy().numerical_value_in(kcal_per_mol),
        3.9128709291752739,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getImproperEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        3.19504825211347,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -6.39009650422694,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        3.19504825211347,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        10.224154406763104,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.9170289512680818,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -1.2780193008453877,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.63900965042269386,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -2.5560386016907759,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.278019300845388,
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
    EXPECT_THAT(
        diagonal(physicalData.getVirial()).numerical_value_in(kcal_per_mol),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6)
        )
    );

    molecule.setAtomForce(AtomIndex{0}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{1}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{2}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{3}, Force3D::zero());
    physicalData.reset();

    bondForceField.setIsLinker(false);

    bondForceField.calculateEnergyAndForces(
        box,
        physicalData,
        true,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.getSmoothingMethod()
    );

    EXPECT_NEAR(
        physicalData.getImproperEnergy().numerical_value_in(kcal_per_mol),
        3.9128709291752739,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getDihedralEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        3.19504825211347,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -6.39009650422694,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        3.19504825211347,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        10.224154406763104,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.9170289512680818,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -1.2780193008453877,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.63900965042269386,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -2.5560386016907759,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.278019300845388,
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
    EXPECT_THAT(
        diagonal(physicalData.getVirial()).numerical_value_in(kcal_per_mol),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6)
        )
    );

    molecule.setAtomForce(AtomIndex{0}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{1}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{2}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{3}, Force3D::zero());
    physicalData.reset();

    bondForceField.setIsLinker(true);

    bondForceField.calculateEnergyAndForces(
        box,
        physicalData,
        false,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.getSmoothingMethod()
    );

    EXPECT_NEAR(
        physicalData.getDihedralEnergy().numerical_value_in(kcal_per_mol),
        3.9128709291752739,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getImproperEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        2.2090108292824047,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -6.8831152156424729,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        2.4555201849901707,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        10.224154406763104,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -5.1120772033815518,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.9170289512680818,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -1.2780193008453877,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.63900965042269386,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.98603742283106555,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -2.063019890275243,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{3})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        2.0175473679686871,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        4.1158570930777021,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        -4.100545344959669e-05,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getVirial()[0][0].numerical_value_in(kcal_per_mol),
        3.9441496913242622,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getVirial()[1][1].numerical_value_in(kcal_per_mol),
        0.98603742283106555,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getVirial()[2][2].numerical_value_in(kcal_per_mol),
        2.2185842013698975,
        1e-6
    );
}
