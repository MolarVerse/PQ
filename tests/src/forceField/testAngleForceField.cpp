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
#include <cstddef>
#include <memory>

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "angleForceField.hpp"
#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "gmock/gmock.h"
#include "hybridSettings.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"
#include "molecule.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "potentialSettings.hpp"
#include "simulationBox.hpp"
#include "strongTypes.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "vector3d/vector3dClass.hpp"

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

class TestAngleForceField : public TestNonCoulombPotentialFF
{
};

TEST_F(TestAngleForceField, calculateEnergyAndForces)
{
    auto box = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    auto potentialSettings = PotentialSettings();
    potentialSettings.setCoulombRadiusCutoffByUser(10.0 * angstrom);
    auto physicalData     = physicalData::PhysicalData();
    auto coulombPotential = pot::CoulombShiftedPotential(potentialSettings);

    auto nonCoulombPair = pot::LennardJonesPair(
        ExtVdwType(1),
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
    _setNonCoulombPairsMatrix(1, 1, nonCoulombPair);

    auto molecule = molsys::Molecule();

    molecule.setMoltype(MolType{0});

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    auto atom3 = std::make_shared<molsys::Atom>();

    atom1->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
    atom2->setPosition(linalg::Vec3D{1.0, 1.0, 1.0} * angstrom);
    atom3->setPosition(linalg::Vec3D{1.0, 2.0, 3.0} * angstrom);

    atom1->setForce(Force3D::zero());
    atom2->setForce(Force3D::zero());
    atom3->setForce(Force3D::zero());

    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom3->setInternalGlobalVDWType(VdwType{1});

    atom1->setAtomType(AtomType{0});
    atom2->setAtomType(AtomType{1});
    atom3->setAtomType(AtomType{1});

    atom1->setPartialCharge(1.0 * elementary_charge);
    atom2->setPartialCharge(-0.5 * elementary_charge);
    atom3->setPartialCharge(-0.5 * elementary_charge);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    auto angleFF = ff::AngleForceField(
        {&molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    const auto equilibrium   = 90 * M_PI / 180.0 * rad;
    const auto forceConstant = 3.0 * kcal_per_mol / rad / rad;
    angleFF.setParams(
        AngleParams{.equilibrium = equilibrium, .forceConstant = forceConstant}
    );
    angleFF.setIsLinker(false);

    const HybridSettings hybridSettings;
    angleFF.calculateEnergyAndForces(
        box,
        physicalData,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.get_smoothing(),
        potentialSettings
    );

    EXPECT_NEAR(
        physicalData.get_angleEnergy().numerical_value_in(kcal_per_mol),
        2.0999420826401303,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -0.62105043904006785,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.20701681301335595,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.0350840650667796,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.4491176910934915,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -1.4491176910934915,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -0.82806725205342369,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -0.20701681301335595,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.41403362602671184,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_coulombEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_nonCoulombEnergy().numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_THAT(
        diagonal(physicalData.get_virial()).numerical_value_in(kcal_per_mol),
        testing::ElementsAre(
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6),
            testing::DoubleNear(0.0, 1e-6)
        )
    );

    molecule.setAtomForce(AtomIndex{0}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{1}, Force3D::zero());
    molecule.setAtomForce(AtomIndex{2}, Force3D::zero());
    physicalData.reset();

    angleFF.setIsLinker(true);

    angleFF.calculateEnergyAndForces(
        box,
        physicalData,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.get_smoothing(),
        potentialSettings
    );

    EXPECT_NEAR(
        physicalData.get_angleEnergy().numerical_value_in(kcal_per_mol),
        2.0999420826401303,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -0.62105043904006785,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.20701681301335595,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.0350840650667796,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        1.4491176910934915,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        7.0737262359370403,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        12.69833478078059,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -0.82806725205342369,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -7.2807430489503959,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{2})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        -13.733418845847369,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_coulombEnergy().numerical_value_in(kcal_per_mol),
        -22.378958701288319,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_nonCoulombEnergy().numerical_value_in(kcal_per_mol),
        -0.016255999999999989,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_virial()[0][0].numerical_value_in(kcal_per_mol),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_virial()[1][1].numerical_value_in(kcal_per_mol),
        -7.0737262359370403,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.get_virial()[2][2].numerical_value_in(kcal_per_mol),
        -28.294904943748161,
        1e-6
    );
}

/**
 * @brief regression test for the sin(alpha) guard: a strictly collinear
 * arrangement (alpha == pi) used to produce NaN forces via division by
 * sin(alpha) in the cross-product force decomposition. The guard skips
 * just that block; the energy contribution and the linker correction
 * (when enabled) still run, and no force component is NaN/Inf.
 */
TEST_F(TestAngleForceField, collinearAngleProducesFiniteForces)
{
    auto box = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {100.0 * angstrom, 100.0 * angstrom, 100.0 * angstrom}
    };
    box.setBox(orthoBox, false);

    auto              physicalData = physicalData::PhysicalData();
    PotentialSettings potentialSettings;
    potentialSettings.setCoulombRadiusCutoffByUser(10.0 * angstrom);
    auto coulombPotential = pot::CoulombShiftedPotential(potentialSettings);

    auto molecule = molsys::Molecule();
    molecule.setMoltype(MolType{0});

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();
    auto atom3 = std::make_shared<molsys::Atom>();

    // Strictly collinear: atom2 is the central atom, atom1 and atom3 are
    // 180 degrees apart along the x axis. alpha = pi -> sin(alpha) = 0.
    atom1->setPosition(linalg::Vec3D{1.0, 0.0, 0.0} * angstrom);
    atom2->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
    atom3->setPosition(linalg::Vec3D{-1.0, 0.0, 0.0} * angstrom);

    atom1->setForce(Force3D::zero());
    atom2->setForce(Force3D::zero());
    atom3->setForce(Force3D::zero());

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);
    molecule.addAtom(atom3);

    auto angleForceField = ff::AngleForceField(
        {&molecule, &molecule, &molecule},
        {AtomIndex{0}, AtomIndex{1}, AtomIndex{2}},
        AngleId{0}
    );
    angleForceField.setParams(
        AngleParams{
            .equilibrium   = M_PI * rad,
            .forceConstant = 3.0 * kcal_per_mol / rad / rad
        }
    );
    angleForceField.setIsLinker(false);

    const HybridSettings hybridSettings;
    angleForceField.calculateEnergyAndForces(
        box,
        physicalData,
        coulombPotential,
        *_nonCoulombPotential,
        hybridSettings.get_smoothing(),
        potentialSettings
    );

    // Energy is finite at the singular angle (harmonic, deltaAngle ~ 0).
    EXPECT_FALSE(isnan(physicalData.get_angleEnergy()));
    EXPECT_FALSE(isinf(physicalData.get_angleEnergy()));

    // All per-atom forces must be finite. Without the guard these would
    // be NaN from dividing by sin(pi) == 0 in the cross-product block.
    for (AtomIndex atomIdx{0}; atomIdx.get() < 3; ++atomIdx)
    {
        for (size_t i = 0; i < 3; ++i)
        {
            EXPECT_FALSE(isnan(molecule.getAtomForce(atomIdx)[i]));
            EXPECT_FALSE(isinf(molecule.getAtomForce(atomIdx)[i]));
        }
    }
}
