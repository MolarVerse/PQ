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
#include <vector>

#include "../potential/nonCoulomb/testForceFieldNonCoulomb.hpp"
#include "atom.hpp"
#include "coulombShiftedPotential.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "intraNonBondedContainer.hpp"
#include "intraNonBondedMap.hpp"
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

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

class TestIntraNonBondedMap : public TestNonCoulombPotentialFF
{
};

/**
 * @brief Test fixture class for the IntraNonBondedMap class
 */
TEST_F(TestIntraNonBondedMap, calculateSingleInteractionAndCalculate)
{
    auto molecule = molsys::Molecule{MolType{0}};

    auto atom1 = std::make_shared<molsys::Atom>();
    auto atom2 = std::make_shared<molsys::Atom>();

    atom1->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
    atom2->setPosition(linalg::Vec3D{0.0, 0.0, 11.0} * angstrom);
    atom1->setForce(Force3D::zero());
    atom2->setForce(Force3D::zero());
    atom1->setInternalGlobalVDWType(VdwType{0});
    atom2->setInternalGlobalVDWType(VdwType{1});
    atom1->setAtomType(AtomType{0});
    atom2->setAtomType(AtomType{1});
    atom1->setPartialCharge(0.5 * elementary_charge);
    atom2->setPartialCharge(-0.5 * elementary_charge);

    molecule.addAtom(atom1);
    molecule.addAtom(atom2);

    settings::PotentialSettings::setScale14Coulomb(0.75);
    settings::PotentialSettings::setScale14VanDerWaals(0.75);

    auto intraNonBondedType =
        intraNonBonded::IntraNonBondedContainer(MolType{0}, {{-1}});
    auto intraNonBondedMap =
        intraNonBonded::IntraNonBondedMap(&molecule, &intraNonBondedType);

    auto coulombPotential = pot::CoulombShiftedPotential(10.0 * angstrom);
    _setNonCoulombPairsMatrix(
        linalg::Matrix<std::shared_ptr<pot::NonCoulombPair>>(2, 2)
    );

    auto nonCoulombPair = pot::LennardJonesPair(
        ExtVdwType(0),
        ExtVdwType(1),
        10.0 * angstrom,
        LJParams{
            .c6  = 2.0 * kcal_per_mol * angstrom6,
            .c12 = 3.0 * kcal_per_mol * angstrom12
        }
    );
    _setNonCoulombPairsMatrix(0, 1, nonCoulombPair);
    _setNonCoulombPairsMatrix(1, 0, nonCoulombPair);

    auto simulationBox = molsys::SimulationBox();

    molsys::OrthorhombicBox orthoBox{
        {10.0 * angstrom, 10.0 * angstrom, 10.0 * angstrom}
    };
    simulationBox.setBox(orthoBox, false);

    auto physicalData = physicalData::PhysicalData();

    const auto [coulombEnergy, nonCoulombEnergy] =
        intraNonBondedMap.calculateSingleInteraction(
            AtomIndex{0},
            intraNonBondedType.getAtomIndices()[0][0],
            simulationBox.getBoxDimensions(),
            physicalData,
            &coulombPotential,
            _nonCoulombPotential.get()
        );

    EXPECT_NEAR(
        coulombEnergy.numerical_value_in(kcal_per_mol),
        -67.242901903583757 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        nonCoulombEnergy.numerical_value_in(kcal_per_mol),
        5.0 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        34.185768993269036 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
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
        -34.185768993269036 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol
        ),
        341.85768993269039 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );

    molecule.setAtomForcesToZero();
    molecule.getAtom(AtomIndex{0}).setShiftForce(ShiftForce3D::zero());
    molecule.getAtom(AtomIndex{1}).setShiftForce(ShiftForce3D::zero());
    physicalData.reset();

    intraNonBondedMap.calculate(
        &coulombPotential,
        _nonCoulombPotential.get(),
        simulationBox,
        physicalData
    );

    EXPECT_NEAR(
        physicalData.getCoulombEnergy().numerical_value_in(kcal_per_mol),
        -67.242901903583757 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getIntraCoulombEnergy().numerical_value_in(kcal_per_mol),
        -67.242901903583757 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getNonCoulombEnergy().numerical_value_in(kcal_per_mol),
        5.0 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        physicalData.getIntraNonCoulombEnergy().numerical_value_in(
            kcal_per_mol
        ),
        5.0 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        34.185768993269036 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol / angstrom
        ),
        0.0,
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
        -34.185768993269036 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[0].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[1].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{0})[2].numerical_value_in(
            kcal_per_mol
        ),
        341.85768993269039 * 0.75,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[0].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[1].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
    EXPECT_NEAR(
        molecule.getAtomShiftForce(AtomIndex{1})[2].numerical_value_in(
            kcal_per_mol
        ),
        0.0,
        1e-6
    );
}
