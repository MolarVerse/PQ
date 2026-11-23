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

#include "testPhysicalData.hpp"

#include <array>
#include <cstddef>
#include <new>

#include "physicalData.hpp"
#include "units.hpp"
#include "vector3d.hpp"

/**
 * @brief tests makeAverages function
 *
 */
namespace
{
    // kcal/mol of the kinetic energy 0.5 * m * v^2 for m in g/mol, v in A/s
    const double KINETIC_ENERGY_FACTOR =
        (0.5 * (1.0 * g_per_mol) * (1.0 * angstrom / s) * (1.0 * angstrom / s))
            .numerical_value_in(kcal_per_mol);
}   // namespace

TEST_F(TestPhysicalData, makeAverages)
{
    _physicalData->makeAverages(2);
    EXPECT_EQ(_physicalData->get_coulombEnergy(), 0.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_nonCoulombEnergy(), 1.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_temperature(), toTemperature(1.5));
    EXPECT_EQ(
        _physicalData->get_momentum(),
        linalg::Vec3D(2.0) * g_per_mol * angstrom / s
    );
    EXPECT_EQ(_physicalData->get_kineticEnergy(), 2.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_volume(), 3.0 * angstrom3);
    EXPECT_EQ(_physicalData->get_density(), 3.5 * kg_per_litre);
    EXPECT_EQ(_physicalData->get_pressure(), 4.0 * bar);
    EXPECT_EQ(_physicalData->get_qmEnergy(), 4.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_qmReferenceEnergy(), 0.5 * kcal_per_mol);
}

/**
 * @brief tests copy function
 *
 */
TEST_F(TestPhysicalData, copy)
{
    physicalData::PhysicalData physicalData2;
    physicalData2.copy(*_physicalData);
    EXPECT_EQ(physicalData2.get_coulombEnergy(), 1.0 * kcal_per_mol);
    EXPECT_EQ(physicalData2.get_nonCoulombEnergy(), 2.0 * kcal_per_mol);
    EXPECT_EQ(physicalData2.get_temperature(), toTemperature(3.0));
    EXPECT_EQ(
        physicalData2.get_momentum(),
        linalg::Vec3D(4.0) * g_per_mol * angstrom / s
    );
    EXPECT_EQ(physicalData2.get_kineticEnergy(), 5.0 * kcal_per_mol);
    EXPECT_EQ(physicalData2.get_volume(), 6.0 * angstrom3);
    EXPECT_EQ(physicalData2.get_density(), 7.0 * kg_per_litre);
    EXPECT_EQ(physicalData2.get_pressure(), 8.0 * bar);
    EXPECT_EQ(physicalData2.get_coupledPressure(), 8.5 * bar);
    EXPECT_EQ(physicalData2.get_qmEnergy(), 9.0 * kcal_per_mol);
    EXPECT_EQ(physicalData2.get_qmReferenceEnergy(), 1.0 * kcal_per_mol);
}

/**
 * @brief tests updateAverages function
 *
 */
TEST_F(TestPhysicalData, updateAverages)
{
    const physicalData::PhysicalData physicalData2 = *_physicalData;

    _physicalData->updateAverages(physicalData2);
    EXPECT_EQ(_physicalData->get_coulombEnergy(), 2.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_nonCoulombEnergy(), 4.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_temperature(), toTemperature(6.0));
    EXPECT_EQ(
        _physicalData->get_momentum(),
        linalg::Vec3D(8.0) * g_per_mol * angstrom / s
    );
    EXPECT_EQ(_physicalData->get_kineticEnergy(), 10.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_volume(), 12.0 * angstrom3);
    EXPECT_EQ(_physicalData->get_density(), 14.0 * kg_per_litre);
    EXPECT_EQ(_physicalData->get_pressure(), 16.0 * bar);
    EXPECT_EQ(_physicalData->get_coupledPressure(), 17.0 * bar);
    EXPECT_EQ(_physicalData->get_qmEnergy(), 18.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_qmReferenceEnergy(), 2.0 * kcal_per_mol);
}

/**
 * @brief tests calculateKinetics function
 *
 */
TEST_F(TestPhysicalData, calculateKinetics)
{
    _physicalData->calculateKinetics(*_simulationBox);

    const auto mol1 = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 =
        mol1.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto velocity_mol1_atom2 =
        mol1.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s);
    const auto mass_mol1_atom1 =
        mol1.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);
    const auto mass_mol1_atom2 =
        mol1.getAtomMass(AtomIndex{1}).numerical_value_in(g_per_mol);

    const auto mol2 = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 =
        mol2.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto mass_mol2_atom1 =
        mol2.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);

    const auto momentumVector = velocity_mol1_atom1 * mass_mol1_atom1 +
                                velocity_mol1_atom2 * mass_mol1_atom2 +
                                velocity_mol2_atom1 * mass_mol2_atom1;

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto molecularMomentum = velocity_mol1_atom1 * mass_mol1_atom1 +
                                   velocity_mol1_atom2 * mass_mol1_atom2;
    const auto kineticEnergyMolecularVector =
        molecularMomentum * molecularMomentum /
            (mass_mol1_atom1 + mass_mol1_atom2) +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    EXPECT_EQ(
        _physicalData->get_momentum().numerical_value_in(
            g_per_mol * angstrom / s
        ),
        momentumVector
    );
    const auto atomExpected = kineticEnergyAtomicVector * KINETIC_ENERGY_FACTOR;
    const auto atomActual   = diagonal(_physicalData->getKinEnergyAtomTensor())
                                .numerical_value_in(kcal_per_mol);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(atomActual[i], atomExpected[i]);

    const auto molExpected =
        kineticEnergyMolecularVector * KINETIC_ENERGY_FACTOR;
    const auto molActual = diagonal(_physicalData->getKinEnergyMolTensor())
                               .numerical_value_in(kcal_per_mol);
    for (size_t i = 0; i < 3; ++i)
        EXPECT_DOUBLE_EQ(molActual[i], molExpected[i]);

    EXPECT_DOUBLE_EQ(
        _physicalData->get_kineticEnergy().numerical_value_in(kcal_per_mol),
        sum(kineticEnergyAtomicVector) * KINETIC_ENERGY_FACTOR
    );
}

/**
 * @brief tests calculateTemperature function
 *
 */
TEST_F(TestPhysicalData, calculateTemperature)
{
    _physicalData->calculateTemperature(*_simulationBox);

    const auto mol1 = _simulationBox->getMolecule(0);
    const auto velocity_mol1_atom1 =
        mol1.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto velocity_mol1_atom2 =
        mol1.getAtomVelocity(AtomIndex{1}).numerical_value_in(angstrom / s);
    const auto mass_mol1_atom1 =
        mol1.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);
    const auto mass_mol1_atom2 =
        mol1.getAtomMass(AtomIndex{1}).numerical_value_in(g_per_mol);

    const auto mol2 = _simulationBox->getMolecule(1);
    const auto velocity_mol2_atom1 =
        mol2.getAtomVelocity(AtomIndex{0}).numerical_value_in(angstrom / s);
    const auto mass_mol2_atom1 =
        mol2.getAtomMass(AtomIndex{0}).numerical_value_in(g_per_mol);

    const auto kineticEnergyAtomicVector =
        mass_mol1_atom1 * velocity_mol1_atom1 * velocity_mol1_atom1 +
        mass_mol1_atom2 * velocity_mol1_atom2 * velocity_mol1_atom2 +
        mass_mol2_atom1 * velocity_mol2_atom1 * velocity_mol2_atom1;

    const auto nDOF = _simulationBox->getDegreesOfFreedom();
    // K per (g/mol * A^2/s^2)
    const auto temperatureFactor = ((1.0 * g_per_mol) * (1.0 * angstrom / s) *
                                    (1.0 * angstrom / s) / gas_constant)
                                       .numerical_value_in(K);

    EXPECT_NEAR(
        _physicalData->get_temperature()
            .quantity_from_zero()
            .numerical_value_in(K),
        sum(kineticEnergyAtomicVector) * temperatureFactor /
            static_cast<double>(nDOF),
        1e-15
    );
}

/**
 * @brief test reset function
 *
 */
TEST_F(TestPhysicalData, reset)
{
    _physicalData->set_kineticEnergy(1.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(1.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(1.0 * kcal_per_mol);
    _physicalData->set_intraCoulombEnergy(1.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(1.0 * kcal_per_mol);

    _physicalData->set_bondEnergy(1.0 * kcal_per_mol);
    _physicalData->set_angleEnergy(1.0 * kcal_per_mol);
    _physicalData->set_dihedralEnergy(1.0 * kcal_per_mol);
    _physicalData->set_improperEnergy(1.0 * kcal_per_mol);

    _physicalData->set_temperature(toTemperature(1.0));
    _physicalData->set_momentum(linalg::Vec3D(1.0) * g_per_mol * angstrom / s);
    _physicalData->set_volume(1.0 * angstrom3);
    _physicalData->set_density(1.0 * kg_per_litre);
    _physicalData->set_pressure(1.0 * bar);
    _physicalData->set_coupledPressure(1.0 * bar);
    _physicalData->set_virial(
        diagonalMatrix(linalg::Vec3D(1.0, 1.0, 1.0)) * kcal_per_mol
    );
    _physicalData->set_qmEnergy(1.0 * kcal_per_mol);
    _physicalData->set_qmReferenceEnergy(1.0 * kcal_per_mol);

    _physicalData->reset();

    EXPECT_EQ(_physicalData->get_kineticEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_coulombEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_nonCoulombEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_intraCoulombEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_intraNonCoulombEnergy(), 0.0 * kcal_per_mol);

    EXPECT_EQ(_physicalData->get_bondEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_angleEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_dihedralEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_improperEnergy(), 0.0 * kcal_per_mol);

    EXPECT_EQ(_physicalData->get_temperature(), toTemperature(0.0));
    EXPECT_EQ(
        _physicalData->get_momentum(),
        linalg::Vec3D(0.0) * g_per_mol * angstrom / s
    );
    EXPECT_EQ(_physicalData->get_volume(), 0.0 * angstrom3);
    EXPECT_EQ(_physicalData->get_density(), 0.0 * kg_per_litre);
    EXPECT_EQ(_physicalData->get_pressure(), 0.0 * bar);
    EXPECT_EQ(_physicalData->get_coupledPressure(), 0.0 * bar);
    EXPECT_EQ(
        _physicalData->get_virial(),
        linalg::tensor3D(0.0) * kcal_per_mol
    );
    EXPECT_EQ(_physicalData->get_qmEnergy(), 0.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_qmReferenceEnergy(), 0.0 * kcal_per_mol);
}

/**
 * @brief tests getTotalEnergy function
 *
 */
TEST_F(TestPhysicalData, getTotalEnergy)
{
    _physicalData->set_coulombEnergy(1.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(2.0 * kcal_per_mol);

    _physicalData->set_intraCoulombEnergy(3.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(4.0 * kcal_per_mol);

    _physicalData->set_bondEnergy(5.0 * kcal_per_mol);
    _physicalData->set_angleEnergy(6.0 * kcal_per_mol);
    _physicalData->set_dihedralEnergy(7.0 * kcal_per_mol);
    _physicalData->set_improperEnergy(8.0 * kcal_per_mol);

    _physicalData->set_kineticEnergy(9.0 * kcal_per_mol);
    _physicalData->set_qmEnergy(10.0 * kcal_per_mol);
    _physicalData->set_qmReferenceEnergy(11.0 * kcal_per_mol);

    EXPECT_EQ(
        _physicalData->getTotalEnergy(),
        1.0 * kcal_per_mol + 2.0 * kcal_per_mol + 5.0 * kcal_per_mol +
            6.0 * kcal_per_mol + 7.0 * kcal_per_mol + 8.0 * kcal_per_mol +
            9.0 * kcal_per_mol + 10.0 * kcal_per_mol - 11.0 * kcal_per_mol
    );
}

/**
 * @brief tests addIntraCoulombEnergy function
 */
TEST_F(TestPhysicalData, addIntraCoulombEnergy)
{
    _physicalData->set_coulombEnergy(0.0 * kcal_per_mol);
    _physicalData->set_intraCoulombEnergy(0.0 * kcal_per_mol);
    _physicalData->addIntraCoulombEnergy(1.0 * kcal_per_mol);

    EXPECT_EQ(_physicalData->get_intraCoulombEnergy(), 1.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_coulombEnergy(), 1.0 * kcal_per_mol);
}

/**
 * @brief tests addIntraNonCoulombEnergy function
 */
TEST_F(TestPhysicalData, addIntraNonCoulombEnergy)
{
    _physicalData->set_nonCoulombEnergy(0.0 * kcal_per_mol);
    _physicalData->set_intraNonCoulombEnergy(0.0 * kcal_per_mol);
    _physicalData->addIntraNonCoulombEnergy(1.0 * kcal_per_mol);

    EXPECT_EQ(_physicalData->get_intraNonCoulombEnergy(), 1.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_nonCoulombEnergy(), 1.0 * kcal_per_mol);
}

/* ---------- Energy accumulators (add… functions) ---------- */

TEST_F(TestPhysicalData, addCoulombEnergyAccumulates)
{
    _physicalData->set_coulombEnergy(2.0 * kcal_per_mol);
    _physicalData->addCoulombEnergy(3.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_coulombEnergy(), 5.5 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addNonCoulombEnergyAccumulates)
{
    _physicalData->set_nonCoulombEnergy(1.0 * kcal_per_mol);
    _physicalData->addNonCoulombEnergy(0.25 * kcal_per_mol);
    _physicalData->addNonCoulombEnergy(0.25 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_nonCoulombEnergy(), 1.5 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addBondEnergyAccumulates)
{
    _physicalData->set_bondEnergy(0.0 * kcal_per_mol);
    _physicalData->addBondEnergy(1.0 * kcal_per_mol);
    _physicalData->addBondEnergy(2.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_bondEnergy(), 3.0 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addAngleEnergyAccumulates)
{
    _physicalData->set_angleEnergy(0.0 * kcal_per_mol);
    _physicalData->addAngleEnergy(1.0 * kcal_per_mol);
    _physicalData->addAngleEnergy(-0.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_angleEnergy(), 0.5 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addDihedralEnergyAccumulates)
{
    _physicalData->set_dihedralEnergy(0.0 * kcal_per_mol);
    _physicalData->addDihedralEnergy(1.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_dihedralEnergy(), 1.0 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addImproperEnergyAccumulates)
{
    _physicalData->set_improperEnergy(0.0 * kcal_per_mol);
    _physicalData->addImproperEnergy(1.5 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_improperEnergy(), 1.5 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addRingPolymerEnergyAccumulates)
{
    _physicalData->set_ringPolymerEnergy(0.0 * kcal_per_mol);
    _physicalData->addRingPolymerEnergy(2.0 * kcal_per_mol);
    _physicalData->addRingPolymerEnergy(3.0 * kcal_per_mol);
    EXPECT_EQ(_physicalData->get_ringPolymerEnergy(), 5.0 * kcal_per_mol);
}

TEST_F(TestPhysicalData, addVirialAccumulates)
{
    const auto virial0 =
        diagonalMatrix(linalg::Vec3D(1.0, 2.0, 3.0)) * kcal_per_mol;
    const auto virial1 =
        diagonalMatrix(linalg::Vec3D(0.5, 0.5, 0.5)) * kcal_per_mol;
    _physicalData->set_virial(virial0);
    _physicalData->addVirial(virial1);
    EXPECT_EQ(_physicalData->get_virial(), virial0 + virial1);
}

TEST(PhysicalDataKinetics, molecularPressureExcludesInternalMotion)
{
    for (const auto opposing : {false, true})
    {
        molsys::SimulationBox box;
        molsys::Molecule      molecule;
        for (const auto sign : {1.0, opposing ? -1.0 : 1.0})
        {
            auto atom = std::make_shared<molsys::Atom>();
            atom->setMass(1.0 * g_per_mol);
            atom->setPosition(linalg::Vec3D{0.0, 0.0, 0.0} * angstrom);
            atom->setVelocity(
                sign * linalg::Vec3D{1.0, 2.0, 0.0} * angstrom / s
            );
            molecule.addAtom(atom);
            box.addAtom(atom);
        }
        box.addMolecule(molecule);
        box.calculateTotalMass();
        physicalData::PhysicalData data;
        data.calculateKinetics(box);

        const auto expected = opposing ? linalg::tensor3D{0.0}
                                       : linalg::tensor3D{
                                             {2.0, 4.0, 0.0},
                                             {4.0, 8.0, 0.0},
                                             {0.0, 0.0, 0.0}
                                         };
        for (size_t i = 0; i < 3; ++i)
        {
            for (size_t j = 0; j < 3; ++j)
            {
                EXPECT_DOUBLE_EQ(
                    data.getKinEnergyMolTensor()[i][j].numerical_value_in(
                        kcal_per_mol
                    ),
                    expected[i][j] * KINETIC_ENERGY_FACTOR
                );
            }
        }
        EXPECT_DOUBLE_EQ(
            data.get_kineticEnergy().numerical_value_in(kcal_per_mol),
            10.0 * KINETIC_ENERGY_FACTOR
        );
    }
}

/**
 * @brief a default initialised PhysicalData has no garbage in it
 *
 * @details Vec3D and tensor3D do not initialise their components. The
 * PhysicalData that holds the averages is a member of the engine and default
 * initialised (`PhysicalData data;`, unlike `PhysicalData()`, which zeroes the
 * whole object first), so the first energy and momentum line of a run showed
 * whatever the memory held, now and then inf, which failed the regression test
 * of the smoke run. The storage is filled with 0xFF (a NaN as a double) before
 * the object is built into it.
 */
TEST_F(TestPhysicalData, defaultConstructedIsZero)
{
    using physicalData::PhysicalData;

    alignas(PhysicalData) std::array<std::byte, sizeof(PhysicalData)> storage;
    storage.fill(std::byte{0xFF});

    auto *data = new (storage.data()) PhysicalData;

    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_EQ(data->get_momentum()[i], Momentum::zero());
        EXPECT_EQ(data->get_angularMomentum()[i], AngularMomentum::zero());

        for (size_t j = 0; j < 3; ++j)
        {
            EXPECT_EQ(data->get_virial()[i][j], VirialPerMol::zero());
            EXPECT_EQ(data->get_stressTensor()[i][j], Stress::zero());
            EXPECT_EQ(
                data->getKinEnergyAtomTensor()[i][j],
                EnergyPerMol::zero()
            );
            EXPECT_EQ(
                data->getKinEnergyMolTensor()[i][j],
                EnergyPerMol::zero()
            );
        }
    }

    data->~PhysicalData();
}

TEST_F(TestPhysicalData, potentialEnergyLeavesOutKineticEnergy)
{
    _physicalData->set_kineticEnergy(10.0 * kcal_per_mol);
    _physicalData->set_coulombEnergy(-3.0 * kcal_per_mol);
    _physicalData->set_nonCoulombEnergy(1.5 * kcal_per_mol);

    EXPECT_DOUBLE_EQ(
        _physicalData->getPotentialEnergy().numerical_value_in(kcal_per_mol),
        _physicalData->getTotalEnergy().numerical_value_in(kcal_per_mol) - 10.0
    );
    EXPECT_DOUBLE_EQ(
        _physicalData->getPotentialEnergy().numerical_value_in(kcal_per_mol),
        6.5
    );   // + QM 9 - 1
}
