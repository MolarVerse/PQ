#include <gtest/gtest.h>

// /**
//  * @brief tests brute force potential calculation
//  *
//  */
// TEST(TestPotential, bruteForce)
// {
//     auto engine = engine::Engine();

//     auto molecule1 = simulationBox::Molecule();
//     molecule1.setNumberOfAtoms(2);
//     molecule1.setMoltype(1);
//     molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
//     molecule1.addAtomPosition(linearAlgebra::Vec3D(1.0, 20.0, 30.0));
//     molecule1.addAtomForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
//     molecule1.addAtomForce(linearAlgebra::Vec3D(2.0, 3.0, 4.0));
//     molecule1.addAtomType(0);
//     molecule1.addAtomType(1);
//     molecule1.addExternalAtomType(0);
//     molecule1.addExternalAtomType(1);
//     molecule1.resizeAtomShiftForces();

//     auto molecule2 = simulationBox::Molecule();
//     molecule2.setNumberOfAtoms(1);
//     molecule2.setMoltype(2);
//     molecule2.addAtomPosition(linearAlgebra::Vec3D(10.0, 1.0, 1.0));
//     molecule2.addAtomForce(linearAlgebra::Vec3D(1.0, 1.0, 1.0));
//     molecule2.addAtomType(0);
//     molecule2.addExternalAtomType(0);
//     molecule2.resizeAtomShiftForces();

//     engine.getSimulationBox().addMolecule(molecule1);
//     engine.getSimulationBox().addMolecule(molecule2);
//     engine.getSimulationBox().addMoleculeType(molecule1);
//     engine.getSimulationBox().addMoleculeType(molecule2);

//     engine.getSimulationBox().setCoulombRadiusCutOff(1000.0);
//     engine.getSimulationBox().setBoxDimensions(linearAlgebra::Vec3D(100.0, 100.0, 100.0));

//     engine.getSettings().setGuffDatFilename("data/guffDatReader/guff.dat");
//     auto guffReader = readInput::GuffDatReader(engine);
//     guffReader.setupGuffMaps();

//     const auto coefficients1 = std::vector{1.0, 1.0, 1.0, 1.0};
//     const auto coefficients2 = std::vector{2.0, 1.0, 2.0, 1.0};

//     const auto coulombCoefficient     = 330.0;
//     const auto nonCoulombRadiusCutoff = 10.0;

//     engine.getPotential().getNonCoulombPotential().setGuffCoefficients(1, 2, 0, 0, coefficients1);
//     engine.getSimulationBox().setGuffCoefficients(1, 2, 1, 0, coefficients2);

//     engine.getSimulationBox().setNonCoulombRadiusCutOff(1, 2, 0, 0, nonCoulombRadiusCutoff);
//     engine.getSimulationBox().setNonCoulombRadiusCutOff(1, 2, 1, 0, nonCoulombRadiusCutoff);

//     engine.getSimulationBox().setCoulombCoefficient(1, 2, 0, 0, coulombCoefficient);
//     engine.getSimulationBox().setCoulombCoefficient(1, 2, 1, 0, coulombCoefficient);

//     engine.getSimulationBox().setGuffCoefficients(2, 1, 0, 0, coefficients1);
//     engine.getSimulationBox().setGuffCoefficients(2, 1, 0, 1, coefficients2);

//     engine.getSimulationBox().setNonCoulombRadiusCutOff(2, 1, 0, 0, nonCoulombRadiusCutoff);
//     engine.getSimulationBox().setNonCoulombRadiusCutOff(2, 1, 0, 1, nonCoulombRadiusCutoff);

//     engine.getSimulationBox().setCoulombCoefficient(2, 1, 0, 0, coulombCoefficient);
//     engine.getSimulationBox().setCoulombCoefficient(2, 1, 0, 1, coulombCoefficient);

//     auto potential = potential::PotentialBruteForce();
//     potential.setCoulombPotential(potential::GuffCoulomb());
//     potential.setNonCoulombPotential(potential::GuffLennardJones());
//     auto physicalData = physicalData::PhysicalData();

//     potential.calculateForces(engine.getSimulationBox(), physicalData, engine.getCellList());

//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(0)[0], -3.0740753285297435);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(0)[1], 1.0);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(0)[2], 1.0);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(1)[0], 1.9353726326765324);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(1)[1], 3.13643555323843);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(0).getAtomForce(1)[2], 4.2082437391533958);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(1).getAtomForce(0)[0], 5.13870269585321);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(1).getAtomForce(0)[1], 0.863564446761568);
//     EXPECT_DOUBLE_EQ(engine.getSimulationBox().getMolecule(1).getAtomForce(0)[2], 0.791756260846604);

//     EXPECT_DOUBLE_EQ(physicalData.getCoulombEnergy(), 45.879656919556552);
//     EXPECT_DOUBLE_EQ(physicalData.getNonCoulombEnergy(), 1.8816799638650823e-06);
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}