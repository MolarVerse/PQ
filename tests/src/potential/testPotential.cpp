#include "exceptions.hpp"
#include "guffDatReader.hpp"
#include "lennardJonesPair.hpp"
#include "potential.hpp"
#include "throwWithMessage.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests determineInternalGlobalVdwTypes function
 *
 */
TEST(TestPotential, determineInternalGlobalVdwTypes)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));

    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});

    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_EQ(potential.getNonCoulombicPairsVector()[0]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsVector()[0]->getInternalType2(), 2);
    EXPECT_EQ(potential.getNonCoulombicPairsVector()[1]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsVector()[1]->getInternalType2(), 1);
}

/**
 * @brief tests fillDiagonalElementsOfNonCoulombicPairsMatrix function
 *
 */
TEST(TestPotential, fillDiagonalElementsOfNonCoulombicPairsMatrix)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(1), 2.0, 1.0, 1.0);
    nonCoulombicPair1.setInternalType1(0);
    nonCoulombicPair1.setInternalType2(0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(1), 2.0, 1.0, 1.0);
    nonCoulombicPair2.setInternalType1(9);
    nonCoulombicPair2.setInternalType2(9);

    std::vector<std::shared_ptr<potential::NonCoulombPair>> diagonalElements = {
        std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1),
        std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2)};

    potential.fillDiagonalElementsOfNonCoulombicPairsMatrix(diagonalElements);

    EXPECT_EQ(potential.getNonCoulombicPairsMatrix().rows(), 2);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix().cols(), 2);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][0]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][0]->getInternalType2(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][1]->getInternalType1(), 9);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][1]->getInternalType2(), 9);
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if only one type is found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findOneType)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto nonCoulombicPair = potential.findNonCoulombicPairByInternalTypes(0, 2);
    EXPECT_EQ((*nonCoulombicPair)->getInternalType1(), 0);
    EXPECT_EQ((*nonCoulombicPair)->getInternalType2(), 2);
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if no type is found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findNothing)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto nonCoulombicPair = potential.findNonCoulombicPairByInternalTypes(0, 3);
    EXPECT_EQ(nonCoulombicPair, std::nullopt);
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if multiple types are found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findMultipleTypes)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 5.0, 1.0);
    auto nonCoulombicPair3 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair3));

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_THROW_MSG(potential.findNonCoulombicPairByInternalTypes(0, 2),
                     customException::ParameterFileException,
                     "Non coulombic pair with global van der waals types 1 and 5 is defined twice in the parameter file.");
}

/**
 * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is not found
 *
 */
TEST(TestPotential, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_ElementNotFound)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));

    potential.initNonCoulombicPairsMatrix(2);

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_THROW_MSG(
        potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix(),
        customException::ParameterFileException,
        "Not all combinations of global van der Waals types are defined in the parameter file - and no mixing rules were chosen");
}

/**
 * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found with lower index first
 *
 */
TEST(TestPotential, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithLowerIndexFirst)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));

    potential.initNonCoulombicPairsMatrix(2);

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 1);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 1);
}

/**
 * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found with higher index first
 *
 */
TEST(TestPotential, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithHigherIndexFirst)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(2), size_t(1), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));

    potential.initNonCoulombicPairsMatrix(2);

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 1);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 1);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 0);
}

/**
 * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found for both index combinations with
 * same parameters
 *
 */
TEST(TestPotential, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withSameParams)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(2), size_t(1), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));

    potential.initNonCoulombicPairsMatrix(2);

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 1);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 1);
}

/**
 * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found for both index combinations with
 * different parameters
 *
 */
TEST(TestPotential, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withDifferentParams)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(2), size_t(1), 5.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));

    potential.initNonCoulombicPairsMatrix(2);

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_THROW_MSG(
        potential.fillNonDiagonalElementsOfNonCoulombicPairsMatrix(),
        customException::ParameterFileException,
        "Non-coulombic pairs with global van der Waals types 1, 2 and 2, 1 in the parameter file have different parameters");
}

/**
 * @brief tests getSelfInteractionNonCoulombicPairs function
 *
 */
TEST(TestPotential, getSelfInteractionNonCoulombicPairs)
{
    auto potential         = potential::PotentialBruteForce();
    auto nonCoulombicPair1 = potential::LennardJonesPair(size_t(1), size_t(5), 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = potential::LennardJonesPair(size_t(1), size_t(2), 2.0, 1.0, 1.0);
    auto nonCoulombicPair3 = potential::LennardJonesPair(size_t(2), size_t(2), 2.0, 1.0, 1.0);
    auto nonCoulombicPair4 = potential::LennardJonesPair(size_t(5), size_t(5), 2.0, 1.0, 1.0);

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair3));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(nonCoulombicPair4));

    // these two lines were already tested in TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto selfInteractionNonCoulombicPairs = potential.getSelfInteractionNonCoulombicPairs();

    EXPECT_EQ(selfInteractionNonCoulombicPairs.size(), 2);
}

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

//     engine.getSimulationBox().setGuffCoefficients(1, 2, 0, 0, coefficients1);
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

// /**
//  * @brief tests cell list potential calculation
//  *
//  */
// TEST(TestPotential, cellList)
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

//     engine.getSimulationBox().setGuffCoefficients(1, 2, 0, 0, coefficients1);
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

//     auto potential = potential::PotentialCellList();

//     auto cell1 = simulationBox::Cell();
//     auto cell2 = simulationBox::Cell();
//     cell1.addMolecule(&(engine.getSimulationBox().getMolecule(0)));
//     cell2.addMolecule(&(engine.getSimulationBox().getMolecule(1)));
//     cell1.addNeighbourCell(&cell2);
//     cell1.addAtomIndices(std::vector<size_t>{0, 1});
//     cell2.addAtomIndices(std::vector<size_t>{0});
//     engine.getCellList().addCell(cell1);
//     engine.getCellList().addCell(cell2);

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