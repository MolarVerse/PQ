#include "exceptions.hpp"
#include "forceFieldNonCoulomb.hpp"
#include "guffDatReader.hpp"
#include "lennardJonesPair.hpp"
#include "throwWithMessage.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests determineInternalGlobalVdwTypes function
 *
 */
TEST(TestPotential, determineInternalGlobalVdwTypes)
{
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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
    auto potential         = potential::ForceFieldNonCoulomb();
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}