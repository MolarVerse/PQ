#include "exceptions.hpp"
#include "forceField.hpp"
#include "throwWithMessage.hpp"

#include <gtest/gtest.h>

/**
 * @brief tests findBondTypeById function
 *
 */
TEST(TestForceField, findBondTypeById)
{
    auto forceField = forceField::ForceField();
    auto bondType   = forceField::BondType(0, 1.0, 1.0);

    forceField.addBondType(bondType);

    EXPECT_EQ(forceField.findBondTypeById(0), bondType);
}

/**
 * @brief tests findBondTypeById function for not found error
 *
 */
TEST(TestForceField, findBondTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findBondTypeById(0),
                     customException::TopologyException,
                     "Bond type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findAngleTypeById function
 *
 */
TEST(TestForceField, findAngleTypeById)
{
    auto forceField = forceField::ForceField();
    auto angleType  = forceField::AngleType(0, 1.0, 1.0);

    forceField.addAngleType(angleType);

    EXPECT_EQ(forceField.findAngleTypeById(0), angleType);
}

/**
 * @brief tests findAngleTypeById function for not found error
 *
 */
TEST(TestForceField, findAngleTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findAngleTypeById(0),
                     customException::TopologyException,
                     "Angle type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findDihedralTypeById function
 *
 */
TEST(TestForceField, findDihedralTypeById)
{
    auto forceField   = forceField::ForceField();
    auto dihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

    forceField.addDihedralType(dihedralType);

    EXPECT_EQ(forceField.findDihedralTypeById(0), dihedralType);
}

/**
 * @brief tests findDihedralTypeById function for not found error
 *
 */
TEST(TestForceField, findDihedralTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findDihedralTypeById(0),
                     customException::TopologyException,
                     "Dihedral type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests findImproperDihedralTypeById function
 *
 */
TEST(TestForceField, findImproperDihedralTypeById)
{
    auto forceField           = forceField::ForceField();
    auto improperDihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

    forceField.addImproperDihedralType(improperDihedralType);

    EXPECT_EQ(forceField.findImproperDihedralTypeById(0), improperDihedralType);
}

/**
 * @brief tests findImproperDihedralTypeById function for not found error
 *
 */
TEST(TestForceField, findImproperDihedralTypeById_notFoundError)
{
    auto forceField = forceField::ForceField();

    EXPECT_THROW_MSG(forceField.findImproperDihedralTypeById(0),
                     customException::TopologyException,
                     "Improper dihedral type with id " + std::to_string(0) + " not found.");
}

/**
 * @brief tests deleteNotNeededNonCoulombicPairs function by deleting nothing
 *
 * @details if both vdw types of non coulombic pair are in vector of global vdw types, then non coulombic pair is need and not
 * deleted
 *
 */
TEST(TestForceField, deleteNotNeededNonCoulombicPairs_deleteNothing)
{
    auto forceField       = forceField::ForceField();
    auto nonCoulombicPair = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair));
    forceField.deleteNotNeededNonCoulombicPairs({1, 2, 3});

    EXPECT_EQ(forceField.getNonCoulombicPairsVector().size(), 1);
}

/**
 * @brief tests deleteNotNeededNonCoulombicPairs function by deleting one pair
 *
 * @details if both vdw types of non coulombic pair are in vector of global vdw types, then non coulombic pair is need and not
 * deleted
 *
 */
TEST(TestForceField, deleteNotNeededNonCoulombicPairs_deleteOnePair)
{
    auto forceField       = forceField::ForceField();
    auto nonCoulombicPair = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);

    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair));
    forceField.deleteNotNeededNonCoulombicPairs({1, 2, 3});

    EXPECT_EQ(forceField.getNonCoulombicPairsVector().size(), 0);
}

/**
 * @brief tests determineInternalGlobalVdwTypes function
 *
 */
TEST(TestForceField, determineInternalGlobalVdwTypes)
{
    auto forceField        = forceField::ForceField();
    auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair1));
    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair2));

    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});

    forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_EQ(forceField.getNonCoulombicPairsVector()[0]->getInternalType1(), 0);
    EXPECT_EQ(forceField.getNonCoulombicPairsVector()[0]->getInternalType2(), 2);
    EXPECT_EQ(forceField.getNonCoulombicPairsVector()[1]->getInternalType1(), 0);
    EXPECT_EQ(forceField.getNonCoulombicPairsVector()[1]->getInternalType2(), 1);
}

/**
 * @brief tests getSelfInteractionNonCoulombicPairs function
 *
 */
TEST(TestForceField, getSelfInteractionNonCoulombicPairs)
{
    auto forceField        = forceField::ForceField();
    auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
    auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);
    auto nonCoulombicPair3 = forceField::LennardJonesPair(2, 2, 2.0, 1.0, 1.0);
    auto nonCoulombicPair4 = forceField::LennardJonesPair(5, 5, 2.0, 1.0, 1.0);

    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair1));
    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair2));
    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair3));
    forceField.addNonCoulombicPair(std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair4));

    // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto selfInteractionNonCoulombicPairs = forceField.getSelfInteractionNonCoulombicPairs();

    EXPECT_EQ(selfInteractionNonCoulombicPairs.size(), 2);
}

TEST(TestForceField, fillDiagonalElementsOfNonCoulombicPairsMatrix)
{
    auto forceField        = forceField::ForceField();
    auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 1, 2.0, 1.0, 1.0);
    nonCoulombicPair1.setInternalType1(0);
    nonCoulombicPair1.setInternalType2(0);
    auto nonCoulombicPair2 = forceField::LennardJonesPair(5, 5, 2.0, 1.0, 1.0);
    nonCoulombicPair2.setInternalType1(9);
    nonCoulombicPair2.setInternalType2(9);

    std::vector<std::unique_ptr<forceField::NonCoulombicPair>> diagonalElements = {
        std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair1),
        std::make_unique<forceField::NonCoulombicPair>(nonCoulombicPair2)};

    forceField.fillDiagonalElementsOfNonCoulombicPairsMatrix(diagonalElements);

    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix().rows(), 2);
    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix().cols(), 2);
    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][0]->getInternalType1(), 0);
    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][0]->getInternalType2(), 0);
    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][1]->getInternalType1(), 9);
    EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][1]->getInternalType2(), 9);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}