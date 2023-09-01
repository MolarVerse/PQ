#include "angleType.hpp"          // for AngleType
#include "bondType.hpp"           // for BondType
#include "dihedralType.hpp"       // for DihedralType
#include "exceptions.hpp"         // for TopologyException
#include "forceField.hpp"         // for ForceField
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

#include "gtest/gtest.h"   // for Message, TestPartResult
#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), TEST, EXP...
#include <string>          // for allocator, operator+, to_string, cha...

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

// /**
//  * @brief tests deleteNotNeededNonCoulombicPairs function by deleting nothing
//  *
//  * @details if both vdw types of non coulombic pair are in vector of global vdw types, then non coulombic pair is need and not
//  * deleted
//  *
//  */
// // TEST(TestForceField, deleteNotNeededNonCoulombicPairs_deleteNothing)
// // {
// //     auto forceField       = forceField::ForceField();
// //     auto nonCoulombicPair = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

// //     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair));
// //     forceField.deleteNotNeededNonCoulombicPairs({1, 2, 3});

// //     EXPECT_EQ(forceField.getNonCoulombPairsVector().size(), 1);
// // }

// /**
//  * @brief tests deleteNotNeededNonCoulombicPairs function by deleting one pair
//  *
//  * @details if both vdw types of non coulombic pair are in vector of global vdw types, then non coulombic pair is need and not
//  * deleted
//  *
//  */
// TEST(TestForceField, deleteNotNeededNonCoulombicPairs_deleteOnePair)
// {
//     auto forceField       = forceField::ForceField();
//     auto nonCoulombicPair = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair));
//     forceField.deleteNotNeededNonCoulombicPairs({1, 2, 3});

//     EXPECT_EQ(forceField.getNonCoulombPairsVector().size(), 0);
// }

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return ::RUN_ALL_TESTS();
}