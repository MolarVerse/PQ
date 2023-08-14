// #include "exceptions.hpp"
// #include "forceField.hpp"
// #include "throwWithMessage.hpp"

// #include <gtest/gtest.h>

// /**
//  * @brief tests findBondTypeById function
//  *
//  */
// TEST(TestForceField, findBondTypeById)
// {
//     auto forceField = forceField::ForceField();
//     auto bondType   = forceField::BondType(0, 1.0, 1.0);

//     forceField.addBondType(bondType);

//     EXPECT_EQ(forceField.findBondTypeById(0), bondType);
// }

// /**
//  * @brief tests findBondTypeById function for not found error
//  *
//  */
// TEST(TestForceField, findBondTypeById_notFoundError)
// {
//     auto forceField = forceField::ForceField();

//     EXPECT_THROW_MSG(forceField.findBondTypeById(0),
//                      customException::TopologyException,
//                      "Bond type with id " + std::to_string(0) + " not found.");
// }

// /**
//  * @brief tests findAngleTypeById function
//  *
//  */
// TEST(TestForceField, findAngleTypeById)
// {
//     auto forceField = forceField::ForceField();
//     auto angleType  = forceField::AngleType(0, 1.0, 1.0);

//     forceField.addAngleType(angleType);

//     EXPECT_EQ(forceField.findAngleTypeById(0), angleType);
// }

// /**
//  * @brief tests findAngleTypeById function for not found error
//  *
//  */
// TEST(TestForceField, findAngleTypeById_notFoundError)
// {
//     auto forceField = forceField::ForceField();

//     EXPECT_THROW_MSG(forceField.findAngleTypeById(0),
//                      customException::TopologyException,
//                      "Angle type with id " + std::to_string(0) + " not found.");
// }

// /**
//  * @brief tests findDihedralTypeById function
//  *
//  */
// TEST(TestForceField, findDihedralTypeById)
// {
//     auto forceField   = forceField::ForceField();
//     auto dihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

//     forceField.addDihedralType(dihedralType);

//     EXPECT_EQ(forceField.findDihedralTypeById(0), dihedralType);
// }

// /**
//  * @brief tests findDihedralTypeById function for not found error
//  *
//  */
// TEST(TestForceField, findDihedralTypeById_notFoundError)
// {
//     auto forceField = forceField::ForceField();

//     EXPECT_THROW_MSG(forceField.findDihedralTypeById(0),
//                      customException::TopologyException,
//                      "Dihedral type with id " + std::to_string(0) + " not found.");
// }

// /**
//  * @brief tests findImproperDihedralTypeById function
//  *
//  */
// TEST(TestForceField, findImproperDihedralTypeById)
// {
//     auto forceField           = forceField::ForceField();
//     auto improperDihedralType = forceField::DihedralType(0, 1.0, 1.0, 1.0);

//     forceField.addImproperDihedralType(improperDihedralType);

//     EXPECT_EQ(forceField.findImproperDihedralTypeById(0), improperDihedralType);
// }

// /**
//  * @brief tests findImproperDihedralTypeById function for not found error
//  *
//  */
// TEST(TestForceField, findImproperDihedralTypeById_notFoundError)
// {
//     auto forceField = forceField::ForceField();

//     EXPECT_THROW_MSG(forceField.findImproperDihedralTypeById(0),
//                      customException::TopologyException,
//                      "Improper dihedral type with id " + std::to_string(0) + " not found.");
// }

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

// //     EXPECT_EQ(forceField.getNonCoulombicPairsVector().size(), 1);
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

//     EXPECT_EQ(forceField.getNonCoulombicPairsVector().size(), 0);
// }

// /**
//  * @brief tests determineInternalGlobalVdwTypes function
//  *
//  */
// TEST(TestForceField, determineInternalGlobalVdwTypes)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));

//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});

//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     EXPECT_EQ(forceField.getNonCoulombicPairsVector()[0]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsVector()[0]->getInternalType2(), 2);
//     EXPECT_EQ(forceField.getNonCoulombicPairsVector()[1]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsVector()[1]->getInternalType2(), 1);
// }

// /**
//  * @brief tests getSelfInteractionNonCoulombicPairs function
//  *
//  */
// TEST(TestForceField, getSelfInteractionNonCoulombicPairs)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair3 = forceField::LennardJonesPair(2, 2, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair4 = forceField::LennardJonesPair(5, 5, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair3));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair4));

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     auto selfInteractionNonCoulombicPairs = forceField.getSelfInteractionNonCoulombicPairs();

//     EXPECT_EQ(selfInteractionNonCoulombicPairs.size(), 2);
// }

// /**
//  * @brief tests fillDiagonalElementsOfNonCoulombicPairsMatrix function
//  *
//  */
// TEST(TestForceField, fillDiagonalElementsOfNonCoulombicPairsMatrix)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 1, 2.0, 1.0, 1.0);
//     nonCoulombicPair1.setInternalType1(0);
//     nonCoulombicPair1.setInternalType2(0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(5, 5, 2.0, 1.0, 1.0);
//     nonCoulombicPair2.setInternalType1(9);
//     nonCoulombicPair2.setInternalType2(9);

//     std::vector<std::shared_ptr<forceField::NonCoulombPair>> diagonalElements = {
//         std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1),
//         std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2)};

//     forceField.fillDiagonalElementsOfNonCoulombicPairsMatrix(diagonalElements);

//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix().rows(), 2);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix().cols(), 2);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][0]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][0]->getInternalType2(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][1]->getInternalType1(), 9);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][1]->getInternalType2(), 9);
// }

// /**
//  * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if only one type is found
//  *
//  */
// TEST(TestForceField, findNonCoulombicPairByInternalTypes_findOneType)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     auto nonCoulombicPair = forceField.findNonCoulombicPairByInternalTypes(0, 2);
//     EXPECT_EQ((*nonCoulombicPair)->getInternalType1(), 0);
//     EXPECT_EQ((*nonCoulombicPair)->getInternalType2(), 2);
// }

// /**
//  * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if no type is found
//  *
//  */
// TEST(TestForceField, findNonCoulombicPairByInternalTypes_findNothing)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     auto nonCoulombicPair = forceField.findNonCoulombicPairByInternalTypes(0, 3);
//     EXPECT_EQ(nonCoulombicPair, std::nullopt);
// }

// /**
//  * @brief tests fillOffDiagonalElementsOfNonCoulombicPairsMatrix function if multiple types are found
//  *
//  */
// TEST(TestForceField, findNonCoulombicPairByInternalTypes_findMultipleTypes)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(1, 5, 2.0, 5.0, 1.0);
//     auto nonCoulombicPair3 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair3));

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     EXPECT_THROW_MSG(forceField.findNonCoulombicPairByInternalTypes(0, 2),
//                      customException::ParameterFileException,
//                      "Non coulombic pair with global van der waals types 1 and 5 is defined twice in the parameter file.");
// }

// /**
//  * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is not found
//  *
//  */
// TEST(TestForceField, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_ElementNotFound)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 5, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));

//     forceField.initNonCoulombicPairsMatrix(2);

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     EXPECT_THROW_MSG(
//         forceField.fillNonDiagonalElementsOfNonCoulombicPairsMatrix(),
//         customException::ParameterFileException,
//         "Not all combinations of global van der Waals types are defined in the parameter file - and no mixing rules were
//         chosen");
// }

// /**
//  * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found with lower index first
//  *
//  */
// TEST(TestForceField, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithLowerIndexFirst)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));

//     forceField.initNonCoulombicPairsMatrix(2);

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);
//     forceField.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 1);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 1);
// }

// /**
//  * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found with higher index first
//  *
//  */
// TEST(TestForceField, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithHigherIndexFirst)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(2, 1, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));

//     forceField.initNonCoulombicPairsMatrix(2);

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);
//     forceField.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 1);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 1);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 0);
// }

// /**
//  * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found for both index combinations with
//  * same parameters
//  *
//  */
// TEST(TestForceField, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withSameParams)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(2, 1, 2.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));

//     forceField.initNonCoulombicPairsMatrix(2);

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);
//     forceField.fillNonDiagonalElementsOfNonCoulombicPairsMatrix();

//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[0][1]->getInternalType2(), 1);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType1(), 0);
//     EXPECT_EQ(forceField.getNonCoulombicPairsMatrix()[1][0]->getInternalType2(), 1);
// }

// /**
//  * @brief tests fillNonDiagonalElementsOfNonCoulombicPairsMatrix function if element is found for both index combinations with
//  * different parameters
//  *
//  */
// TEST(TestForceField, fillNonDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withDifferentParams)
// {
//     auto forceField        = forceField::ForceField();
//     auto nonCoulombicPair1 = forceField::LennardJonesPair(1, 2, 2.0, 1.0, 1.0);
//     auto nonCoulombicPair2 = forceField::LennardJonesPair(2, 1, 5.0, 1.0, 1.0);

//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair1));
//     forceField.addNonCoulombicPair(std::make_shared<forceField::NonCoulombPair>(nonCoulombicPair2));

//     forceField.initNonCoulombicPairsMatrix(2);

//     // these two lines were already tested in TestForceField_determineInternalGlobalVdwTypes
//     std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
//     forceField.determineInternalGlobalVdwTypes(externalToInternalTypes);

//     EXPECT_THROW_MSG(
//         forceField.fillNonDiagonalElementsOfNonCoulombicPairsMatrix(),
//         customException::ParameterFileException,
//         "Non-coulombic pairs with global van der Waals types 1, 2 and 2, 1 in the parameter file have different parameters");
// }

// int main(int argc, char **argv)
// {
//     ::testing::InitGoogleTest(&argc, argv);
//     return ::RUN_ALL_TESTS();
// }