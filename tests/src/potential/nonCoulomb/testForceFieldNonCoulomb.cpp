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

#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo

#include <algorithm>   // for max
#include <cstddef>     // for size_t
#include <map>         // for map
#include <memory>      // for make_shared, shared_ptr
#include <optional>    // for optional, nullopt
#include <vector>      // for vector

#include "exceptions.hpp"             // for ParameterFileException
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "gtest/gtest.h"              // for Message, TestPartResult
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "matrix.hpp"                 // for Matrix
#include "nonCoulombPair.hpp"         // for NonCoulombPair
#include "throwWithMessage.hpp"       // for EXPECT_THROW_MSG

/**
 * @brief tests determineInternalGlobalVdwTypes function
 *
 */
TEST(TestPotential, determineInternalGlobalVdwTypes)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));

    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});

    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_EQ(potential.getNonCoulombPairsVector()[0]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombPairsVector()[0]->getInternalType2(), 2);
    EXPECT_EQ(potential.getNonCoulombPairsVector()[1]->getInternalType1(), 0);
    EXPECT_EQ(potential.getNonCoulombPairsVector()[1]->getInternalType2(), 1);
}

/**
 * @brief tests fillDiagonalElementsOfNonCoulombPairsMatrix function
 *
 */
TEST(TestPotential, fillDiagonalElementsOfNonCoulombPairsMatrix)
{
    auto potential = potential::ForceFieldNonCoulomb();
    auto nonCoulombicPair1 =
        potential::LennardJonesPair(size_t(1), size_t(1), 2.0, 1.0, 1.0);
    nonCoulombicPair1.setInternalType1(0);
    nonCoulombicPair1.setInternalType2(0);
    auto nonCoulombicPair2 =
        potential::LennardJonesPair(size_t(9), size_t(9), 2.0, 1.0, 1.0);
    nonCoulombicPair2.setInternalType1(9);
    nonCoulombicPair2.setInternalType2(9);

    std::vector<std::shared_ptr<potential::NonCoulombPair>> diagonalElements = {
        std::make_shared<potential::LennardJonesPair>(nonCoulombicPair1),
        std::make_shared<potential::LennardJonesPair>(nonCoulombicPair2)
    };

    potential.fillDiagonalElementsOfNonCoulombPairsMatrix(diagonalElements);

    EXPECT_EQ(potential.getNonCoulombPairsMatrix().rows(), 2);
    EXPECT_EQ(potential.getNonCoulombPairsMatrix().cols(), 2);
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 0)->getInternalType1(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 0)->getInternalType2(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 1)->getInternalType1(),
        9
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 1)->getInternalType2(),
        9
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if only
 * one type is found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findOneType)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto nonCoulombicPair = potential.findNonCoulombicPairByInternalTypes(0, 2);
    EXPECT_EQ((*nonCoulombicPair)->getInternalType1(), 0);
    EXPECT_EQ((*nonCoulombicPair)->getInternalType2(), 2);
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if no
 * type is found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findNothing)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto nonCoulombicPair = potential.findNonCoulombicPairByInternalTypes(0, 3);
    EXPECT_EQ(nonCoulombicPair, std::nullopt);
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * multiple types are found
 *
 */
TEST(TestPotential, findNonCoulombicPairByInternalTypes_findMultipleTypes)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        5.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    EXPECT_THROW_MSG(
        [[maybe_unused]] const auto dummy =
            potential.findNonCoulombicPairByInternalTypes(0, 2),
        customException::ParameterFileException,
        "Non coulombic pair with global van der waals types 1 and 5 is defined "
        "twice in the parameter file."
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * element is not found
 *
 */
TEST(
    TestPotential,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrix_ElementNotFound
)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(3)
    );

    EXPECT_THROW_MSG(
        potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix(),
        customException::ParameterFileException,
        "Not all combinations of global van der Waals types are defined in the "
        "parameter file - and no mixing rules were chosen"
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * element is found with lower index first
 *
 */
TEST(
    TestPotential,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithLowerIndexFirst
)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(3)
    );
    potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix();

    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType1(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType2(),
        1
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType1(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType2(),
        1
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * element is found with higher index first
 *
 */
TEST(
    TestPotential,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrix_foundOnlyPairWithHigherIndexFirst
)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(1),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(5),
        size_t(1),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(5),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(3)
    );
    potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix();

    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType1(),
        1
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType2(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType1(),
        1
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType2(),
        0
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * element is found for both index combinations with same parameters
 *
 */
TEST(
    TestPotential,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withSameParams
)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(1),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(3)
    );
    potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix();

    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType1(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(0, 1)->getInternalType2(),
        1
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType1(),
        0
    );
    EXPECT_EQ(
        potential.getNonCoulombPairsMatrix()(1, 0)->getInternalType2(),
        1
    );
}

/**
 * @brief tests fillOffDiagonalElementsOfNonCoulombPairsMatrix function if
 * element is found for both index combinations with different parameters
 *
 */
TEST(
    TestPotential,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrix_foundBothPairs_withDifferentParams
)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(1),
        5.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);
    potential.setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<potential::NonCoulombPair>>(3)
    );

    EXPECT_THROW_MSG(
        potential.fillOffDiagonalElementsOfNonCoulombPairsMatrix(),
        customException::ParameterFileException,
        "Non-coulombic pairs with global van der Waals types 1, 2 and 2, 1 in "
        "the parameter file have different parameters"
    );
}

/**
 * @brief tests getSelfInteractionNonCoulombicPairs function
 *
 */
TEST(TestPotential, getSelfInteractionNonCoulombicPairs)
{
    auto potential = potential::ForceFieldNonCoulomb();

    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(2),
        2.0,
        1.0,
        1.0
    ));
    potential.addNonCoulombicPair(std::make_shared<potential::LennardJonesPair>(
        size_t(5),
        size_t(5),
        2.0,
        1.0,
        1.0
    ));

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::map<size_t, size_t> externalToInternalTypes({{1, 0}, {2, 1}, {5, 2}});
    potential.determineInternalGlobalVdwTypes(externalToInternalTypes);

    auto selfInteractionNonCoulombicPairs =
        potential.getSelfInteractionNonCoulombicPairs();

    EXPECT_EQ(selfInteractionNonCoulombicPairs.size(), 2);
}

/**
 * @brief tests sortNonCoulombicsPairs
 *
 */
TEST(TestPotential, sortNonCoulombicsPairs)
{
    auto potential = potential::ForceFieldNonCoulomb();

    auto vector = std::vector<std::shared_ptr<potential::NonCoulombPair>>();

    auto pair1 = std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(1),
        2.0,
        1.0,
        1.0
    );
    pair1->setInternalType1(1);
    pair1->setInternalType2(5);
    vector.push_back(pair1);
    auto pair2 = std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(2),
        2.0,
        1.0,
        1.0
    );
    pair2->setInternalType1(2);
    pair2->setInternalType2(2);
    vector.push_back(pair2);
    auto pair3 = std::make_shared<potential::LennardJonesPair>(
        size_t(2),
        size_t(3),
        2.0,
        1.0,
        1.0
    );
    pair3->setInternalType1(2);
    pair3->setInternalType2(3);
    vector.push_back(pair3);
    auto pair4 = std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(4),
        2.0,
        1.0,
        1.0
    );
    pair4->setInternalType1(1);
    pair4->setInternalType2(4);
    vector.push_back(pair4);

    potential.sortNonCoulombicsPairs(vector);

    EXPECT_EQ(vector[0]->getInternalType1(), 1);
    EXPECT_EQ(vector[0]->getInternalType2(), 4);
    EXPECT_EQ(vector[1]->getInternalType1(), 1);
    EXPECT_EQ(vector[1]->getInternalType2(), 5);
    EXPECT_EQ(vector[2]->getInternalType1(), 2);
    EXPECT_EQ(vector[2]->getInternalType2(), 2);
    EXPECT_EQ(vector[3]->getInternalType1(), 2);
    EXPECT_EQ(vector[3]->getInternalType2(), 3);

    auto pair5 = std::make_shared<potential::LennardJonesPair>(
        size_t(1),
        size_t(1),
        2.0,
        1.0,
        1.0
    );
    pair5->setInternalType1(1);
    pair5->setInternalType2(5);
    vector.push_back(pair5);

    EXPECT_THROW_MSG(
        potential.sortNonCoulombicsPairs(vector),
        customException::ParameterFileException,
        "Non-coulombic pairs with global van der Waals types 1 and 1 in the "
        "parameter file are defined twice"
    );
}