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

#include "nonCoulomb/testForceFieldNonCoulomb.hpp"

#include <gtest/gtest.h>   // for Test, EXPECT_EQ, TestInfo

#include <memory>     // for make_shared, shared_ptr
#include <optional>   // for optional, nullopt
#include <utility>    // for move
#include <vector>     // for vector

#include "exceptions.hpp"             // for ParameterFileException
#include "forceFieldNonCoulomb.hpp"   // for ForceFieldNonCoulomb
#include "gtest/gtest.h"              // for Message, TestPartResult
#include "lennardJonesPair.hpp"       // for LennardJonesPair
#include "matrix.hpp"                 // for Matrix
#include "nonCoulombPair.hpp"         // for NonCoulombPair
#include "strongTypes.hpp"
#include "throwWithMessage.hpp"   // for EXPECT_THROW_MSG

TEST_F(TestNonCoulombPotentialFF, copyConstructorCopiesOwnedMatrix)
{
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(1)
    );
    const auto pair = pot::LennardJonesPair(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    setNonCoulombPairsMatrix(0, 0, pair);
    _nonCoulombPotential->setNonCoulombPairsVector(
        {std::make_shared<pot::LennardJonesPair>(pair)}
    );

    auto copy = pot::ForceFieldNonCoulomb(*_nonCoulombPotential);

    EXPECT_EQ(copy.getNonCoulombPairsVector().size(), 1);
    EXPECT_EQ(
        getNonCoulombPairsMatrix(copy)(0, 0),
        getNonCoulombPairsMatrix()(0, 0)
    );

    const auto replacement = pot::LennardJonesPair(
        ExtVdwType(1),
        ExtVdwType(1),
        3.0,
        LJParams{.c6 = 2.0, .c12 = 1.0}
    );
    setNonCoulombPairsMatrix(*_nonCoulombPotential, 0, 0, replacement);
    EXPECT_NE(
        getNonCoulombPairsMatrix(copy)(0, 0),
        getNonCoulombPairsMatrix()(0, 0)
    );
}

TEST_F(TestNonCoulombPotentialFF, copyAssignmentCopiesOwnedMatrix)
{
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(1)
    );
    const auto pair = pot::LennardJonesPair(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    setNonCoulombPairsMatrix(0, 0, pair);

    auto copy = pot::ForceFieldNonCoulomb();
    copy      = *_nonCoulombPotential;

    const auto  matrixElement = getNonCoulombPairsMatrix(copy)(0, 0);
    const auto* self          = &copy;
    copy                      = *self;
    EXPECT_EQ(getNonCoulombPairsMatrix(copy)(0, 0), matrixElement);
}

TEST_F(TestNonCoulombPotentialFF, moveOperationsTransferOwnedMatrix)
{
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(1)
    );
    const auto pair = pot::LennardJonesPair(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    setNonCoulombPairsMatrix(0, 0, pair);

    auto moved = pot::ForceFieldNonCoulomb(std::move(*_nonCoulombPotential));
    EXPECT_NE(getNonCoulombPairsMatrix(moved)(0, 0), nullptr);

    auto assigned = pot::ForceFieldNonCoulomb();
    assigned      = std::move(moved);
    EXPECT_NE(getNonCoulombPairsMatrix(assigned)(0, 0), nullptr);
}

/**
 * @brief tests determineInternalGlobalVdwTypes function
 *
 */
TEST_F(TestNonCoulombPotentialFF, determineInternalGlobalVdwTypes)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );

    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );

    const auto& nonCoulPairsVec =
        _nonCoulombPotential->getNonCoulombPairsVector();

    EXPECT_EQ(nonCoulPairsVec[0]->getInternalType1(), VdwType{0});
    EXPECT_EQ(nonCoulPairsVec[0]->getInternalType2(), VdwType{2});
    EXPECT_EQ(nonCoulPairsVec[1]->getInternalType1(), VdwType{0});
    EXPECT_EQ(nonCoulPairsVec[1]->getInternalType2(), VdwType{1});
}

/**
 * @brief tests fillDiagOfNonCoulPairsMatrix function
 *
 */
TEST_F(TestNonCoulombPotentialFF, fillDiagOfNonCoulPairsMatrix)
{
    auto nonCoulombicPair1 = pot::LennardJonesPair(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    nonCoulombicPair1.setInternalType1(VdwType{0});
    nonCoulombicPair1.setInternalType2(VdwType{0});
    auto nonCoulombicPair2 = pot::LennardJonesPair(
        ExtVdwType(9),
        ExtVdwType(9),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    nonCoulombicPair2.setInternalType1(VdwType{9});
    nonCoulombicPair2.setInternalType2(VdwType{9});

    std::vector<std::shared_ptr<pot::NonCoulombPair>> diagonalElements = {
        std::make_shared<pot::LennardJonesPair>(nonCoulombicPair1),
        std::make_shared<pot::LennardJonesPair>(nonCoulombicPair2)
    };

    _nonCoulombPotential->fillDiagOfNonCoulPairsMatrix(diagonalElements);

    EXPECT_EQ(getNonCoulombPairsMatrix().rows(), 2);
    EXPECT_EQ(getNonCoulombPairsMatrix().cols(), 2);
    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 0)->getInternalType1(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 0)->getInternalType2(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 1)->getInternalType1(), VdwType{9});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 1)->getInternalType2(), VdwType{9});
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if only
 * one type is found
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    findNonCoulombicPairByInternalTypesFindOneType
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );

    auto nonCoulombicPair =
        _nonCoulombPotential->findNonCoulPairByInternalTypes(
            VdwType{0},
            VdwType{2}
        );
    EXPECT_EQ((*nonCoulombicPair)->getInternalType1(), VdwType{0});
    EXPECT_EQ((*nonCoulombicPair)->getInternalType2(), VdwType{2});
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if no
 * type is found
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    findNonCoulombicPairByInternalTypesFindNothing
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );

    auto nonCoulombicPair =
        _nonCoulombPotential->findNonCoulPairByInternalTypes(
            VdwType{0},
            VdwType{3}
        );
    EXPECT_EQ(nonCoulombicPair, std::nullopt);
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * multiple types are found
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    findNonCoulombicPairByInternalTypesFindMultipleTypes
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 5.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );

    EXPECT_THROW_MSG(
        [[maybe_unused]] const auto dummy =
            _nonCoulombPotential
                ->findNonCoulPairByInternalTypes(VdwType{0}, VdwType{2}),
        exc::ParameterFileException,
        std::format(
            "Non coulombic pair with global van der waals types {} and {} is "
            "defined twice in the parameter file.",
            ExtVdwType{1}.toString(),
            ExtVdwType{5}.toString()
        )
    );
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * element is not found
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrixElementNotFound
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(3)
    );

    EXPECT_THROW_MSG(
        _nonCoulombPotential->fillOffDiagOfNonCoulPairsMatrix(),
        exc::ParameterFileException,
        "Not all combinations of global van der Waals types are defined in the "
        "parameter file - and no mixing rules were chosen"
    );
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * element is found with lower index first
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrixFoundOnlyPairWithLowerIndexFirst
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(3)
    );
    _nonCoulombPotential->fillOffDiagOfNonCoulPairsMatrix();

    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType1(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType2(), VdwType{1});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType1(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType2(), VdwType{1});
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * element is found with higher index first
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrixFoundOnlyPairWithHigherIndexFirst
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(1),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(5),
            ExtVdwType(1),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(5),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(3)
    );
    _nonCoulombPotential->fillOffDiagOfNonCoulPairsMatrix();

    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType1(), VdwType{1});
    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType2(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType1(), VdwType{1});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType2(), VdwType{0});
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * element is found for both index combinations with same parameters
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrixFoundBothPairsWithSameParams
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(1),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(3)
    );
    _nonCoulombPotential->fillOffDiagOfNonCoulPairsMatrix();

    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType1(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(0, 1)->getInternalType2(), VdwType{1});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType1(), VdwType{0});
    EXPECT_EQ(getNonCoulombPairsMatrix()(1, 0)->getInternalType2(), VdwType{1});
}

/**
 * @brief tests fillOffDiagOfNonCoulPairsMatrix function if
 * element is found for both index combinations with different parameters
 *
 */
TEST_F(
    TestNonCoulombPotentialFF,
    fillOffDiagonalElementsOfNonCoulombicPairsMatrixFoundBothPairsWithDifferentParams
)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(1),
            5.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );
    setNonCoulombPairsMatrix(
        linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>(3)
    );

    EXPECT_THROW_MSG(
        _nonCoulombPotential->fillOffDiagOfNonCoulPairsMatrix(),
        exc::ParameterFileException,
        std::format(
            "Non-coulombic pairs with global van der Waals types {}, {} and "
            "{}, {} in "
            "the parameter file have different parameters",
            ExtVdwType{1}.toString(),
            ExtVdwType{2}.toString(),
            ExtVdwType{2}.toString(),
            ExtVdwType{1}.toString()
        )
    );
}

/**
 * @brief tests getSelfInteractionNonCoulPairs function
 *
 */
TEST_F(TestNonCoulombPotentialFF, getSelfInteractionNonCoulPairs)
{
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(1),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(2),
            ExtVdwType(2),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );
    _nonCoulombPotential->addNonCoulombicPair(
        std::make_shared<pot::LennardJonesPair>(
            ExtVdwType(5),
            ExtVdwType(5),
            2.0,
            LJParams{.c6 = 1.0, .c12 = 1.0}
        )
    );

    // these two lines were already tested in
    // TestPotential_determineInternalGlobalVdwTypes
    std::unordered_map<ExtVdwType, VdwType> externalToInternalTypes(
        {{ExtVdwType{1}, VdwType{0}},
         {ExtVdwType{2}, VdwType{1}},
         {ExtVdwType{5}, VdwType{2}}}
    );
    _nonCoulombPotential->determineInternalGlobalVdwTypes(
        externalToInternalTypes
    );

    auto selfInteractionNonCoulombicPairs =
        _nonCoulombPotential->getSelfInteractionNonCoulPairs();

    EXPECT_EQ(selfInteractionNonCoulombicPairs.size(), 2);
}

/**
 * @brief tests sortNonCoulombicsPairs
 *
 */
TEST_F(TestNonCoulombPotentialFF, sortNonCoulombicsPairs)
{
    auto vector = std::vector<std::shared_ptr<pot::NonCoulombPair>>();

    auto pair1 = std::make_shared<pot::LennardJonesPair>(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    pair1->setInternalType1(VdwType{1});
    pair1->setInternalType2(VdwType{5});
    vector.push_back(pair1);
    auto pair2 = std::make_shared<pot::LennardJonesPair>(
        ExtVdwType(2),
        ExtVdwType(2),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    pair2->setInternalType1(VdwType{2});
    pair2->setInternalType2(VdwType{2});
    vector.push_back(pair2);
    auto pair3 = std::make_shared<pot::LennardJonesPair>(
        ExtVdwType(2),
        ExtVdwType(3),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    pair3->setInternalType1(VdwType{2});
    pair3->setInternalType2(VdwType{3});
    vector.push_back(pair3);
    auto pair4 = std::make_shared<pot::LennardJonesPair>(
        ExtVdwType(1),
        ExtVdwType(4),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    pair4->setInternalType1(VdwType{1});
    pair4->setInternalType2(VdwType{4});
    vector.push_back(pair4);

    _nonCoulombPotential->sortNonCoulombicsPairs(vector);

    EXPECT_EQ(vector[0]->getInternalType1(), VdwType{1});
    EXPECT_EQ(vector[0]->getInternalType2(), VdwType{4});
    EXPECT_EQ(vector[1]->getInternalType1(), VdwType{1});
    EXPECT_EQ(vector[1]->getInternalType2(), VdwType{5});
    EXPECT_EQ(vector[2]->getInternalType1(), VdwType{2});
    EXPECT_EQ(vector[2]->getInternalType2(), VdwType{2});
    EXPECT_EQ(vector[3]->getInternalType1(), VdwType{2});
    EXPECT_EQ(vector[3]->getInternalType2(), VdwType{3});

    auto pair5 = std::make_shared<pot::LennardJonesPair>(
        ExtVdwType(1),
        ExtVdwType(1),
        2.0,
        LJParams{.c6 = 1.0, .c12 = 1.0}
    );
    pair5->setInternalType1(VdwType{1});
    pair5->setInternalType2(VdwType{5});
    vector.push_back(pair5);

    EXPECT_THROW_MSG(
        _nonCoulombPotential->sortNonCoulombicsPairs(vector),
        exc::ParameterFileException,
        std::format(
            "Non-coulombic pairs with global van der Waals types {} and {} in "
            "the "
            "parameter file are defined twice",
            ExtVdwType{1}.toString(),
            ExtVdwType{1}.toString()
        )
    );
}
