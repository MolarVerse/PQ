#ifndef _TEST_FORCE_FIELD_NON_COULOMB_HPP_
#define _TEST_FORCE_FIELD_NON_COULOMB_HPP_

#include <gtest/gtest.h>

#include "forceFieldNonCoulomb.hpp"
#include "forceFieldNonCoulombImpl.hpp"
#include "lennardJonesPair.hpp"
#include "matrix.hpp"

class TestNonCoulombPotentialFF : public ::testing::Test
{
   protected:
    void SetUp() override
    {
        _nonCoulombPotential = new potential::ForceFieldNonCoulomb();
    }

    [[nodiscard]]
    linearAlgebra::Matrix<pq::SharedNonCoulPair> getNonCoulombPairsMatrix(
    ) const
    {
        return _nonCoulombPotential->_nonCoulPairsMatPtr->matrix;
    }

    void setNonCoulombPairsMatrix(
        const linearAlgebra::Matrix<pq::SharedNonCoulPair> &matrix
    )
    {
        _nonCoulombPotential->_nonCoulPairsMatPtr->matrix = matrix;
    }

    void setNonCoulombPairsMatrix(
        size_t                             row,
        size_t                             col,
        const potential::LennardJonesPair &pair
    )
    {
        _nonCoulombPotential->_nonCoulPairsMatPtr->matrix(row, col) =
            std::make_shared<potential::LennardJonesPair>(pair);
    }

    void TearDown() override { delete _nonCoulombPotential; }

    potential::ForceFieldNonCoulomb *_nonCoulombPotential;
};

#endif   // _TEST_FORCE_FIELD_NON_COULOMB_HPP_
