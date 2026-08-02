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
