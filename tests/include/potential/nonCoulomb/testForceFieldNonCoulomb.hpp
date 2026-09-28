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
        _nonCoulombPotential = new pot::ForceFieldNonCoulomb();
    }

    [[nodiscard]]
    linalg::Matrix<
        std::shared_ptr<pot::NonCoulombPair>> _getNonCoulombPairsMatrix() const
    {
        return _getNonCoulombPairsMatrix(*_nonCoulombPotential);
    }

    [[nodiscard]]
    static linalg::
        Matrix<std::shared_ptr<pot::NonCoulombPair>> _getNonCoulombPairsMatrix(
            const pot::ForceFieldNonCoulomb &potential
        )
    {
        return potential._nonCoulPairsMatPtr->matrix;
    }

    void _setNonCoulombPairsMatrix(
        const linalg::Matrix<std::shared_ptr<pot::NonCoulombPair>> &matrix
    )
    {
        _setNonCoulombPairsMatrix(*_nonCoulombPotential, matrix);
    }

    static void _setNonCoulombPairsMatrix(
        pot::ForceFieldNonCoulomb                                  &potential,
        const linalg::Matrix<std::shared_ptr<pot::NonCoulombPair>> &matrix
    )
    {
        potential._nonCoulPairsMatPtr->matrix = matrix;
    }

    void _setNonCoulombPairsMatrix(
        size_t                       row,
        size_t                       col,
        const pot::LennardJonesPair &pair
    )
    {
        _setNonCoulombPairsMatrix(*_nonCoulombPotential, row, col, pair);
    }

    static void _setNonCoulombPairsMatrix(
        pot::ForceFieldNonCoulomb   &potential,
        const size_t                 row,
        const size_t                 col,
        const pot::LennardJonesPair &pair
    )
    {
        potential._nonCoulPairsMatPtr->matrix(row, col) =
            std::make_shared<pot::LennardJonesPair>(pair);
    }

    void TearDown() override { delete _nonCoulombPotential; }

    pot::ForceFieldNonCoulomb *_nonCoulombPotential;
};

#endif   // _TEST_FORCE_FIELD_NON_COULOMB_HPP_
