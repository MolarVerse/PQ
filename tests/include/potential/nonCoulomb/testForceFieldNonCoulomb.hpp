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
    linearAlgebra::Matrix<
        std::shared_ptr<pot::NonCoulombPair>> getNonCoulombPairsMatrix() const
    {
        return getNonCoulombPairsMatrix(*_nonCoulombPotential);
    }

    [[nodiscard]] static linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>> getNonCoulombPairsMatrix(
        const pot::ForceFieldNonCoulomb &potential
    )
    {
        return potential._nonCoulPairsMatPtr->matrix;
    }

    void setNonCoulombPairsMatrix(
        const linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>
            &matrix
    )
    {
        setNonCoulombPairsMatrix(*_nonCoulombPotential, matrix);
    }

    static void setNonCoulombPairsMatrix(
        pot::ForceFieldNonCoulomb &potential,
        const linearAlgebra::Matrix<std::shared_ptr<pot::NonCoulombPair>>
            &matrix
    )
    {
        potential._nonCoulPairsMatPtr->matrix = matrix;
    }

    void setNonCoulombPairsMatrix(
        size_t                       row,
        size_t                       col,
        const pot::LennardJonesPair &pair
    )
    {
        setNonCoulombPairsMatrix(*_nonCoulombPotential, row, col, pair);
    }

    static void setNonCoulombPairsMatrix(
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
