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

#ifndef _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_
#define _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_

#include "forceFieldNonCoulomb.hpp"
#include "matrix.hpp"

namespace pot
{
    class NonCoulombPair;   // forward declaration
}   // namespace pot

/**
 * @brief struct to hold the non-coulombic pairs matrix
 *
 */
struct pot::ForceFieldNonCoulomb::matrix
{
    linearAlgebra::Matrix<std::shared_ptr<NonCoulombPair>> matrix;
};

#endif   // _FORCE_FIELD_NON_COULOMB_PIMPL_HPP_
