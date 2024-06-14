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

#ifndef _STEEPEST_DESCENT_HPP_

#define _STEEPEST_DESCENT_HPP_

#include "optimizer.hpp"

namespace optimizer
{
    /**
     * @class SteepestDescent
     *
     * @brief Steepest Descent optimizer
     *
     */
    class SteepestDescent : public Optimizer
    {
       public:
        SteepestDescent(const size_t, const double);

        SteepestDescent()        = default;
        ~SteepestDescent() final = default;
    };

}   // namespace optimizer

#endif   // _STEEPEST_DESCENT_HPP_