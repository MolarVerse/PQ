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

#ifndef _GUFF_COEFFICIENTS_HPP_

#define _GUFF_COEFFICIENTS_HPP_

#include <vector>   // for vector

namespace constants
{
    static const std::vector<double> _SPC_FW_GUFF_COEFFICIENTS_OO_ = []
    {
        std::vector<double> coefficients(22, 0.0);
        coefficients[0] = 625.5024676571352;   // LJ C6 in A^6 kcal mol^-1
        coefficients[1] = 6.0;
        coefficients[2] = -629326.9774051674;   // LJ C12 in A^12 kcal mol^-1
        coefficients[3] = 12.0;
        return coefficients;
    }();

    static const std::vector<double> _QSPC_FW_GUFF_COEFFICIENTS_OO_ = []
    {
        std::vector<double> coefficients(22, 0.0);
        coefficients[0] = 625.5020652114152;   // LJ C6 in A^6 kcal mol^-1
        coefficients[1] = 6.0;
        coefficients[2] = -629326.5724987736;   // LJ C12 in A^12 kcal mol^-1
        coefficients[3] = 12.0;
        return coefficients;
    }();

    static const std::vector<double> _TIP3P_GUFF_COEFFICIENTS_OO_ = []
    {
        std::vector<double> coefficients(22, 0.0);
        coefficients[0] = 595.0676884276835;   // LJ C6 in A^6 kcal mol^-1
        coefficients[1] = 6.0;
        coefficients[2] = -582015.099443679;   // LJ C12 in A^12 kcal mol^-1
        coefficients[3] = 12.0;
        return coefficients;
    }();

    static const std::vector<double> _OPC3_GUFF_COEFFICIENTS_OO_ = []
    {
        std::vector<double> coefficients(22, 0.0);
        coefficients[0] = 668.637501352773;   // LJ C6 in A^6 kcal mol^-1
        coefficients[1] = 6.0;
        coefficients[2] = -683996.5615895836;   // LJ C12 in A^12 kcal mol^-1
        coefficients[3] = 12.0;
        return coefficients;
    }();

    static const std::vector<double> _ZERO_GUFF_COEFFICIENTS_(22, 0.0);

}   // namespace constants

#endif   // _GUFF_COEFFICIENTS_HPP_
