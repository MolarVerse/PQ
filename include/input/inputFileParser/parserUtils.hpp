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

#ifndef _PARSER_UTILS_HPP_

#define _PARSER_UTILS_HPP_

namespace input
{
    // somewhere central, e.g. a small "functional utils" header
    template <typename T, typename Ret, typename... Args>
    auto bindMember(Ret (T::*method)(Args...), T *obj);

    // const-qualified overload, for const member functions
    template <typename T, typename Ret, typename... Args>
    auto bindMember(Ret (T::*method)(Args...) const, const T *obj);

}   // namespace input

#ifndef _PARSER_UTILS_TPP_
#include "parserUtils.tpp"   // IWYU pragma: export
#endif

#endif   // _PARSER_UTILS_HPP_
