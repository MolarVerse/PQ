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

#ifndef _PARSER_UTILS_TPP_

#define _PARSER_UTILS_TPP_

namespace input
{
    /**
     * @brief bindMember is a utility function that binds a member function to
     * an object, allowing it to be called as a regular function.
     *
     * @tparam T
     * @tparam Ret
     * @tparam Args
     * @param method
     * @param obj
     * @return auto
     */
    template <typename T, typename Ret, typename... Args>
    auto bindMember(Ret (T::*method)(Args...), T *obj)
    {
        return [obj, method](Args... args) -> Ret
        { return (obj->*method)(args...); };
    }

    /**
     * @brief bindMember is a utility function that binds a const member
     * function to a const object, allowing it to be called as a regular
     * function.
     *
     * @tparam T
     * @tparam Ret
     * @tparam Args
     * @param method
     * @param obj
     * @return auto
     */
    template <typename T, typename Ret, typename... Args>
    auto bindMember(Ret (T::*method)(Args...) const, const T *obj)
    {
        return [obj, method](Args... args) -> Ret
        { return (obj->*method)(args...); };
    }
}   // namespace input

#endif   // _PARSER_UTILS_TPP_
