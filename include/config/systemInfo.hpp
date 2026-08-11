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

#ifndef _SYSTEM_INFO_HPP_

#define _SYSTEM_INFO_HPP_

#define COMPILE_VERSION_ _COMPILE_VERSION_

namespace sysinfo
{
    static constexpr auto* AUTHOR   = "Jakob Gamper";
    static constexpr auto* JOSEF    = "Josef M. Gallmetzer";
    static constexpr auto* STEFAN   = "Stefan Seiwald";
    static constexpr auto* BENJAMIN = "Benjamin Reitmair";
    static constexpr auto* ARMIN    = "Armin Penz";

    static constexpr auto* EMAIL        = "97gamjak@gmail.com";
    static constexpr auto* COMPILE_DATE = __DATE__ " " __TIME__;
    static constexpr auto* VERSION      = COMPILE_VERSION_;

}   // namespace sysinfo

#endif   // _SYSTEM_INFO_HPP_
