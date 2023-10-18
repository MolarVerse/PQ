/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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
    static constexpr char _AUTHOR_[] = "Jakob Gamper";
    static constexpr char _JOSEF_[]  = "Josef M. Gallmetzer";

    static constexpr char _EMAIL_[]        = "97gamjak@gmail.com";
    static constexpr char _COMPILE_DATE_[] = __DATE__ " " __TIME__;
    static constexpr char _VERSION_[]      = COMPILE_VERSION_;

}   // namespace sysinfo

#endif   // _SYSTEM_INFO_HPP_