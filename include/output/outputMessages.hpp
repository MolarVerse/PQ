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

#include <string>   // for operator==

namespace output
{
    static constexpr char _WARNING_[]  = "WARNING: ";
    static constexpr char _INFO_[]     = "INFO:    ";
    static constexpr char _OUTPUT_[]   = "         ";
    static constexpr char _ANGSTROM_[] = "\u212b";

    std::string header();
    std::string endedNormally();

    std::string initialMomentumMessage(const double initialMomentum);

    std::string elapsedTimeMessage(const double elapsedTime);

    std::string setupMessage(const std::string &setup);
    std::string setupCompletedMessage();
    std::string readMessage(const std::string &message, const std::string &file);

}   // namespace output