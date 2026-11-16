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

#ifndef OUTPUT__EXTXYZ_OUTPUT_HPP
#define OUTPUT__EXTXYZ_OUTPUT_HPP

#include <cstddef>

#include "output.hpp"
#include "textBuffer.hpp"

namespace out
{
    struct Frame;   // forward declaration

    /**
     * @brief extended xyz trajectory with positions, velocities, forces and
     * charges of each frame in PQ units
     */
    class ExtxyzOutput : public Output
    {
       private:
        TextBuffer _buffer;

       public:
        using Output::Output;

        void write(const Frame &frame, size_t step);
    };

}   // namespace out

#endif   // OUTPUT__EXTXYZ_OUTPUT_HPP
