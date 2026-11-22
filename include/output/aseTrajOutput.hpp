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

#ifndef OUTPUT__ASE_TRAJ_OUTPUT_HPP
#define OUTPUT__ASE_TRAJ_OUTPUT_HPP

#include <cstddef>
#include <cstdint>
#include <mstd/utils.hpp>
#include <string>
#include <string_view>
#include <vector>

#include "output.hpp"
#include "ulmWriter.hpp"

namespace out
{
    struct Frame;   // forward declaration

    /**
     * @brief ASE trajectory (.traj) with positions, cell, momenta, charges,
     * energy and forces of each frame in ASE units
     *
     * @details readable with ase.io.read without PQ
     */
    class AseTrajOutput : public Output
    {
       private:
        ulm::Writer               _writer;
        ulm::JsonBuilder          _json;
        std::vector<double>       _buffer;
        std::vector<std::int64_t> _numbers;
        std::size_t               _nAtoms = 0;

       public:
        explicit AseTrajOutput(std::string filename);
        ~AseTrajOutput();

        MSTD_NO_COPY(AseTrajOutput)
        MSTD_NO_MOVE(AseTrajOutput)

        void write(const Frame &frame, size_t step);

       private:
        void _checkFrame(const Frame &frame);
        void _addSystem(const Frame &frame);
        void _addBuffer(std::string_view key, ulm::Shape shape);
    };

}   // namespace out

#endif   // OUTPUT__ASE_TRAJ_OUTPUT_HPP
