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
#include <fstream>
#include <mstd/utils.hpp>
#include <span>
#include <string>
#include <vector>

namespace out
{
    struct Frame;   // forward declaration

    /**
     * @brief ASE trajectory (.traj, ULM format) with positions, cell,
     * momenta, charges, energy and forces of each frame in ASE units
     *
     * @details written natively, readable with ase.io.read without PQ
     */
    class AseTrajOutput
    {
       private:
        std::string               _filename;
        std::ofstream             _file;
        std::vector<std::int64_t> _offsets;
        std::int64_t              _offsetsPosition;
        std::int64_t              _nItems = 0;
        std::size_t               _nAtoms = 0;
        std::vector<double>       _values;
        std::vector<char>         _bytes;

       public:
        AseTrajOutput(const std::string &filename, bool overwrite);
        ~AseTrajOutput();

        MSTD_NO_COPY(AseTrajOutput)
        MSTD_NO_MOVE(AseTrajOutput)

        void write(const Frame &frame, size_t step);
        void close();

       private:
        [[nodiscard]] std::int64_t _position();
        [[nodiscard]] std::int64_t _align();

        void _writeInt(std::int64_t value);
        void _writeIntAt(std::int64_t value, std::int64_t position);
        void _addItem(std::int64_t jsonPosition);
        void _checkFile() const;

        template <typename T>
        void _writeRaw(std::span<const T> values);

        template <typename T>
        [[nodiscard]] std::string _writeArray(
            std::span<const T>           values,
            std::span<const std::size_t> shape
        );

        template <typename Get>
        [[nodiscard]] std::string _writeVectors(const Frame &frame, Get get);
    };

}   // namespace out

#endif   // OUTPUT__ASE_TRAJ_OUTPUT_HPP
