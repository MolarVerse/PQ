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

#ifndef OUTPUT__H5MD_OUTPUT_HPP
#define OUTPUT__H5MD_OUTPUT_HPP

#include <cstddef>
#include <memory>
#include <string>

namespace out
{
    struct Frame;   // forward declaration

    /**
     * @brief H5MD trajectory (HDF5) with positions, velocities, forces,
     * charges, box and potential energy of each frame in PQ units
     *
     * @details the HDF5 code lives in the source file only; without
     * BUILD_WITH_HDF5 the constructor throws
     */
    class H5mdOutput
    {
       private:
        struct Impl;
        std::unique_ptr<Impl> _impl;

       public:
        H5mdOutput(const std::string &filename, bool overwrite, double timeStep);
        ~H5mdOutput();

        H5mdOutput(const H5mdOutput &)            = delete;
        H5mdOutput &operator=(const H5mdOutput &) = delete;
        H5mdOutput(H5mdOutput &&)                 = delete;
        H5mdOutput &operator=(H5mdOutput &&)      = delete;

        void write(const Frame &frame, size_t step);
        void close();

        [[nodiscard]] static bool available();
    };

}   // namespace out

#endif   // OUTPUT__H5MD_OUTPUT_HPP
