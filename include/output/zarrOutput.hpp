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

#ifndef OUTPUT__ZARR_OUTPUT_HPP
#define OUTPUT__ZARR_OUTPUT_HPP

#include <cstddef>
#include <filesystem>
#include <memory>
#include <mstd/utils.hpp>
#include <string>
#include <vector>

#include "unit_types.hpp"
#include "zarrWriter.hpp"

namespace out
{
    struct Frame;   // forward declaration

    inline constexpr std::size_t ZARR_CHUNK_FRAMES = 100;

    /**
     * @brief Zarr v3 trajectory (a directory) with the groups, arrays and
     * units of the H5MD output
     *
     * @details frames are written in chunks of ZARR_CHUNK_FRAMES; close writes the last
     * chunk that is not full yet
     */
    class ZarrOutput
    {
       private:
        std::filesystem::path _path;
        Time                  _timeStep;
        zarr::Codec           _codec;
        std::size_t           _nAtoms = 0;

        std::unique_ptr<zarr::Element<double>> _box;
        std::unique_ptr<zarr::Element<float>>  _position;
        std::unique_ptr<zarr::Element<float>>  _velocity;
        std::unique_ptr<zarr::Element<float>>  _force;
        std::unique_ptr<zarr::Element<float>>  _charge;
        std::unique_ptr<zarr::Element<double>> _potentialEnergy;

        std::vector<float> _buffer;

       public:
        ZarrOutput(const std::string &filename, bool overwrite, Time timeStep);
        ~ZarrOutput();

        MSTD_NO_COPY(ZarrOutput)
        MSTD_NO_MOVE(ZarrOutput)

        void write(const Frame &frame, size_t step);
        void close();

       private:
        void _init(const Frame &frame);
    };

}   // namespace out

#endif   // OUTPUT__ZARR_OUTPUT_HPP
