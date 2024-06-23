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

#include "stressOutput.hpp"

#include <format>    // for format
#include <fstream>   // for basic_ostream, ofstream
#include <string>    // for operator<<

#include "physicalData.hpp"   // for PhysicalData

using output::StressOutput;

/**
 * @brief Write the stress output
 *
 * @details The stress output is written in the following format:
 * - step
 * - s_xx
 * - s_xy
 * - s_xz
 * - s_yx
 * - s_yy
 * - s_yz
 * - s_zx
 * - s_zy
 * - s_zz
 *
 * @param step
 * @param data
 */
void StressOutput::write(
    const size_t                      step,
    const physicalData::PhysicalData &data
)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[0][0]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[0][1]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[0][2]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[1][0]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[1][1]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[1][2]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[2][0]);
    _fp << std::format("{:20.5e}\t", data.getStressTensor()[2][1]);
    _fp << std::format("{:20.5e}\n", data.getStressTensor()[2][2]);

    _fp << std::flush;
}