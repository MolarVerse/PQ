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

#include "momentumOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData
#include "vector3d.hpp"       // for Vec3D, norm

#include <format>    // for format
#include <fstream>   // for basic_ostream, ofstream
#include <string>    // for operator<<

using output::MomentumOutput;

/**
 * @brief Write the momentum output
 *
 * @details The momentum output is written in the following format:
 * - step
 * - norm of momentum
 * - momentum x
 * - momentum y
 * - momentum z
 * - norm of angular momentum
 * - angular momentum x
 * - angular momentum y
 * - angular momentum z
 *
 * @param step
 * @param data
 */
void MomentumOutput::write(const size_t step, const physicalData::PhysicalData &data)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.5e}\t", norm(data.getMomentum()));
    _fp << std::format("{:20.5e}\t", data.getMomentum()[0]);
    _fp << std::format("{:20.5e}\t", data.getMomentum()[1]);
    _fp << std::format("{:20.5e}\t", data.getMomentum()[2]);
    _fp << std::format("{:20.5e}\t", norm(data.getAngularMomentum()));
    _fp << std::format("{:20.5e}\t", data.getAngularMomentum()[0]);
    _fp << std::format("{:20.5e}\t", data.getAngularMomentum()[1]);
    _fp << std::format("{:20.5e}\n", data.getAngularMomentum()[2]);
}