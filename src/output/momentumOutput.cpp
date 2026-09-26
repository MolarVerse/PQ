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

#include <format>   // for format

#include "physicalData.hpp"   // for PhysicalData

namespace out
{
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
     * @param physicalData the physical data of the system
     */
    void MomentumOutput::write(
        size_t                            step,
        const physicalData::PhysicalData &physicalData
    )
    {
        _fp << std::format("{:10d}\t", step);
        _fp << std::format("{:20.5e}\t", norm(physicalData.getMomentum()));
        _fp << std::format("{:20.5e}\t", physicalData.getMomentum()[0]);
        _fp << std::format("{:20.5e}\t", physicalData.getMomentum()[1]);
        _fp << std::format("{:20.5e}\t", physicalData.getMomentum()[2]);
        _fp << std::format(
            "{:20.5e}\t",
            norm(physicalData.getAngularMomentum())
        );
        _fp << std::format("{:20.5e}\t", physicalData.getAngularMomentum()[0]);
        _fp << std::format("{:20.5e}\t", physicalData.getAngularMomentum()[1]);
        _fp << std::format("{:20.5e}\n", physicalData.getAngularMomentum()[2]);

        _fp << std::flush;
    }

}   // namespace out
