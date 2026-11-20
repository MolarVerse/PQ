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

#include "energyOutput.hpp"

#include <cstddef>
#include <format>
#include <iterator>
#include <ostream>

#include "energyColumns.hpp"
#include "units.hpp"

namespace out
{

    /**
     * @brief Write energy file metadata
     *
     * @param timeStep simulation timestep in fs
     */
    void EnergyOutput::writeHeader(const Time &timeStep)
    {
        _writeMarker();
        _getFilePointer() << std::format("# timestep = {}\n", timeStep.in(fs));
    }

    /**
     * @brief Write the energy output
     *
     * @details the columns come from EnergyColumns, the same list as the info
     * file; the first call also writes the header lines
     *
     * @param step
     * @param physicalData the physical data of the system
     * @param settings The simulation settings containing relevant configuration
     * for output.
     */
    void EnergyOutput::write(
        size_t                            step,
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        using enum EnergyColumnFormat;

        const EnergyColumns columns(physicalData, settings);

        if (!_columnHeaderWritten)
        {
            _writeColumnHeader(columns);
        }

        auto out = std::ostreambuf_iterator<char>(_getFilePointer());

        std::format_to(out, "{:10d}", step);

        for (const auto &column : columns.get())
        {
            switch (column.format)
            {
                case SCIENTIFIC:
                    std::format_to(out, "\t{:20.5e}", column.value);
                    break;
                case TIME:
                    std::format_to(out, "\t{:12.5f}", column.value);
                    break;
                case FIXED:
                    std::format_to(out, "\t{:20.12f}", column.value);
                    break;
            }
        }

        _getFilePointer() << '\n' << std::flush;
    }

    /**
     * @brief Write the PQ energy file marker line once
     *
     */
    void EnergyOutput::_writeMarker()
    {
        if (_markerWritten)
        {
            return;
        }

        _getFilePointer() << energyFileMarker() << '\n';

        _markerWritten = true;
    }

    /**
     * @brief Write the marker (if not yet written) and the column names line
     *
     * @param columns
     */
    void EnergyOutput::_writeColumnHeader(const EnergyColumns &columns)
    {
        _writeMarker();

        _getFilePointer() << energyFileHeader(columns) << '\n';

        _columnHeaderWritten = true;
    }

}   // namespace out
