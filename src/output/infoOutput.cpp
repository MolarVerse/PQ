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

#include "infoOutput.hpp"

#include <format>

#include "energyColumns.hpp"
#include "globalTimer.hpp"
#include "timerId.hpp"
#include "units.hpp"

namespace out
{
    /**
     * @brief Prepare the info output file for writing
     *
     * @details Closes the current file pointer, opens the info output file,
     * and writes the header.
     */
    void InfoOutput::_prepare()
    {
        _getFilePointer().close();
        _openFile();
        _writeHeader();
    }

    /**
     * @brief Write the simulation time to the info output
     *
     * @param simulationTime the current simulation time of the system
     */
    void InfoOutput::_writeSimulationTime(const Time &simulationTime)
    {
        _writeLeft(
            simulationTime.numerical_value_in(ps),
            "SIMULATION-TIME",
            "ps"
        );
    }

    /**
     * @brief Write the effective step to the info output
     *
     * @param effectiveStep the current effective step of the simulation
     */
    void InfoOutput::_writeEffectiveStep(double effectiveStep)
    {
        _writeLeft(effectiveStep, "EFFECTIVE STEPS", "-");
    }

    /**
     * @brief Finish writing the info output
     *
     * @details writes the columns of EnergyColumns, the same list as the
     * energy file, two per row after the time cell
     *
     */
    void InfoOutput::_finish(
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        const EnergyColumns columns(physicalData, settings);

        auto left = false;

        for (const auto &column : columns.get())
        {
            _writeColumn(column, left);

            if (left && column.aloneInInfoRow)
            {
                _writeRight();
            }
            else
            {
                left = !left;
            }
        }

        _getFilePointer() << std::format("{:-^89}", "") << "\n\n";

        _getFilePointer().flush();
    }

    /**
     * @brief write info file with the current simulation time
     *
     * @param simulationTime
     * @param physicalData the physical data of the system
     * @param settings the simulation settings
     */
    void InfoOutput::write(
        const Time                       &simulationTime,
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        auto _ = scopedTimer(TimerId::Output, "InfoOutput");

        _prepare();

        _writeSimulationTime(simulationTime);

        _finish(physicalData, settings);
    }

    /**
     * @brief write info file with effective step
     *
     * @param effectiveStep
     * @param physicalData the physical data of the system
     * @param settings the simulation settings
     */
    void InfoOutput::write(
        double                            effectiveStep,
        const physicalData::PhysicalData &physicalData,
        const Settings                   &settings
    )
    {
        auto _ = scopedTimer(TimerId::Output, "InfoOutput");

        _prepare();

        _writeEffectiveStep(effectiveStep);

        _finish(physicalData, settings);
    }

    /**
     * @brief write header of info file
     *
     */
    void InfoOutput::_writeHeader()
    {
        _getFilePointer() << std::format("{:-^89}", "") << '\n';

        _getFilePointer() << '|' << std::format("{:^87}", "PQ info file") << '|'
                          << '\n';

        _getFilePointer() << std::format("{:-^89}", "") << '\n';
    }

    /**
     * @brief write left column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeLeft(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _getFilePointer(
        ) << std::format("|   {:<15} {:15.5f} {:<8} ", name, value, unit);
    }

    /**
     * @brief write left column of info file
     *
     * @param value
     * @param name
     * @param unit
     */
    void InfoOutput::_writeLeftInteger(
        double                  value,
        const std::string_view &name,
        const std::string_view &unit
    )
    {
        _getFilePointer() << std::format(
            "|   {:<15} {:15d} {:<8} ",
            name,
            static_cast<int>(value),
            unit
        );
    }

    /**
     * @brief write one energy column into the left or right half of a row
     *
     * @param column
     * @param left
     */
    void InfoOutput::_writeColumn(const EnergyColumn &column, bool left)
    {
        const auto value = column.format == EnergyColumnFormat::SCIENTIFIC
                               ? std::format("{:15.1e}", column.value)
                               : std::format("{:15.5f}", column.value);

        if (left)
        {
            _getFilePointer() << std::format(
                "|   {:<15} {} {:<8} ",
                column.name,
                value,
                column.unit
            );
        }
        else
        {
            _getFilePointer() << std::format(
                "{:<15} {} {:<8}   |\n",
                column.name,
                value,
                column.unit
            );
        }
    }

    /**
     * @brief write empty right column to info file
     *
     */
    void InfoOutput::_writeRight()
    {
        _getFilePointer() << "                                           |\n";
    }

}   // namespace out
