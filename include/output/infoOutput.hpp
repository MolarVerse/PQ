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

#ifndef OUTPUT__INFO_OUTPUT_HPP
#define OUTPUT__INFO_OUTPUT_HPP

#include <cstddef>

#include "energyColumns.hpp"
#include "output.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

struct Settings;   // forward declaration

namespace out
{

    /**
     * @brief InfoOutput inherits from Output
     *
     * @details Output file for info file
     *
     */
    class InfoOutput : public Output
    {
       public:
        using Output::Output;

        void write(
            std::size_t                       step,
            const physicalData::PhysicalData &physicalData,
            const Settings                   &settings
        );

       private:
        void _prepare();
        void _writeHeader();
        void _writeStep(std::size_t step);
        void _finish(
            const physicalData::PhysicalData &physicalData,
            const Settings                   &settings
        );
        void _writeColumn(const EnergyColumn &column, bool left);
        void _writeRight();
    };

}   // namespace out

#endif /* OUTPUT__INFO_OUTPUT_HPP */
