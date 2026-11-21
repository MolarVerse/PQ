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

#ifndef OUTPUT__ENERGY_COLUMNS_HPP
#define OUTPUT__ENERGY_COLUMNS_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>
#include <string_view>

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

struct Settings;   // forward declaration

namespace out
{
    /**
     * @brief number format of an energy column
     */
    enum class EnergyColumnFormat : std::uint8_t
    {
        FIXED,
        SCIENTIFIC,
        TIME
    };

    /**
     * @brief one column of the energy and info output
     *
     * @details aloneInInfoRow leaves the right half of its info row empty
     */
    struct EnergyColumn
    {
        std::string_view   name;
        std::string_view   unit;
        double             value          = 0.0;
        EnergyColumnFormat format         = EnergyColumnFormat::FIXED;
        bool               aloneInInfoRow = false;
    };

    /**
     * @class EnergyColumns
     *
     * @brief the active energy columns in output order, shared by the
     * energy (.en) and info (.info) output
     *
     */
    class EnergyColumns
    {
       public:
        static constexpr std::size_t maxColumns = 24;

       private:
        std::array<EnergyColumn, maxColumns> _columns{};
        std::size_t                          _size = 0;

       public:
        EnergyColumns(
            const physicalData::PhysicalData &physicalData,
            const Settings                   &settings
        );

        [[nodiscard]] std::span<const EnergyColumn> get() const;

       private:
        void _add(const EnergyColumn &column);
    };

    inline constexpr std::string_view energyStepName = "SIMULATION-TIME";
    inline constexpr std::string_view energyStepUnit = "step";

    [[nodiscard]] std::string energyColumnLabel(const EnergyColumn &column);
    [[nodiscard]] std::string energyFileMarker();
    [[nodiscard]] std::string energyFileHeader(const EnergyColumns &columns);

}   // namespace out

#endif   // OUTPUT__ENERGY_COLUMNS_HPP
