/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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

#ifndef _SIMULATION_BOX_SETTINGS_HPP_

#define _SIMULATION_BOX_SETTINGS_HPP_

namespace settings
{
    /**
     * @class SimulationBoxSettings
     *
     * @brief static class to store settings of the simulation box
     *
     */
    class SimulationBoxSettings
    {
      private:
        static inline bool _isDensitySet = false;
        static inline bool _isBoxSet     = false;

        static inline bool _initializeVelocities = false;

      public:
        SimulationBoxSettings()  = delete;
        ~SimulationBoxSettings() = delete;

        static void setDensitySet(const bool densitySet) { _isDensitySet = densitySet; }
        static void setBoxSet(const bool boxSet) { _isBoxSet = boxSet; }
        static void setInitializeVelocities(const bool initializeVelocities) { _initializeVelocities = initializeVelocities; }

        [[nodiscard]] static bool getDensitySet() { return _isDensitySet; }
        [[nodiscard]] static bool getBoxSet() { return _isBoxSet; }
        [[nodiscard]] static bool getInitializeVelocities() { return _initializeVelocities; }
    };
}   // namespace settings

#endif   // _SIMULATION_BOX_SETTINGS_HPP_