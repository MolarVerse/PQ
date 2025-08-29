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

#ifndef _SIMULATION_BOX_SETTINGS_HPP_

#define _SIMULATION_BOX_SETTINGS_HPP_

#include <cstddef>   // for size_t

namespace settings
{
    /**
     * @class enum InitVelocities
     */
    enum class InitVelocities : size_t
    {
        FALSE,
        TRUE,
        FORCE
    };

    /**
     * @class SimulationBoxSettings
     *
     * @brief static class to store settings of the simulation box
     *
     */
    class SimulationBoxSettings
    {
       private:
        static inline bool _isDensitySet   = false;
        static inline bool _isBoxSet       = false;
        static inline bool _isBoxTriclinic = false;

        static inline InitVelocities _initializeVelocities =
            InitVelocities::FALSE;

       public:
        SimulationBoxSettings()  = delete;
        ~SimulationBoxSettings() = delete;

        /********************
         * standard setters *
         ********************/

        static void setDensitySet(const bool densitySet);
        static void setBoxSet(const bool boxSet);
        static void setIsBoxTriclinic(const bool isBoxTriclinic);
        static void setInitializeVelocities(
            const InitVelocities initializeVelocities
        );

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static bool           getDensitySet();
        [[nodiscard]] static bool           getBoxSet();
        [[nodiscard]] static bool           isBoxTriclinic();
        [[nodiscard]] static InitVelocities getInitializeVelocities();
    };
}   // namespace settings

#endif   // _SIMULATION_BOX_SETTINGS_HPP_