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

#ifndef MANOSTAT__MANOSTAT_FACTORY_HPP
#define MANOSTAT__MANOSTAT_FACTORY_HPP

#include <memory>

class ManostatSettings;

namespace manostat
{
    class Manostat;

    /**
     * @brief Factory class for creating Manostat instances
     *
     * This class provides a static method to create Manostat objects based on
     * the provided ManostatSettings. It prevents direct instantiation.
     */
    class ManostatFactory
    {
       public:
        [[nodiscard]]
        static std::unique_ptr<Manostat> createManostat(
            const ManostatSettings& settings
        );

       private:
        ManostatFactory() = default;

        static std::unique_ptr<Manostat> _setupBerendsenManostat(
            const ManostatSettings& settings
        );
        static std::unique_ptr<Manostat> _setupStochasticRescalingManostat(
            const ManostatSettings& settings
        );
    };
}   // namespace manostat

#endif   // MANOSTAT__MANOSTAT_FACTORY_HPP
