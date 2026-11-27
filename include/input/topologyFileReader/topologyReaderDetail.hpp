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

#ifndef INPUT__TOPOLOGY_FILE_READER__TOPOLOGY_READER_DETAIL_HPP
#define INPUT__TOPOLOGY_FILE_READER__TOPOLOGY_READER_DETAIL_HPP

#include <mstd/types/ref.hpp>
#include <mstd/utils.hpp>

#include "simulationBox.hpp"

namespace input::topology::detail
{
    /**
     * @brief builds the atom index lookup of the simulation box and clears it
     * again when the topology file is read (or reading throws)
     */
    class AtomIndexLookupGuard
    {
        MSTD_RAII_NO_MOVE(AtomIndexLookupGuard);

       private:
        mstd::Ref<molsys::SimulationBox> _simulationBox;

       public:
        explicit AtomIndexLookupGuard(molsys::SimulationBox &simulationBox);

        ~AtomIndexLookupGuard();
    };
}   // namespace input::topology::detail

#endif   // INPUT__TOPOLOGY_FILE_READER__TOPOLOGY_READER_DETAIL_HPP
