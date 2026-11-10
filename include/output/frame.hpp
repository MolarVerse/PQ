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

#ifndef OUTPUT__FRAME_HPP
#define OUTPUT__FRAME_HPP

#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "box.hpp"
#include "strongTypes.hpp"
#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace thermostat
{
    class Thermostat;   // forward declaration
}   // namespace thermostat

namespace out
{
    /**
     * @brief copy of the data the trajectory and restart writers read, so
     * they can format it while the simulation box changes
     */
    struct Frame
    {
        struct Atom
        {
            std::string   name;
            MolType       moltype;
            std::size_t   indexInMolecule;
            linalg::Vec3D position;
            linalg::Vec3D velocity;
            linalg::Vec3D force;
            double        charge;
        };

        // (chi, zeta) of each chain link
        using ChainLink = std::pair<double, double>;

        std::unique_ptr<molsys::Box> box;
        std::vector<Atom>            atoms;
        std::vector<ChainLink>       noseHooverChain;

        explicit Frame(const molsys::SimulationBox &simulationBox);
        Frame(
            const molsys::SimulationBox  &simulationBox,
            const thermostat::Thermostat &thermostat
        );

        [[nodiscard]] double totalForce() const;
    };

}   // namespace out

#endif   // OUTPUT__FRAME_HPP
