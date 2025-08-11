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

#include "hybridConfigurator.hpp"

#include <stdexcept>   // for domain_error

#include "atom.hpp"            // for Atom
#include "simulationBox.hpp"   // for SimulationBox

using namespace pq;
using namespace configurator;
using namespace simulationBox;

void HybridConfigurator::calculateInnerRegionCenter(SimBox& simBox)
{
    const auto& indices = simBox.getInnerRegionCenterAtomIndices();

    if (indices.empty())
        throw(std::domain_error(
            "Cannot calculate inner region center: no center atoms specified"
        ));

    Vec3D  center     = {0.0, 0.0, 0.0};
    double total_mass = 0.0;

    for (const auto index : indices)
    {
        const auto& atom  = simBox.getAtom(index);
        const auto  mass  = atom.getMass();
        center           += atom.getPosition() * mass;
        total_mass       += mass;
    }

    center /= total_mass;
    setInnerRegionCenter(center);
}

/********************************
 * standard getters and setters *
 ********************************/

/**
 * @brief get the center of the inner region of the hybrid calculation
 *
 * @return Vec3D innerRegionCenter
 */
Vec3D HybridConfigurator::getInnerRegionCenter() { return _innerRegionCenter; }

/**
 * @brief set the center of the inner region of the hybrid calculation
 *
 * @param innerRegionCenter
 */
void HybridConfigurator::setInnerRegionCenter(Vec3D innerRegionCenter)
{
    _innerRegionCenter = innerRegionCenter;
}
