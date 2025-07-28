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

#include "integrator.hpp"

#include <algorithm>    // for __for_each_fn, for_each
#include <functional>   // for identity

#include "atom.hpp"                                  // for Atom
#include "constants/conversionFactors.hpp"           // for _FS_TO_S_
#include "constants/internalConversionFactors.hpp"   // for _V_VERLET_VELOCITY_FACTOR_
#include "simulationBox.hpp"                         // for SimulationBox
#include "timingsSettings.hpp"                       // for TimingsSettings
#include "vector3d.hpp"                              // for operator*, Vector3D

using namespace integrator;
using namespace simulationBox;
using namespace settings;
using namespace constants;

/**
 * @brief Construct a new Integrator:: Integrator object
 *
 * @param integratorType
 */
Integrator::Integrator(const std::string_view integratorType)
    : _integratorType(integratorType)
{
}

/**
 * @brief integrates the velocities of a single atom
 *
 * @param molecule
 * @param index
 */
void Integrator::integrateVelocities(Atom *atom) const
{
    auto       velocity = atom->getVelocity();
    const auto force    = atom->getForce();
    const auto mass     = atom->getMass();
    const auto dt       = TimingsSettings::getTimeStep();

    velocity += dt * force / mass * _V_VERLET_VELOCITY_FACTOR_;

    atom->setVelocity(velocity);
}

/**
 * @brief integrates the positions of a single atom
 *
 * @param molecule
 * @param index
 * @param simBox
 */
void Integrator::integratePositions(
    Atom                *atom,
    const SimulationBox &simBox
) const
{
    auto       position = atom->getPosition();
    const auto velocity = atom->getVelocity();

    position += TimingsSettings::getTimeStep() * velocity * _FS_TO_S_;

    simBox.applyPBC(position);

    atom->setPosition(position);
}

/********************************
 * standard getters and setters *
 ********************************/

/**
 * @brief get the integrator type
 *
 * @return std::string_view
 */
std::string_view Integrator::getIntegratorType() const
{
    return _integratorType;
}