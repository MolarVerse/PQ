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

#include "atom.hpp"

#include "box.hpp"                // for Box
#include "manostatSettings.hpp"   // for ManostatSettings
#include "vector3d.hpp"           // for Vec3D

using simulationBox::Atom;

/**
 * @brief scales the velocities of the atom in orthogonal space
 *
 * @param scalingFactor
 * @param box
 */
void Atom::scaleVelocityOrthogonalSpace(const linearAlgebra::tensor3D &scalingTensor, const simulationBox::Box &box)
{
    if (settings::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
        _velocity = box.transformIntoOrthogonalSpace(_velocity);

    _velocity = scalingTensor * _velocity;

    if (settings::ManostatSettings::getIsotropy() != settings::Isotropy::FULL_ANISOTROPIC)
        _velocity = box.transformIntoSimulationSpace(_velocity);
}