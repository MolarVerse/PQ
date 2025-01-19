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

#include "coordinates.hpp"

#include <cassert>

using namespace simulationBox;

/**
 * @brief resize host vectors
 *
 * @param nAtoms
 * @param nMolecules
 */
void Coordinates::resizeHostVectors(
    const size_t nAtoms,
    const size_t nMolecules
)
{
    assert(nMolecules > 0);
    assert(nAtoms >= nMolecules);

    _pos.resize(3 * nAtoms);
    _vel.resize(3 * nAtoms);
    _forces.resize(3 * nAtoms);
    _shiftForces.resize(3 * nAtoms);
    _oldPos.resize(3 * nAtoms);
    _oldVel.resize(3 * nAtoms);
    _oldForces.resize(3 * nAtoms);

    _comMolecules.resize(3 * nMolecules);
}

/**
 * @brief get pointer to positions
 *
 * @return pointer to positions
 */
Real* Coordinates::getPosPtr() { return _pos.data(); }

/**
 * @brief get pointer to velocities
 *
 * @return pointer to velocities
 */
Real* Coordinates::getVelPtr() { return _vel.data(); }

/**
 * @brief get pointer to forces
 *
 * @return pointer to forces
 */
Real* Coordinates::getForcesPtr() { return _forces.data(); }

/**
 * @brief get pointer to shift forces
 *
 * @return pointer to shift forces
 */
Real* Coordinates::getShiftForcesPtr() { return _shiftForces.data(); }

/**
 * @brief get pointer to old positions
 *
 * @return pointer to old positions
 */
Real* Coordinates::getOldPosPtr() { return _oldPos.data(); }

/**
 * @brief get pointer to old velocities
 *
 * @return pointer to old velocities
 */
Real* Coordinates::getOldVelPtr() { return _oldVel.data(); }

/**
 * @brief get pointer to old forces
 *
 * @return pointer to old forces
 */
Real* Coordinates::getOldForcesPtr() { return _oldForces.data(); }

/**
 * @brief get pointer to center of mass of molecules
 *
 * @return pointer to center of mass of molecules
 */
Real* Coordinates::getComMoleculesPtr() { return _comMolecules.data(); }

/**
 * @brief get positions
 *
 * @return positions
 */
std::vector<Real> Coordinates::getPos() const { return _pos; }

/**
 * @brief get velocities
 *
 * @return velocities
 */
std::vector<Real> Coordinates::getVel() const { return _vel; }

/**
 * @brief get forces
 *
 * @return forces
 */
std::vector<Real> Coordinates::getForces() const { return _forces; }

/**
 * @brief get shift forces
 *
 * @return shift forces
 */
std::vector<Real> Coordinates::getShiftForces() const { return _shiftForces; }

/**
 * @brief get old positions
 *
 * @return old positions
 */
std::vector<Real> Coordinates::getOldPos() const { return _oldPos; }

/**
 * @brief get old velocities
 *
 * @return old velocities
 */
std::vector<Real> Coordinates::getOldVel() const { return _oldVel; }

/**
 * @brief get old forces
 *
 * @return old forces
 */
std::vector<Real> Coordinates::getOldForces() const { return _oldForces; }

/**
 * @brief get center of mass of molecules
 *
 * @return center of mass of molecules
 */
std::vector<Real> Coordinates::getComMolecules() const { return _comMolecules; }