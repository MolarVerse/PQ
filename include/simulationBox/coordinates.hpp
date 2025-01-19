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

#ifndef __COORDINATES_HPP__
#define __COORDINATES_HPP__

#include "typeAliases.hpp"

namespace simulationBox
{
    class Coordinates
    {
       protected:
        std::vector<Real> _pos;
        std::vector<Real> _vel;
        std::vector<Real> _forces;
        std::vector<Real> _shiftForces;
        std::vector<Real> _oldPos;
        std::vector<Real> _oldVel;
        std::vector<Real> _oldForces;

        std::vector<Real> _comMolecules;

       public:
        virtual ~Coordinates() = default;

        virtual void resizeHostVectors(cul nAtoms, cul nMolecules);

        [[nodiscard]] virtual Real* getPosPtr();
        [[nodiscard]] virtual Real* getVelPtr();
        [[nodiscard]] virtual Real* getForcesPtr();
        [[nodiscard]] virtual Real* getShiftForcesPtr();
        [[nodiscard]] virtual Real* getOldPosPtr();
        [[nodiscard]] virtual Real* getOldVelPtr();
        [[nodiscard]] virtual Real* getOldForcesPtr();
        [[nodiscard]] virtual Real* getComMoleculesPtr();

        [[nodiscard]] std::vector<Real> getPos() const;
        [[nodiscard]] std::vector<Real> getVel() const;
        [[nodiscard]] std::vector<Real> getForces() const;
        [[nodiscard]] std::vector<Real> getShiftForces() const;
        [[nodiscard]] std::vector<Real> getOldPos() const;
        [[nodiscard]] std::vector<Real> getOldVel() const;
        [[nodiscard]] std::vector<Real> getOldForces() const;
        [[nodiscard]] std::vector<Real> getComMolecules() const;
    };
}   // namespace simulationBox

#endif   // __COORDINATES_HPP__