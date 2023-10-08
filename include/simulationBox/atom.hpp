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

#ifndef _ATOM_HPP_

#define _ATOM_HPP_

#include "vector3d.hpp"   // for Vec3D

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

namespace simulationBox
{
    class Box;   // forward declaration

    /**
     * @class Atom
     *
     * @brief containing all information about an atom
     */
    class Atom
    {
      private:
        std::string _name;
        std::string _atomTypeName;

        size_t _externalGlobalVDWType;
        size_t _internalGlobalVDWType;

        size_t _externalAtomType;
        size_t _atomType;

        int    _atomicNumber;
        double _mass;
        double _partialCharge;

        linearAlgebra::Vec3D _position;
        linearAlgebra::Vec3D _velocity;
        linearAlgebra::Vec3D _force;
        linearAlgebra::Vec3D _shiftForce;

      public:
        Atom() = default;

        void addPosition(const linearAlgebra::Vec3D &position) { _position += position; }
        void addVelocity(const linearAlgebra::Vec3D &velocity) { _velocity += velocity; }
        void addForce(const linearAlgebra::Vec3D &force) { _force += force; }
        void addShiftForce(const linearAlgebra::Vec3D &shiftForce) { _shiftForce += shiftForce; }

        void scaleVelocity(const double scaleFactor) { _velocity *= scaleFactor; }
        void scaleVelocity(const linearAlgebra::Vec3D &scaleFactor) { _velocity *= scaleFactor; }
        void scaleVelocityOrthogonalSpace(const linearAlgebra::Vec3D &scaleFactor, const Box &box);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] std::string getName() const { return _name; }
        [[nodiscard]] std::string getAtomTypeName() const { return _atomTypeName; }

        [[nodiscard]] size_t getExternalAtomType() const { return _externalAtomType; }
        [[nodiscard]] size_t getAtomType() const { return _atomType; }

        [[nodiscard]] size_t getExternalGlobalVDWType() const { return _externalGlobalVDWType; }
        [[nodiscard]] size_t getInternalGlobalVDWType() const { return _internalGlobalVDWType; }

        [[nodiscard]] int    getAtomicNumber() const { return _atomicNumber; }
        [[nodiscard]] double getMass() const { return _mass; }
        [[nodiscard]] double getPartialCharge() const { return _partialCharge; }

        [[nodiscard]] linearAlgebra::Vec3D getPosition() const { return _position; }
        [[nodiscard]] linearAlgebra::Vec3D getVelocity() const { return _velocity; }
        [[nodiscard]] linearAlgebra::Vec3D getForce() const { return _force; }
        [[nodiscard]] linearAlgebra::Vec3D getShiftForce() const { return _shiftForce; }

        /***************************
         * standard setter methods *
         ***************************/

        void setName(const std::string_view &name) { _name = name; }
        void setAtomTypeName(const std::string_view &atomTypeName) { _atomTypeName = atomTypeName; }
        void setAtomicNumber(const int atomicNumber) { _atomicNumber = atomicNumber; }

        void setMass(const double mass) { _mass = mass; }
        void setPartialCharge(const double partialCharge) { _partialCharge = partialCharge; }

        void setAtomType(const size_t atomType) { _atomType = atomType; }
        void setExternalAtomType(const size_t externalAtomType) { _externalAtomType = externalAtomType; }
        void setExternalGlobalVDWType(const size_t externalGlobalVDWType) { _externalGlobalVDWType = externalGlobalVDWType; }
        void setInternalGlobalVDWType(const size_t internalGlobalVDWType) { _internalGlobalVDWType = internalGlobalVDWType; }

        void setPosition(const linearAlgebra::Vec3D &position) { _position = position; }
        void setVelocity(const linearAlgebra::Vec3D &velocity) { _velocity = velocity; }
        void setForce(const linearAlgebra::Vec3D &force) { _force = force; }
        void setShiftForce(const linearAlgebra::Vec3D &shiftForce) { _shiftForce = shiftForce; }

        void setForceToZero() { _force = {0.0, 0.0, 0.0}; }
    };
}   // namespace simulationBox

#endif   // _ATOM_HPP_