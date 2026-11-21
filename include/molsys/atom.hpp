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

#ifndef MOLSYS__ATOM_HPP
#define MOLSYS__ATOM_HPP

#include <optional>
#include <string>
#include <string_view>

#include "strongTypes.hpp"
#include "unit_types.hpp"
#include "vector3d.hpp"

namespace molsys
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

        ExtVdwType _externalGlobalVDWType;
        VdwType    _internalGlobalVDWType;

        ExtAtomType _externalAtomType;
        AtomType    _atomType;

        bool _isActive = true;

        AtomNumber            _atomicNumber;
        Mass                  _mass;
        Charge                _partialCharge;
        std::optional<Charge> _qmCharge;

        Pos3D _position;
        Pos3D _positionOld;

        Vel3D _velocity;
        Vel3D _velocityOld;

        Force3D      _force;
        Force3D      _forceOld;
        Force3D      _forceInner;
        Force3D      _forceOuter;
        ShiftForce3D _shiftForce;

       public:
        Atom() = default;

        void initMass();

        void updateOldPosition();
        void updateOldVelocity();
        void updateOldForce();

        /*******************
         * scaling methods *
         *******************/

        void scaleVelocity(double scaleFactor);
        void scaleVelocity(const linalg::Vec3D &scaleFactor);
        void scaleForce(double scaleFactor);
        void scaleForce(const linalg::Vec3D &scaleFactor);

        /**************************
         * standard adder methods *
         **************************/

        void addPosition(const Pos3D &position);
        void addVelocity(const Vel3D &velocity);
        void addForce(const Force3D &force) { _force += force; }
        void addForceInner(const Force3D &force);
        void addForceOuter(const Force3D &force);
        void addShiftForce(const ShiftForce3D &shiftForce)
        {
            _shiftForce += shiftForce;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] bool isActive() const;

        [[nodiscard]] const std::string &getName() const { return _name; }
        [[nodiscard]] std::string        getAtomTypeName() const;

        [[nodiscard]] ExtAtomType getExternalAtomType() const;
        [[nodiscard]] AtomType    getAtomType() const;

        [[nodiscard]] ExtVdwType getExternalGlobalVDWType() const;
        [[nodiscard]] VdwType    getInternalGlobalVDWType() const;

        [[nodiscard]]
        AtomNumber getAtomicNumber() const
        {
            return _atomicNumber;
        }

        [[nodiscard]] const Mass   &getMass() const;
        [[nodiscard]] const Charge &getPartialCharge() const
        {
            return _partialCharge;
        }
        [[nodiscard]] const std::optional<Charge> &getQMCharge() const;

        [[nodiscard]] const Pos3D   &getPosition() const { return _position; }
        [[nodiscard]] const Pos3D   &getPositionOld() const;
        [[nodiscard]] const Vel3D   &getVelocity() const;
        [[nodiscard]] const Force3D &getForce() const;
        [[nodiscard]] const Force3D &getForceOld() const;
        [[nodiscard]] const Force3D &getForceInner() const;
        [[nodiscard]] const Force3D &getForceOuter() const;
        [[nodiscard]] const ShiftForce3D &getShiftForce() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setActive(bool isActive);

        void setName(const std::string_view &name);
        void setAtomTypeName(const std::string_view &atomTypeName);
        void setAtomicNumber(AtomNumber atomicNumber);

        void setMass(const Mass &mass);
        void setPartialCharge(const Charge &partialCharge);
        void setQMCharge(const Charge &charge);

        void setAtomType(AtomType atomType);
        void setExternalAtomType(ExtAtomType externalAtomType);
        void setExternalGlobalVDWType(ExtVdwType externalGlobalVDWType);
        void setInternalGlobalVDWType(VdwType internalGlobalVDWType);

        void setPosition(const Pos3D &position);
        void setVelocity(const Vel3D &velocity);
        void setForce(const Force3D &force);
        void setForceInner(const Force3D &force);
        void setForceOuter(const Force3D &force);
        void setShiftForce(const ShiftForce3D &shiftForce);

        void setPositionOld(const Pos3D &positionOld);
        void setVelocityOld(const Vel3D &velocityOld);
        void setForceOld(const Force3D &forceOld);

        void setForceToZero();
        void setInnerForceToZero();
        void setOuterForceToZero();

        void resetQMCharge();
    };
}   // namespace molsys

#endif   // MOLSYS__ATOM_HPP
