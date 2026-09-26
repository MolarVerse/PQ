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

#ifndef _ATOM_HPP_

#define _ATOM_HPP_

#include <optional>      // for optional
#include <string>        // for string
#include <string_view>   // for string_view

#include "staticMatrix.hpp"
#include "strongTypes.hpp"
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
        double                _mass;
        double                _partialCharge;
        std::optional<double> _qmCharge;

        linalg::Vec3D _position;
        linalg::Vec3D _positionOld;

        linalg::Vec3D _velocity;
        linalg::Vec3D _velocityOld;

        linalg::Vec3D _force;
        linalg::Vec3D _forceOld;
        linalg::Vec3D _forceInner;
        linalg::Vec3D _forceOuter;
        linalg::Vec3D _shiftForce;

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
        void scaleVelocityOrthogonalSpace(
            const linalg::tensor3D &,
            const Box &
        );
        void scaleForce(double scaleFactor);
        void scaleForce(const linalg::Vec3D &scaleFactor);

        /**************************
         * standard adder methods *
         **************************/

        void addPosition(const linalg::Vec3D &position);
        void addVelocity(const linalg::Vec3D &velocity);
        void addForce(const linalg::Vec3D &force) { _force += force; }
        void addForce(double, double, double);
        void addForceInner(const linalg::Vec3D &force);
        void addForceOuter(const linalg::Vec3D &force);
        void addShiftForce(const linalg::Vec3D &shiftForce)
        {
            _shiftForce += shiftForce;
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] bool isActive() const;
        [[nodiscard]] bool isQMAtom() const;
        [[nodiscard]] bool isMMAtom() const;

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

        [[nodiscard]] double getMass() const;
        [[nodiscard]] double getPartialCharge() const { return _partialCharge; }
        [[nodiscard]] std::optional<double> getQMCharge() const;

        [[nodiscard]] const linalg::Vec3D &getPosition() const
        {
            return _position;
        }
        [[nodiscard]] linalg::Vec3D getPositionOld() const;
        [[nodiscard]] linalg::Vec3D getVelocity() const;
        [[nodiscard]] linalg::Vec3D getForce() const;
        [[nodiscard]] linalg::Vec3D getForceOld() const;
        [[nodiscard]] linalg::Vec3D getForceInner() const;
        [[nodiscard]] linalg::Vec3D getForceOuter() const;
        [[nodiscard]] linalg::Vec3D getShiftForce() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setActive(bool isActive);

        void setName(const std::string_view &name);
        void setAtomTypeName(const std::string_view &atomTypeName);
        void setAtomicNumber(AtomNumber atomicNumber);

        void setMass(double mass);
        void setPartialCharge(double partialCharge);
        void setQMCharge(double charge);

        void setAtomType(AtomType atomType);
        void setExternalAtomType(ExtAtomType externalAtomType);
        void setExternalGlobalVDWType(ExtVdwType externalGlobalVDWType);
        void setInternalGlobalVDWType(VdwType internalGlobalVDWType);

        void setPosition(const linalg::Vec3D &position);
        void setVelocity(const linalg::Vec3D &velocity);
        void setForce(const linalg::Vec3D &force);
        void setForceInner(const linalg::Vec3D &force);
        void setForceOuter(const linalg::Vec3D &force);
        void setShiftForce(const linalg::Vec3D &shiftForce);

        void setPositionOld(const linalg::Vec3D &positionOld);
        void setVelocityOld(const linalg::Vec3D &velocityOld);
        void setForceOld(const linalg::Vec3D &forceOld);

        void setForceToZero();
        void setInnerForceToZero();
        void setOuterForceToZero();

        void resetQMCharge();
    };
}   // namespace molsys

#endif   // _ATOM_HPP_
