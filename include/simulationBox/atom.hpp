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

#include <cstddef>       // for size_t
#include <string>        // for string
#include <string_view>   // for string_view

#include "typeAliases.hpp"

namespace simulationBox
{
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

        bool _isQMOnly = false;
        bool _isMMOnly = false;

        int    _atomicNumber;
        double _mass;
        double _partialCharge;

        pq::Vec3D _position;
        pq::Vec3D _positionOld;

        pq::Vec3D _velocity;
        pq::Vec3D _velocityOld;

        pq::Vec3D _force;
        pq::Vec3D _forceOld;
        pq::Vec3D _shiftForce;

       public:
        Atom() = default;

        void initMass();

        void updateOldPosition();
        void updateOldVelocity();
        void updateOldForce();

        /*******************
         * scaling methods *
         *******************/

        void scaleVelocity(const double scaleFactor);
        void scaleVelocity(const pq::Vec3D &scaleFactor);
        void scaleVelocityOrthogonalSpace(const pq::tensor3D &, const Box &);

        /**************************
         * standard adder methods *
         **************************/

        void addPosition(const pq::Vec3D &position);
        void addVelocity(const pq::Vec3D &velocity);
        void addForce(const pq::Vec3D &force);
        void addForce(const double, const double, const double);
        void addShiftForce(const pq::Vec3D &shiftForce);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] bool isQMOnly() const;
        [[nodiscard]] bool isMMOnly() const;

        [[nodiscard]] std::string getName() const;
        [[nodiscard]] std::string getAtomTypeName() const;

        [[nodiscard]] size_t getExternalAtomType() const;
        [[nodiscard]] size_t getAtomType() const;

        [[nodiscard]] size_t getExternalGlobalVDWType() const;
        [[nodiscard]] size_t getInternalGlobalVDWType() const;

        [[nodiscard]] int    getAtomicNumber() const;
        [[nodiscard]] double getMass() const;
        [[nodiscard]] double getPartialCharge() const;

        [[nodiscard]] pq::Vec3D getPosition() const;
        [[nodiscard]] pq::Vec3D getPositionOld() const;
        [[nodiscard]] pq::Vec3D getVelocity() const;
        [[nodiscard]] pq::Vec3D getForce() const;
        [[nodiscard]] pq::Vec3D getForceOld() const;
        [[nodiscard]] pq::Vec3D getShiftForce() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setQMOnly(const bool isQMOnly);
        void setMMOnly(const bool isMMOnly);

        void setName(const std::string_view &name);
        void setAtomTypeName(const std::string_view &atomTypeName);
        void setAtomicNumber(const int atomicNumber);

        void setMass(const double mass);
        void setPartialCharge(const double partialCharge);

        void setAtomType(const size_t atomType);
        void setExternalAtomType(const size_t externalAtomType);
        void setExternalGlobalVDWType(const size_t externalGlobalVDWType);
        void setInternalGlobalVDWType(const size_t internalGlobalVDWType);

        void setPosition(const pq::Vec3D &position);
        void setVelocity(const pq::Vec3D &velocity);
        void setForce(const pq::Vec3D &force);
        void setShiftForce(const pq::Vec3D &shiftForce);

        void setPositionOld(const pq::Vec3D &positionOld);
        void setVelocityOld(const pq::Vec3D &velocityOld);
        void setForceOld(const pq::Vec3D &forceOld);

        void setForceToZero();
    };
}   // namespace simulationBox

#endif   // _ATOM_HPP_