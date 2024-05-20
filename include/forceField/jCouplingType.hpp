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

#ifndef _J_COUPLING_TYPE_HPP_

#define _J_COUPLING_TYPE_HPP_

#include <cstddef>

namespace forceField
{

    /**
     * @class JCouplingType
     *
     * @brief represents a j coupling type
     *
     * @details this is a class representing a j coupling type defined in the
     * parameter file
     *
     */
    class JCouplingType
    {
       private:
        size_t _id;
        bool   _upperSymmetry = true;
        bool   _lowerSymmetry = true;

        double _J0;
        double _forceConstant;
        double _a;
        double _b;
        double _c;
        double _phaseShift;

       public:
        JCouplingType(
            const size_t id,
            const double J0,
            const double forceConstant,
            const double a,
            const double b,
            const double c,
            const double phaseShift
        )
            : _id(id),
              _J0(J0),
              _forceConstant(forceConstant),
              _a(a),
              _b(b),
              _c(c),
              _phaseShift(phaseShift)
        {
        }

        [[nodiscard]] bool operator==(const JCouplingType &other) const;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getId() const { return _id; }
        [[nodiscard]] double getJ0() const { return _J0; }
        [[nodiscard]] double getForceConstant() const { return _forceConstant; }
        [[nodiscard]] double getA() const { return _a; }
        [[nodiscard]] double getB() const { return _b; }
        [[nodiscard]] double getC() const { return _c; }
        [[nodiscard]] double getPhaseShift() const { return _phaseShift; }

        /***************************
         * standard setter methods *
         ***************************/

        void setUpperSymmetry(const bool boolean) { _upperSymmetry = boolean; }
        void setLowerSymmetry(const bool boolean) { _lowerSymmetry = boolean; }
    };

}   // namespace forceField

#endif   // _J_COUPLING_TYPE_HPP_