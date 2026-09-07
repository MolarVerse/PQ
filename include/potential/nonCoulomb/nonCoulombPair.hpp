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

#ifndef _NON_COULOMB_PAIR_HPP_

#define _NON_COULOMB_PAIR_HPP_

#include <utility>   // for pair

#include "strongTypes.hpp"

namespace pot
{
    /**
     * @class NonCoulombPair
     *
     * @brief base class representing a pair of non-coulombic types
     *
     * @details constructor with van der Waals types and cut-off radius is for
     * force field parameters constructor with cut-off radius only is for guff
     * representation
     *
     */
    class NonCoulombPair
    {
       protected:
        ExtVdwType _vanDerWaalsType1{0};
        ExtVdwType _vanDerWaalsType2{0};
        VdwType    _internalType1{0};
        VdwType    _internalType2{0};

        double _radialCutOff;
        double _energyCutOff = 0.0;
        double _forceCutOff  = 0.0;

       public:
        explicit NonCoulombPair(
            const ExtVdwType,
            const ExtVdwType,
            const double
        );
        explicit NonCoulombPair(const double);
        explicit NonCoulombPair(const double, const double, const double);

        virtual ~NonCoulombPair() = default;

        [[nodiscard]] bool operator==(const NonCoulombPair &other) const;

        [[nodiscard]] virtual std::pair<double, double> calculate(
            const double distance
        ) const = 0;

        /********************
         * standard setters *
         ********************/

        void setInternalType1(const VdwType internalType1);
        void setInternalType2(const VdwType internalType2);
        void setRadialCutOff(const double radialCutoff);
        void setEnergyCutOff(const double energyCutoff);
        void setForceCutOff(const double forceCutoff);

        /********************
         * standard getters *
         ********************/

        [[nodiscard]] ExtVdwType getVanDerWaalsType1() const;
        [[nodiscard]] ExtVdwType getVanDerWaalsType2() const;
        [[nodiscard]] VdwType    getInternalType1() const;
        [[nodiscard]] VdwType    getInternalType2() const;
        [[nodiscard]] double getRadialCutOff() const { return _radialCutOff; }
        [[nodiscard]] double getEnergyCutOff() const;
        [[nodiscard]] double getForceCutOff() const;
    };

}   // namespace pot

#endif   // _NON_COULOMB_PAIR_HPP_
