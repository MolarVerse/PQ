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

#include "nonCoulombPair.hpp"

#include "mathUtilities.hpp"

namespace pot
{

    /**
     * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
     *
     * @param vanDerWaalsType1
     * @param vanDerWaalsType2
     * @param cutOff
     */
    NonCoulombPair::NonCoulombPair(
        ExtVdwType vanDerWaalsType1,
        ExtVdwType vanDerWaalsType2,
        double     cutOff
    )
        : _vanDerWaalsTypes{vanDerWaalsType1, vanDerWaalsType2},
          _radialCutOff(cutOff)
    {
    }

    /**
     * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
     *
     * @param cutOff
     */
    NonCoulombPair::NonCoulombPair(double cutOff) : _radialCutOff(cutOff) {}

    /**
     * @brief Construct a new Non Coulomb Pair:: Non Coulomb Pair object
     *
     * @param cutoff
     * @param energyCutoff
     * @param forceCutoff
     */
    NonCoulombPair::NonCoulombPair(
        double cutoff,
        double energyCutoff,
        double forceCutoff
    )
        : _radialCutOff(cutoff),
          _energyCutOff(energyCutoff),
          _forceCutOff(forceCutoff)
    {
    }

    /**
     * @brief operator overload for the comparison of two NonCoulombPair objects
     *
     * @details returns also true if the two types are switched
     *
     * @note uses only the van der Waals types and the radial cut off
     *
     * @param other
     * @return true
     * @return false
     */
    bool NonCoulombPair::operator==(const NonCoulombPair &other) const
    {
        auto isEq = true;
        isEq = isEq && _vanDerWaalsTypes.first == other._vanDerWaalsTypes.first;
        isEq =
            isEq && _vanDerWaalsTypes.second == other._vanDerWaalsTypes.second;
        isEq = isEq && utilities::compare(_radialCutOff, other._radialCutOff);

        auto isEqSymm = true;
        isEqSymm      = isEqSymm &&
                   _vanDerWaalsTypes.first == other._vanDerWaalsTypes.second;
        isEqSymm = isEqSymm &&
                   _vanDerWaalsTypes.second == other._vanDerWaalsTypes.first;
        isEqSymm =
            isEqSymm && utilities::compare(_radialCutOff, other._radialCutOff);

        return isEq || isEqSymm;
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief set internal type 1
     *
     * @param internalType1
     */
    void NonCoulombPair::setInternalType1(VdwType internalType1)
    {
        _internalTypes.first = internalType1;
    }

    /**
     * @brief set internal type 2
     *
     * @param internalType2
     */
    void NonCoulombPair::setInternalType2(VdwType internalType2)
    {
        _internalTypes.second = internalType2;
    }

    /**
     * @brief set radial cut off
     *
     * @param radialCutoff
     */
    void NonCoulombPair::setRadialCutOff(double radialCutoff)
    {
        _radialCutOff = radialCutoff;
    }

    /**
     * @brief set energy cut off
     *
     * @param energyCutoff
     */
    void NonCoulombPair::setEnergyCutOff(double energyCutoff)
    {
        _energyCutOff = energyCutoff;
    }

    /**
     * @brief set force cut off
     *
     * @param forceCutoff
     */
    void NonCoulombPair::setForceCutOff(double forceCutoff)
    {
        _forceCutOff = forceCutoff;
    }

    /***************************
     *                         *
     * standard setter methods *
     *                         *
     ***************************/

    /**
     * @brief get van der Waals type 1
     *
     * @return ExtVdwType
     */
    ExtVdwType NonCoulombPair::getVanDerWaalsType1() const
    {
        return _vanDerWaalsTypes.first;
    }

    /**
     * @brief get van der Waals type 2
     *
     * @return ExtVdwType
     */
    ExtVdwType NonCoulombPair::getVanDerWaalsType2() const
    {
        return _vanDerWaalsTypes.second;
    }

    /**
     * @brief get internal type 1
     *
     * @return VdwType
     */
    VdwType NonCoulombPair::getInternalType1() const
    {
        return _internalTypes.first;
    }

    /**
     * @brief get internal type 2
     *
     * @return VdwType
     */
    VdwType NonCoulombPair::getInternalType2() const
    {
        return _internalTypes.second;
    }

    /**
     * @brief get energy cut off
     *
     * @return double
     */
    double NonCoulombPair::getEnergyCutOff() const { return _energyCutOff; }

    /**
     * @brief get force cut off
     *
     * @return double
     */
    double NonCoulombPair::getForceCutOff() const { return _forceCutOff; }

}   // namespace pot
