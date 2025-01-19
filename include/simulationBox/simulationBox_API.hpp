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

#ifndef __SIMULATION_BOX_API_HPP__
#define __SIMULATION_BOX_API_HPP__

#include "debug.hpp"   // IWYU pragma: keep
#include "typeAliases.hpp"

namespace simulationBox
{
    std::tuple<Real, Real, Real, Real> posMinMaxSumMean(pq::SimBox& simBox);
    std::tuple<Real, Real, Real, Real> velMinMaxSumMean(pq::SimBox& simBox);
    std::tuple<Real, Real, Real, Real> forcesMinMaxSumMean(pq::SimBox& simBox);

    Real      calculateTemperature(pq::SimBox& simBox);
    pq::Vec3D calculateMomentum(pq::SimBox& simBox);
    pq::Vec3D calculateAngularMomentum(pq::SimBox& simBox, const pq::Vec3D&);

    Real      calculateTotalMass(pq::SimBox& simBox, const bool updateSimBox);
    Real      calculateTotalMass(pq::SimBox& simBox);
    Real      calculateTotalCharge(pq::SimBox& simBox, const bool updateSimBox);
    Real      calculateTotalCharge(pq::SimBox& simBox);
    pq::Vec3D calculateCenterOfMass(pq::SimBox& simBox, const bool update);
    pq::Vec3D calculateCenterOfMass(pq::SimBox& simBox);
    void      calculateCenterOfMassMolecules(pq::SimBox& simBox);
    void      calculateMolMasses(pq::SimBox& simBox);

}   // namespace simulationBox

#ifdef __PQ_DEBUG__

    #define __POS_MIN_MAX_SUM_MEAN__(simBox)    \
        config::Debug::debugMinMaxSumMean(      \
            posMinMaxSumMean(simBox),           \
            "Pos:",                             \
            config::DebugLevel::POSITION_DEBUG, \
            "Angstrom"                          \
        );

    #define __VEL_MIN_MAX_SUM_MEAN__(simBox)    \
        config::Debug::debugMinMaxSumMean(      \
            velMinMaxSumMean(simBox),           \
            "Vel:",                             \
            config::DebugLevel::VELOCITY_DEBUG, \
            "Angstrom/s"                        \
        );

    #define __FORCE_MIN_MAX_SUM_MEAN__(simBox) \
        config::Debug::debugMinMaxSumMean(     \
            forcesMinMaxSumMean(simBox),       \
            "Frc:",                            \
            config::DebugLevel::FORCE_DEBUG,   \
            "kcal/mol/Angstrom"                \
        );

#else

    #define __POS_MIN_MAX_SUM_MEAN__(simBox)     // do nothing
    #define __VEL_MIN_MAX_SUM_MEAN__(simBox)     // do nothing
    #define __FORCE_MIN_MAX_SUM_MEAN__(simBox)   // do nothing

#endif

#endif   // __SIMULATION_BOX_API_HPP__
