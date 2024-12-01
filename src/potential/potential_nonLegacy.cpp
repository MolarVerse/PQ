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

#include <cmath>   // for sqrt

#include "box.hpp"                   // for Box
#include "coulombPotential.hpp"      // for CoulombPotential
#include "molecule.hpp"              // for Molecule
#include "nonCoulombPair.hpp"        // for NonCoulombPair
#include "nonCoulombPotential.hpp"   // for NonCoulombPotential
#include "potential.hpp"

using namespace potential;
using namespace simulationBox;
using namespace linearAlgebra;

/**
 * @brief inner part of the double loop to calculate non-bonded inter molecular
 * interactions
 *
 * @param box
 * @param molecule1
 * @param molecule2
 * @param atom1
 * @param atom2
 * @return std::pair<double, double>
 */
std::pair<Real, Real> Potential::calculateSingleInteraction(
    const Box&   box,
    const Real   xi,
    const Real   yi,
    const Real   zi,
    const Real   xj,
    const Real   yj,
    const Real   zj,
    const size_t atomType_i,
    const size_t atomType_j,
    const size_t globalVdwType_i,
    const size_t globalVdwType_j,
    const size_t moltype_i,
    const size_t moltype_j,
    const Real   charge_i,
    const Real   charge_j,
    Real&        fx,
    Real&        fy,
    Real&        fz,
    Real&        shiftfx,
    Real&        shiftfy,
    Real&        shiftfz
) const
{
    auto coulombEnergy    = 0.0;
    auto nonCoulombEnergy = 0.0;

    const auto xyz_i = Vec3D{xi, yi, zi};
    const auto xyz_j = Vec3D{xj, yj, zj};

    auto dxyz = xyz_i - xyz_j;

    const auto txyz = -box.calcShiftVector(dxyz);

    dxyz += txyz;

    const double distanceSquared = normSquared(dxyz);

    if (const auto RcCutOff = CoulombPotential::getCoulombRadiusCutOff();
        distanceSquared < RcCutOff * RcCutOff)
    {
        const double distance = ::sqrt(distanceSquared);

        const auto combinedIdx = {
            moltype_i,
            moltype_j,
            atomType_i,
            atomType_j,
            globalVdwType_i,
            globalVdwType_j
        };

        const auto coulombPreFactor = charge_i * charge_j;

        auto [e, f] = _coulombPotential->calculate(distance, coulombPreFactor);
        coulombEnergy = e;

        const auto nonCoulPair = _nonCoulombPot->getNonCoulPair(combinedIdx);
        const auto rncCutOff   = nonCoulPair->getRadialCutOff();

        if (distance < rncCutOff)
        {
            const auto& [nonCoulE, nonCoulF] = nonCoulPair->calculate(distance);
            nonCoulombEnergy                 = nonCoulE;

            f += nonCoulF;
        }

        f /= distance;

        const auto forcexyz = f * dxyz;

        fx = forcexyz[0];
        fy = forcexyz[1];
        fz = forcexyz[2];

        const auto shiftForcexyz = forcexyz * txyz;

        shiftfx = shiftForcexyz[0];
        shiftfy = shiftForcexyz[1];
        shiftfz = shiftForcexyz[2];
    }

    return {coulombEnergy, nonCoulombEnergy};
}