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

// Fixed-work micro-benchmark of the SHAKE/RATTLE constraint solver. One bond
// constraint per molecule, started at rest (positionOld == position) so the
// solver is stable across iterations.

#include <cstdint>
#include <cstdio>
#include <format>
#include <iostream>

#include "strongTypes.hpp"

#ifdef PQ_WITH_CALLGRIND
#include <valgrind/callgrind.h>
#else
#define CALLGRIND_ZERO_STATS
#endif

#include "bondConstraint.hpp"
#include "constraints.hpp"
#include "perfBenchSetup.hpp"
#include "simulationBox.hpp"
#include "timingsSettings.hpp"

static constexpr std::uint64_t ITERATIONS = 1000;

int main()
{
    settings::TimingsSettings::setTimeStep(0.001);

    auto box =
        benchSetup::makePopulatedBox({.nMolecules = 20, .nAtomsPerMol = 3});
    auto &molecules = box.getMolecules();

    auto constr = constraints::Constraints();
    constr.setShakeMaxIter(100);
    constr.setRattleMaxIter(100);
    constr.setShakeTolerance(1.0e-8);
    constr.setRattleTolerance(1.0e-8);
    constr.activateShake();

    for (auto &molecule : molecules)
    {
        constr.addBondConstraint(
            constraints::BondConstraint(
                &molecule,
                &molecule,
                AtomIndex{0},
                AtomIndex{1},
                0.85
            )
        );
    }

    constr.calculateConstraintBondRefs(box);

    CALLGRIND_ZERO_STATS;

    for (std::uint64_t i = 0; i < ITERATIONS; ++i)
    {
        constr.applyShake(box);
        constr.applyRattle(box);
    }

    // read state so the loop cannot be optimized away
    std::cout << std::format("{:.6f}\n", box.calculateMomentum()[0]);
    return 0;
}
