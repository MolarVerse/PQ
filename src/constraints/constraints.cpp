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

#include "constraints.hpp"

#include <algorithm>    // for ranges::for_each
#include <format>       // for format
#include <functional>   // for identity
#include <string>       // for string
#include <vector>       // for vector

#include "exceptions.hpp"

using namespace constraints;

/**
 * @brief calculates the reference bond data of all bond constraints
 *
 * @param simulationBox
 *
 */
void Constraints::calculateConstraintBondRefs(
    const simulationBox::SimulationBox &simulationBox
)
{
    startTimingsSection("Reference Bond Data");

    std::ranges::for_each(
        _bondConstraints,
        [&simulationBox](auto &bondConstraint)
        { bondConstraint.calculateConstraintBondRef(simulationBox); }
    );

    stopTimingsSection("Reference Bond Data");
}

/**
 * @brief applies the shake algorithm to all bond constraints
 *
 * @throws customException::ShakeException if shake algorithm does not converge
 */
void Constraints::applyShake(const simulationBox::SimulationBox &simulationBox)
{
    if (!_activated)
        return;

    startTimingsSection("Shake");

    std::vector<bool> convergedVector;
    bool              converged = false;

    size_t iter = 0;

    while (!converged && iter <= _shakeMaxIter)
    {
        convergedVector.clear();

        auto applyShakeForSingleBond =
            [&simulationBox, &convergedVector, this](auto &bondConstraint)
        {
            convergedVector.push_back(
                bondConstraint.applyShake(simulationBox, _shakeTolerance)
            );
        };

        std::ranges::for_each(_bondConstraints, applyShakeForSingleBond);

        converged = std::ranges::all_of(
            convergedVector,
            [](const bool isConverged) { return isConverged; }
        );

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException(std::format(
            "Shake algorithm did not converge for {} bonds.",
            std::ranges::count(convergedVector, false)
        ));

    stopTimingsSection("Shake");
}

/**
 * @brief applies the rattle algorithm to all bond constraints
 *
 * @throws customException::ShakeException if rattle algorithm does not converge
 */
void Constraints::applyRattle()
{
    if (!_activated)
        return;

    startTimingsSection("Rattle");

    std::vector<bool> convergedVector;
    bool              converged = false;

    size_t iter = 0;

    while (!converged && iter <= _rattleMaxIter)
    {
        convergedVector.clear();

        auto applyRattleForSingleBond = [&convergedVector,
                                         this](auto &bondConstraint) {
            convergedVector.push_back(
                bondConstraint.applyRattle(_rattleTolerance)
            );
        };

        std::ranges::for_each(_bondConstraints, applyRattleForSingleBond);

        converged = std::ranges::all_of(
            convergedVector,
            [](const bool isConverged) { return isConverged; }
        );

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException(std::format(
            "Rattle algorithm did not converge for {} bonds.",
            std::ranges::count(convergedVector, false)
        ));

    stopTimingsSection("Rattle");
}