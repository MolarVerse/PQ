#include "constraints.hpp"

#include "exceptions.hpp"

#include <algorithm>     // for ranges::for_each
#include <format>        // for format
#include <functional>    // for identity
#include <string_view>   // for string_view
#include <vector>        // for vector

using namespace constraints;

/**
 * @brief calculates the reference bond data of all bond constraints
 *
 * @param simulationBox
 *
 */
void Constraints::calculateConstraintBondRefs(const simulationBox::SimulationBox &simulationBox)
{
    std::ranges::for_each(_bondConstraints,
                          [&simulationBox](auto &bondConstraint) { bondConstraint.calculateConstraintBondRef(simulationBox); });
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

    std::vector<bool> convergedVector;
    bool              converged = false;

    size_t iter = 0;

    while (!converged && iter <= _shakeMaxIter)
    {
        convergedVector.clear();

        auto applyShakeForSingleBond = [&simulationBox, &convergedVector, this](auto &bondConstraint)
        { convergedVector.push_back(bondConstraint.applyShake(simulationBox, _shakeTolerance)); };

        std::ranges::for_each(_bondConstraints, applyShakeForSingleBond);

        converged = std::ranges::all_of(convergedVector, [](bool isConverged) { return isConverged; });

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException(
            std::format("Shake algorithm did not converge for {} bonds.", std::ranges::count(convergedVector, false)));
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

    std::vector<bool> convergedVector;
    bool              converged = false;

    size_t iter = 0;

    while (!converged && iter <= _rattleMaxIter)
    {
        convergedVector.clear();

        auto applyRattleForSingleBond = [&convergedVector, this](auto &bondConstraint)
        { convergedVector.push_back(bondConstraint.applyRattle(_rattleTolerance)); };

        std::ranges::for_each(_bondConstraints, applyRattleForSingleBond);

        converged = std::ranges::all_of(convergedVector, [](bool isConverged) { return isConverged; });

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException(
            std::format("Rattle algorithm did not converge for {} bonds.", std::ranges::count(convergedVector, false)));
}