#include "constraints.hpp"

#include "exceptions.hpp"

#include <algorithm>     // for ranges::for_each
#include <functional>    // for identity
#include <string_view>   // IWYU pragma: keep

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

    auto   converged = false;
    size_t iter      = 0;

    while (!converged && iter <= _shakeMaxIter)
    {
        converged = true;

        std::ranges::for_each(_bondConstraints,
                              [&simulationBox, &converged, this](auto &bondConstraint)
                              { converged = converged && bondConstraint.applyShake(simulationBox, _shakeTolerance, _dt); });

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException("Shake algorithm did not converge.");
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

    auto   converged = false;
    size_t iter      = 0;

    while (!converged && iter <= _rattleMaxIter)
    {
        converged = true;

        std::ranges::for_each(_bondConstraints,
                              [&converged, this](auto &bondConstraint)
                              { converged = converged && bondConstraint.applyRattle(_rattleTolerance); });

        ++iter;
    }

    if (!converged)
        throw customException::ShakeException("Rattle algorithm did not converge.");
}