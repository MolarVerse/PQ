#include "constraints.hpp"

using namespace constraints;

/**
 * @brief calculates the reference bond data of all bond constraints
 *
 * @param simulationBox
 *
 */
void Constraints::calculateConstraintBondRefs(const simulationBox::SimulationBox &simBox)
{
    std::ranges::for_each(_bondConstraints,
                          [&simBox](auto &bondConstraint) { bondConstraint.calculateConstraintBondRef(simBox); });
}

/**
 * @brief applies the shake algorithm to all bond constraints
 *
 */
void Constraints::applyShake(const simulationBox::SimulationBox &simBox)
{
    auto   converged = false;
    size_t iter      = 0;

    while (!converged && iter <= _shakeMaxIter)
    {
        converged = true;

        std::ranges::for_each(_bondConstraints,
                              [&simBox, &converged, this](auto &bondConstraint)
                              { converged = converged && bondConstraint.applyShake(simBox, _shakeTolerance, _dt); });

        ++iter;
    }
}

/**
 * @brief applies the rattle algorithm to all bond constraints
 *
 */
void Constraints::applyRattle()
{
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
}