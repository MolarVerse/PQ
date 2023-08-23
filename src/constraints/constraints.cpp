#include "constraints.hpp"

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
 * @TODO: implement check if not converged with own exception type
 */
void Constraints::applyShake(const simulationBox::SimulationBox &simulationBox)
{
    auto   converged = false;
    size_t iter      = 0;

    while (!converged && iter <= _shakeMaxIter)
    {
        converged = true;

        std::cout << "applyShake: " << iter << "\n";

        std::ranges::for_each(_bondConstraints,
                              [&simulationBox, &converged, this](auto &bondConstraint)
                              { converged = converged && bondConstraint.applyShake(simulationBox, _shakeTolerance, _dt); });

        ++iter;
    }
}

/**
 * @brief applies the rattle algorithm to all bond constraints
 *
 * @TODO: implement check if not converged with own exception type
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