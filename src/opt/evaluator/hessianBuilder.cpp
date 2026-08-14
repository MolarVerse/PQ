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

#include "hessianBuilder.hpp"

#include <memory>

#include "atom.hpp"
#include "evaluator.hpp"
#include "exceptions.hpp"

using namespace opt;
using namespace settings;
using namespace customException;

/**
 * @brief Construct a new Force Difference Hessian Builder:: Force Difference
 * Hessian Builder object
 *
 * @param displacement
 */
ForceDifferenceHessianBuilder::ForceDifferenceHessianBuilder(
    const double displacement
)
    : _displacement(displacement)
{
}

/**
 * @brief Construct a new Central Force Difference Hessian Builder:: Central
 * Force Difference Hessian Builder object
 *
 * @param displacement
 */
std::vector<double> ForceDifferenceHessianBuilder::evaluateForces(
    Evaluator                    &evaluator,
    simulationBox::SimulationBox &simulationBox,
    const size_t                  coordinateIndex,
    const double                  displacement
) const
{
    displaceCoordinate(simulationBox, coordinateIndex, displacement);
    evaluator.evaluate();
    const auto forces = flattenForces(simulationBox);
    displaceCoordinate(simulationBox, coordinateIndex, -displacement);

    return forces;
}

/**
 * @brief restore the original positions of the atoms in the simulation box
 *
 * @param simulationBox
 * @param positions
 */
void ForceDifferenceHessianBuilder::restorePositions(
    simulationBox::SimulationBox            &simulationBox,
    const std::vector<linearAlgebra::Vec3D> &positions
)
{
    for (size_t atomIndex = 0; atomIndex < positions.size(); ++atomIndex)
        simulationBox.getAtom(atomIndex).setPosition(positions[atomIndex]);
}

/**
 * @brief displace a specific coordinate of the simulation box
 *
 * @param simulationBox
 * @param coordinateIndex
 * @param displacement
 */
void ForceDifferenceHessianBuilder::displaceCoordinate(
    simulationBox::SimulationBox &simulationBox,
    const size_t                  coordinateIndex,
    const double                  displacement
)
{
    const auto atomIndex = coordinateIndex / 3;
    const auto dimension = coordinateIndex % 3;

    auto position        = simulationBox.getAtom(atomIndex).getPosition();
    position[dimension] += displacement;

    simulationBox.getAtom(atomIndex).setPosition(position);
}

/**
 * @brief flatten the forces of the simulation box into a 1D vector
 *
 * @param simulationBox
 * @return std::vector<double>
 */
std::vector<double> ForceDifferenceHessianBuilder::flattenForces(
    const simulationBox::SimulationBox &simulationBox
)
{
    std::vector<double> flattenedForces;
    flattenedForces.reserve(3 * simulationBox.getNumberOfAtoms());

    for (const auto &force : simulationBox.getForces())
    {
        flattenedForces.push_back(force[0]);
        flattenedForces.push_back(force[1]);
        flattenedForces.push_back(force[2]);
    }

    return flattenedForces;
}

/**
 * @brief symmetrize the Hessian matrix
 *
 * @param hessian
 */
void ForceDifferenceHessianBuilder::symmetrize(HessianMatrix &hessian)
{
    for (size_t row = 0; row < hessian.size(); ++row)
    {
        for (size_t col = row + 1; col < hessian.size(); ++col)
        {
            const auto value  = 0.5 * (hessian[row][col] + hessian[col][row]);
            hessian[row][col] = value;
            hessian[col][row] = value;
        }
    }
}

/**
 * @brief Construct a new Central Force Difference Hessian Builder:: Central
 * Force Difference Hessian Builder object
 *
 * @param displacement
 */
HessianMatrix CentralForceDifferenceHessianBuilder::build(
    Evaluator                    &evaluator,
    simulationBox::SimulationBox &simulationBox
) const
{
    const auto numberOfCoordinates = 3 * simulationBox.getNumberOfAtoms();
    auto       hessian             = HessianMatrix(
        numberOfCoordinates,
        std::vector<double>(numberOfCoordinates, 0.0)
    );

    const auto originalPositions = simulationBox.getPositions();

    for (size_t col = 0; col < numberOfCoordinates; ++col)
    {
        const auto fPlus =
            evaluateForces(evaluator, simulationBox, col, _displacement);
        const auto fMinus =
            evaluateForces(evaluator, simulationBox, col, -_displacement);

        for (size_t row = 0; row < numberOfCoordinates; ++row)
            hessian[row][col] =
                -(fPlus[row] - fMinus[row]) / (2.0 * _displacement);

        restorePositions(simulationBox, originalPositions);
    }

    evaluator.evaluate();
    symmetrize(hessian);

    return hessian;
}

/**
 * @brief Construct a new Forward Force Difference Hessian Builder:: Forward
 * Force Difference Hessian Builder object
 *
 * @param displacement
 */
HessianMatrix ForwardForceDifferenceHessianBuilder::build(
    Evaluator                    &evaluator,
    simulationBox::SimulationBox &simulationBox
) const
{
    const auto numberOfCoordinates = 3 * simulationBox.getNumberOfAtoms();
    auto       hessian             = HessianMatrix(
        numberOfCoordinates,
        std::vector<double>(numberOfCoordinates, 0.0)
    );

    const auto originalPositions = simulationBox.getPositions();
    evaluator.evaluate();
    const auto f0 = flattenForces(simulationBox);

    for (size_t col = 0; col < numberOfCoordinates; ++col)
    {
        const auto fPlus =
            evaluateForces(evaluator, simulationBox, col, _displacement);

        for (size_t row = 0; row < numberOfCoordinates; ++row)
            hessian[row][col] = -(fPlus[row] - f0[row]) / _displacement;

        restorePositions(simulationBox, originalPositions);
    }

    evaluator.evaluate();
    symmetrize(hessian);

    return hessian;
}

/**
 * @brief Construct a new Five Point Force Difference Hessian Builder:: Five
 * Point Force Difference Hessian Builder object
 *
 * @param displacement
 */
HessianMatrix FivePointForceDifferenceHessianBuilder::build(
    Evaluator                    &evaluator,
    simulationBox::SimulationBox &simulationBox
) const
{
    const auto numberOfCoordinates = 3 * simulationBox.getNumberOfAtoms();
    auto       hessian             = HessianMatrix(
        numberOfCoordinates,
        std::vector<double>(numberOfCoordinates, 0.0)
    );

    const auto originalPositions = simulationBox.getPositions();

    for (size_t col = 0; col < numberOfCoordinates; ++col)
    {
        const auto fPlus =
            evaluateForces(evaluator, simulationBox, col, _displacement);
        const auto fMinus =
            evaluateForces(evaluator, simulationBox, col, -_displacement);
        const auto fPlus2 =
            evaluateForces(evaluator, simulationBox, col, 2.0 * _displacement);
        const auto fMinus2 =
            evaluateForces(evaluator, simulationBox, col, -2.0 * _displacement);

        for (size_t row = 0; row < numberOfCoordinates; ++row)
        {
            const auto derivative = (-fPlus2[row] + 8.0 * fPlus[row] -
                                     8.0 * fMinus[row] + fMinus2[row]) /
                                    (12.0 * _displacement);

            hessian[row][col] = -derivative;
        }

        restorePositions(simulationBox, originalPositions);
    }

    evaluator.evaluate();
    symmetrize(hessian);

    return hessian;
}

/**
 * @brief Construct a new Analytic Hessian Builder:: Analytic Hessian Builder
 * object
 */
HessianMatrix AnalyticHessianBuilder::build(
    Evaluator &evaluator,
    simulationBox::SimulationBox &
) const
{
    if (!evaluator.supportsAnalyticHessian())
        throw UserInputException(
            "The selected evaluator does not support analytic Hessians."
        );

    auto hessian = evaluator.calculateAnalyticHessian();
    ForceDifferenceHessianBuilder::symmetrize(hessian);

    return hessian;
}

/**
 * @brief factory function to create a HessianBuilder object based on the
 * HessianBuilderType
 *
 * @param builder
 * @param displacement
 * @return std::shared_ptr<HessianBuilder>
 */
std::shared_ptr<HessianBuilder> opt::makeHessianBuilder(
    const HessianBuilderType builder,
    const double             displacement
)
{
    using enum HessianBuilderType;

    // TODO: use switch statement
    if (builder == FINITE_DIFFERENCE_FORCES_CENTRAL)
        return std::make_shared<CentralForceDifferenceHessianBuilder>(
            displacement
        );

    if (builder == FINITE_DIFFERENCE_FORCES_FORWARD)
        return std::make_shared<ForwardForceDifferenceHessianBuilder>(
            displacement
        );

    if (builder == FINITE_DIFFERENCE_FORCES_FIVE_POINT)
        return std::make_shared<FivePointForceDifferenceHessianBuilder>(
            displacement
        );

    if (builder == ANALYTIC)
        return std::make_shared<AnalyticHessianBuilder>();

    throw UserInputException("Unknown Hessian builder.");
}
