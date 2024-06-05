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
    if (!_shakeActivated)
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
    if (!_shakeActivated)
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

/**
 * @brief applies the distance constraints to all distance constraints
 *
 * @param simulationBox
 * @param time
 *
 */
void Constraints::applyDistanceConstraints(
    const simulationBox::SimulationBox &simulationBox,
    physicalData::PhysicalData         &data,
    const double                        time
)
{
    if (!_distanceConstActivated)
        return;

    auto effective_time = time - _startTime;

    effective_time = effective_time > 0.0 ? effective_time : -1.0;

    std::ranges::for_each(
        _distanceConstraints,
        [&simulationBox, effective_time](auto &distanceConstraint) {
            distanceConstraint.applyDistanceConstraint(
                simulationBox,
                effective_time
            );
        }
    );

    auto lowerEnergy = 0.0;
    auto upperEnergy = 0.0;

    std::ranges::for_each(
        _distanceConstraints,
        [&lowerEnergy](const auto &distanceConstraint)
        { lowerEnergy += distanceConstraint.getLowerEnergy(); }
    );

    std::ranges::for_each(
        _distanceConstraints,
        [&upperEnergy](const auto &distanceConstraint)
        { upperEnergy += distanceConstraint.getUpperEnergy(); }
    );

    data.setLowerDistanceConstraints(lowerEnergy);
    data.setUpperDistanceConstraints(upperEnergy);
}

/*****************************
 *                           *
 * standard activate methods *
 *                           *
 *****************************/

/**
 * @brief activates the shake algorithm
 *
 */
void Constraints::activateDistanceConstraints()
{
    _distanceConstActivated = true;
}

/**
 * @brief deactivates the shake algorithm
 *
 */
void Constraints::deactivateDistanceConstraints()
{
    _distanceConstActivated = false;
}

/**
 * @brief checks if shake algorithm is active
 *
 * @return true if shake algorithm is active
 */
bool Constraints::isShakeActive() const { return _shakeActivated; }

/**
 * @brief checks if mShake algorithm is active
 *
 * @return true if mShake algorithm is active
 */
bool Constraints::isMShakeActive() const { return _mShakeActivated; }

/**
 * @brief checks if distance constraints are active
 *
 * @return true if distance constraints are active
 */
bool Constraints::isDistanceConstraintsActive() const
{
    return _distanceConstActivated;
}

/**
 * @brief checks if any constraint is active
 *
 */
bool Constraints::isActive() const
{
    return _shakeActivated || _mShakeActivated || _distanceConstActivated;
}

/************************
 *                      *
 * standard add methods *
 *                      *
 ************************/

/**
 * @brief adds a bond constraint to the constraints
 *
 * @param bondConstraint
 *
 */
void Constraints::addBondConstraint(const BondConstraint &bondConstraint)
{
    _bondConstraints.push_back(bondConstraint);
}

/**
 * @brief adds a distance constraint to the constraints
 *
 * @param distanceConstraint
 *
 */
void Constraints::addDistanceConstraint(
    const DistanceConstraint &distanceConstraint
)
{
    _distanceConstraints.push_back(distanceConstraint);
}

/**
 * @brief adds a mShake reference to the constraints
 *
 * @param mShakeReference
 *
 */
void Constraints::addMShakeReference(const MShakeReference &mShakeReference)
{
    _mShakeReferences.push_back(mShakeReference);
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief returns all bond constraints
 *
 * @return all bond constraints
 */
const std::vector<BondConstraint> &Constraints::getBondConstraints() const
{
    return _bondConstraints;
}

/**
 * @brief returns all distance constraints
 *
 * @return all distance constraints
 */
const std::vector<DistanceConstraint> &Constraints::getDistanceConstraints(
) const
{
    return _distanceConstraints;
}

/**
 * @brief returns all mShake references
 *
 * @return all mShake references
 */
const std::vector<MShakeReference> &Constraints::getMShakeReferences() const
{
    return _mShakeReferences;
}

/**
 * @brief returns the number of bond constraints
 *
 * @return the number of bond constraints
 */
size_t Constraints::getNumberOfBondConstraints() const
{
    return _bondConstraints.size();
}

/**
 * @brief returns the number of distance constraints
 *
 * @return the number of distance constraints
 */
size_t Constraints::getNumberOfDistanceConstraints() const
{
    return _distanceConstraints.size();
}

/**
 * @brief returns the maximum number of iterations for the shake algorithm
 *
 * @return the maximum number of iterations for the shake algorithm
 */
size_t Constraints::getShakeMaxIter() const { return _shakeMaxIter; }

/**
 * @brief returns the maximum number of iterations for the rattle algorithm
 *
 * @return the maximum number of iterations for the rattle algorithm
 */
size_t Constraints::getRattleMaxIter() const { return _rattleMaxIter; }

/**
 * @brief returns the shake tolerance
 *
 * @return the shake tolerance
 */
double Constraints::getShakeTolerance() const { return _shakeTolerance; }

/**
 * @brief returns the rattle tolerance
 *
 * @return the rattle tolerance
 */
double Constraints::getRattleTolerance() const { return _rattleTolerance; }

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief sets the maximum number of iterations for the shake algorithm
 *
 * @param shakeMaxIter
 */
void Constraints::setShakeMaxIter(const size_t shakeMaxIter)
{
    _shakeMaxIter = shakeMaxIter;
}

/**
 * @brief sets the maximum number of iterations for the rattle algorithm
 *
 * @param rattleMaxIter
 */
void Constraints::setRattleMaxIter(const size_t rattleMaxIter)
{
    _rattleMaxIter = rattleMaxIter;
}

/**
 * @brief sets the shake tolerance
 *
 * @param shakeTolerance
 */
void Constraints::setShakeTolerance(const double shakeTolerance)
{
    _shakeTolerance = shakeTolerance;
}

/**
 * @brief sets the rattle tolerance
 *
 * @param rattleTolerance
 */
void Constraints::setRattleTolerance(const double rattleTolerance)
{
    _rattleTolerance = rattleTolerance;
}