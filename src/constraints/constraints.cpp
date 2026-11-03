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

#include <algorithm>
#include <format>
#include <vector>

#include "constraintSettings.hpp"
#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "mShake.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"

namespace constraints
{

    /**
     * @brief constructor
     *
     **/
    Constraints::Constraints(const ConstraintSettings &constraintSettings)
        : _constraintSettings(constraintSettings),
          _mShake(std::make_unique<MShake>())
    {
    }

    Constraints::~Constraints() = default;

    /**
     * @brief init M-Shake from M-Shake references
     *
     */
    void Constraints::initMShake() { _mShake->initMShake(); }

    /**
     * @brief calculates the reference bond data of all bond constraints
     *
     * @param simulationBox
     *
     */
    void Constraints::calculateConstraintBondRefs(
        const molsys::SimulationBox &simulationBox
    )
    {
        auto _ = scopedTimer(TimerId::Constraints, "Reference Bond Data");

        std::ranges::for_each(
            _bondConstraints,
            [&simulationBox](auto &bondConstraint)
            { bondConstraint.calculateConstraintBondRef(simulationBox); }
        );
    }

    /**
     * @brief applies both shake and mShake algorithm to all bond constraints
     *
     * @param simulationBox
     */
    void Constraints::applyShake(molsys::SimulationBox &simulationBox)
    {
        if (!_constraintSettings->isShakeLikeActive())
            return;

        if (_constraintSettings->isShakeActivated())
            _applyShake(simulationBox);

        if (_constraintSettings->isMShakeActivated())
            _applyMShake(simulationBox);
    }

    /**
     * @brief applies the shake algorithm to all bond constraints
     *
     * @param simulationBox
     *
     * @throws exc::ShakeException if shake algorithm does not
     * converge
     */
    void Constraints::_applyShake(molsys::SimulationBox &simulationBox)
    {
        auto _ = scopedTimer(TimerId::Constraints, "Shake");

        std::vector<bool> convergedVector;
        bool              converged = false;

        size_t iter = 0;

        while (!converged && iter <= _constraintSettings->getShakeMaxIter())
        {
            convergedVector.clear();

            auto applyShakeSingleBond =
                [&simulationBox, &convergedVector, this](auto &bondConst)
            {
                convergedVector.push_back(bondConst.applyShake(
                    simulationBox,
                    _constraintSettings->getShakeTolerance()
                ));
            };

            std::ranges::for_each(_bondConstraints, applyShakeSingleBond);

            converged = std::ranges::all_of(
                convergedVector,
                [](const bool isConverged) { return isConverged; }
            );

            ++iter;
        }

        if (!converged)
        {
            throw exc::ShakeException(
                std::format(
                    "Shake algorithm did not converge for {} bonds.",
                    std::ranges::count(convergedVector, false)
                )
            );
        }
    }

    /**
     * @brief applies the mShake algorithm to all bond constraints
     *
     * @param simulationBox
     *
     */
    void Constraints::_applyMShake(molsys::SimulationBox &simulationBox)
    {
        auto _ = scopedTimer(TimerId::Constraints, "MShake - Shake");
        _mShake->applyMShake(
            simulationBox,
            _constraintSettings->getMShakeMaxIter(),
            _constraintSettings->getMShakeTolerance()
        );
    }

    /**
     * @brief applies the rattle algorithm to all bond constraints
     *
     * @throws exc::ShakeException if rattle algorithm does not
     * converge
     *
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     */
    void Constraints::applyRattle(molsys::SimulationBox &simulationBox)
    {
        if (!_constraintSettings->isShakeLikeActive())
            return;

        if (_constraintSettings->isShakeActivated())
            _applyRattle();

        if (_constraintSettings->isMShakeActivated())
            _applyMRattle(simulationBox);
    }

    /**
     * @brief applies the rattle algorithm to all bond constraints
     *
     * @throws exc::ShakeException if rattle algorithm does not
     * converge
     */
    void Constraints::_applyRattle()
    {
        auto _ = scopedTimer(TimerId::Constraints, "Rattle");

        std::vector<bool> convergedVector;
        bool              converged = false;

        size_t iter = 0;

        while (!converged && iter <= _constraintSettings->getRattleMaxIter())
        {
            convergedVector.clear();

            auto applyRattleForSingleBond =
                [&convergedVector, this](auto &bondConstraint)
            {
                convergedVector.push_back(bondConstraint.applyRattle(
                    _constraintSettings->getRattleTolerance()
                ));
            };

            std::ranges::for_each(_bondConstraints, applyRattleForSingleBond);

            converged = std::ranges::all_of(
                convergedVector,
                [](const bool isConverged) { return isConverged; }
            );

            ++iter;
        }

        if (!converged)
        {
            throw exc::ShakeException(
                std::format(
                    "Rattle algorithm did not converge for {} bonds.",
                    std::ranges::count(convergedVector, false)
                )
            );
        }
    }

    /**
     * @brief applies M-Shake Rattle algorithm
     *
     * @param simulationBox
     */
    void Constraints::_applyMRattle(molsys::SimulationBox &simulationBox)
    {
        auto _ = scopedTimer(TimerId::Constraints, "MShake - Rattle");
        _mShake->applyMRattle(simulationBox);
    }

    /**
     * @brief applies the distance constraints to all distance constraints
     *
     * @param simulationBox
     * @param physicalData the physical data object to update with constraint
     * energies
     * @param time
     *
     */
    void Constraints::applyDistanceConstraints(
        const molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData  &physicalData,
        double                       time
    )
    {
        if (!_constraintSettings->isDistanceConstraintsActivated())
            return;

        auto effective_time = time - _startTime;

        effective_time = effective_time > 0.0 ? effective_time : -1;

        std::ranges::for_each(
            _distanceConstraints,
            [&simulationBox, effective_time](auto &distanceConstraint)
            {
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

        physicalData.setLowerDistanceConstraints(lowerEnergy);
        physicalData.setUpperDistanceConstraints(upperEnergy);
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
        _mShake->addMShakeReference(mShakeReference);
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
    const std::vector<DistanceConstraint> &Constraints::getDistConstraints(
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
        return _mShake->getMShakeReferences();
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
     * @brief returns the number of mShake constraints
     *
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     *
     * @return the number of mShake constraints
     */
    size_t Constraints::getNumberOfMShakeConstraints(
        molsys::SimulationBox &simulationBox
    ) const
    {
        return _mShake->calcNumberOfBondConstraints(simulationBox);
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

}   // namespace constraints
