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
#include <cstddef>
#include <format>
#include <vector>

#include "constraintSettings.hpp"
#include "exceptions.hpp"
#include "globalTimer.hpp"
#include "mShake.hpp"
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "unit_types.hpp"
#include "units.hpp"

namespace constraints
{
    namespace
    {
        // sets how tightly the RATTLE tolerance holds on the forces
        constexpr Time OPT_TIME_STEP = 1.0 * fs;

        // a projected full force needs more passes than an MD step
        constexpr size_t OPT_MAX_ITER = 1000;
    }   // namespace

    /**
     * @brief constructor
     *
     **/
    Constraints::Constraints(const ConstraintSettings &constraintSettings)
        : _constraintSettings(constraintSettings),
          _mShake(std::make_unique<MShake>()),
          _startTime(0.0 * fs)
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
     * @param simulationBox the simulation box to apply periodic boundary
     * conditions
     * @param timeStep the integration time step used for velocity updates
     */
    void Constraints::applyShake(
        molsys::SimulationBox &simulationBox,
        const Time            &timeStep
    )
    {
        if (!_constraintSettings->isShakeLikeActive())
            return;

        if (_constraintSettings->isShakeActivated())
        {
            _applyShake(
                simulationBox,
                timeStep,
                _constraintSettings->getShakeMaxIter()
            );
        }

        if (_constraintSettings->isMShakeActivated())
            _applyMShake(simulationBox, timeStep);
    }

    /**
     * @brief applies the shake algorithm to all bond constraints
     *
     * @param simulationBox
     * @param timeStep the integration time step used for velocity updates
     * @param maxIter
     *
     * @throws exc::ShakeException if shake algorithm does not
     * converge
     */
    void Constraints::_applyShake(
        molsys::SimulationBox &simulationBox,
        const Time            &timeStep,
        size_t                 maxIter
    )
    {
        auto _ = scopedTimer(TimerId::Constraints, "Shake");

        std::vector<bool> convergedVector;
        bool              converged = false;

        size_t iter = 0;

        while (!converged && iter <= maxIter)
        {
            convergedVector.clear();

            auto applyShakeSingleBond = [&simulationBox,
                                         &convergedVector,
                                         &timeStep,
                                         this](auto &bondConst)
            {
                convergedVector.push_back(bondConst.applyShake(
                    simulationBox,
                    _constraintSettings->getShakeTolerance(),
                    timeStep
                ));
            };

            std::ranges::for_each(_bondConstraints, applyShakeSingleBond);

            converged = std::ranges::all_of(
                convergedVector,
                [](bool isConverged) { return isConverged; }
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
     * @param timeStep the integration time step used for velocity updates
     *
     */
    void Constraints::_applyMShake(
        molsys::SimulationBox &simulationBox,
        const Time            &timeStep
    )
    {
        auto _ = scopedTimer(TimerId::Constraints, "MShake - Shake");
        _mShake->applyMShake(
            simulationBox,
            _constraintSettings->getMShakeMaxIter(),
            _constraintSettings->getMShakeTolerance(),
            timeStep
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
        {
            _applyRattle(_constraintSettings->getRattleMaxIter());
        }

        if (_constraintSettings->isMShakeActivated())
            _applyMRattle(simulationBox);
    }

    /**
     * @brief applies the rattle algorithm to all bond constraints
     *
     * @param maxIter
     *
     * @throws exc::ShakeException if rattle algorithm does not
     * converge
     */
    void Constraints::_applyRattle(size_t maxIter)
    {
        auto _ = scopedTimer(TimerId::Constraints, "Rattle");

        std::vector<bool> convergedVector;
        bool              converged = false;

        size_t iter = 0;

        while (!converged && iter <= maxIter)
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
                [](bool isConverged) { return isConverged; }
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
     * @brief corrects the positions with SHAKE or M-SHAKE, keeping the
     * velocities
     *
     * @details for geometry optimizations, which have no time step; SHAKE
     * iterates up to at least OPT_MAX_ITER times
     *
     * @param simulationBox
     */
    void Constraints::applyShakeToPositions(
        molsys::SimulationBox &simulationBox
    )
    {
        if (!_constraintSettings->isShakeLikeActive())
        {
            return;
        }

        const auto velocities = simulationBox.getVelocities();

        if (_constraintSettings->isShakeActivated())
        {
            _applyShake(
                simulationBox,
                OPT_TIME_STEP,
                std::max(_constraintSettings->getShakeMaxIter(), OPT_MAX_ITER)
            );
        }

        if (_constraintSettings->isMShakeActivated())
        {
            _applyMShake(simulationBox, OPT_TIME_STEP);
        }

        _setVelocities(simulationBox, velocities);
    }

    /**
     * @brief removes the constraint components from the forces
     *
     * @details applies RATTLE to the velocities M^-1 F dt, so the forces
     * become M P M^-1 F with P the RATTLE projection. They vanish exactly at a
     * constrained minimum. The velocities are kept. RATTLE iterates up to at
     * least OPT_MAX_ITER times.
     *
     * @param simulationBox
     */
    void Constraints::projectForces(molsys::SimulationBox &simulationBox)
    {
        if (!_constraintSettings->isShakeLikeActive())
        {
            return;
        }

        const auto velocities = simulationBox.getVelocities();

        for (const auto &atom : simulationBox.getAtoms())
        {
            atom->setVelocity(
                OPT_TIME_STEP * atom->getForce() / atom->getMass()
            );
        }

        if (_constraintSettings->isShakeActivated())
        {
            _applyRattle(
                std::max(_constraintSettings->getRattleMaxIter(), OPT_MAX_ITER)
            );
        }

        if (_constraintSettings->isMShakeActivated())
        {
            _applyMRattle(simulationBox);
        }

        for (const auto &atom : simulationBox.getAtoms())
        {
            atom->setForce(
                atom->getMass() * atom->getVelocity() / OPT_TIME_STEP
            );
        }

        _setVelocities(simulationBox, velocities);
    }

    /**
     * @brief sets the velocities of all atoms in the order of getAtoms()
     *
     * @param simulationBox
     * @param velocities
     */
    void Constraints::_setVelocities(
        molsys::SimulationBox    &simulationBox,
        const std::vector<Vel3D> &velocities
    )
    {
        auto &atoms = simulationBox.getAtoms();

        for (size_t index = 0; index < atoms.size(); ++index)
        {
            atoms[index]->setVelocity(velocities[index]);
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
        const Time                  &time
    )
    {
        if (!_constraintSettings->isDistanceConstraintsActivated())
            return;

        auto effective_time = time - _startTime;

        effective_time = effective_time > 0.0 * fs ? effective_time : -1 * fs;

        _applyDistanceConstraints(simulationBox, physicalData, effective_time);
    }

    /**
     * @brief applies the distance constraints with the time independent
     * spring constant
     *
     * @details for geometry optimizations, which have no simulation time
     *
     * @param simulationBox
     * @param physicalData the physical data object to update with constraint
     * energies
     */
    void Constraints::applyStaticDistanceConstraints(
        const molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData  &physicalData
    )
    {
        if (!_constraintSettings->isDistanceConstraintsActivated())
        {
            return;
        }

        _applyDistanceConstraints(simulationBox, physicalData, 0.0 * fs);
    }

    /**
     * @brief applies all distance constraints and stores their energies
     *
     * @param simulationBox
     * @param physicalData
     * @param timeInterval time since the start of the constraints, negative
     * to skip them
     */
    void Constraints::_applyDistanceConstraints(
        const molsys::SimulationBox &simulationBox,
        physicalData::PhysicalData  &physicalData,
        const Time                  &timeInterval
    )
    {
        std::ranges::for_each(
            _distanceConstraints,
            [&simulationBox, timeInterval](auto &distanceConstraint)
            {
                distanceConstraint.applyDistanceConstraint(
                    simulationBox,
                    timeInterval
                );
            }
        );

        auto lowerEnergy = EnergyPerMol::zero();
        auto upperEnergy = EnergyPerMol::zero();

        std::ranges::for_each(
            _distanceConstraints,
            [&lowerEnergy](const auto &distanceConstraint)
            { lowerEnergy += distanceConstraint.get_lowerEnergy(); }
        );

        std::ranges::for_each(
            _distanceConstraints,
            [&upperEnergy](const auto &distanceConstraint)
            { upperEnergy += distanceConstraint.get_upperEnergy(); }
        );

        physicalData.set_lowerDistanceConstraints(lowerEnergy);
        physicalData.set_upperDistanceConstraints(upperEnergy);
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
     * @brief returns the number of molecules constrained by M-Shake
     *
     * @param simulationBox
     * @return size_t
     */
    size_t Constraints::getNumberOfMShakeMolecules(
        molsys::SimulationBox &simulationBox
    ) const
    {
        return _mShake->calcNumberOfMShakeMolecules(simulationBox);
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
