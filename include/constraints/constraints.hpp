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

#ifndef _CONSTRAINTS_HPP_

#define _CONSTRAINTS_HPP_

#include <cstddef>   // for size_t
#include <vector>    // for vector

#include "bondConstraint.hpp"       // for BondConstraint
#include "defaults.hpp"             // for defaults
#include "distanceConstraint.hpp"   // for DistanceConstraint
#include "mShake.hpp"               // for MShake
#include "mShakeReference.hpp"      // for MShakeReference
#include "timer.hpp"                // for Timer
#include "typeAliases.hpp"

/**
 * @brief namespace for all constraints
 */
namespace constraints
{
    /**
     * @class Constraints
     *
     * @brief class containing all constraints
     *
     * @details it performs the shake and rattle algorithm on all bond
     * constraints
     */
    class Constraints : public timings::Timer
    {
       private:
        MShake _mShake;

        bool _shakeActivated         = defaults::CONSTRAINTS_ACTIVE_DEFAULT;
        bool _mShakeActivated        = defaults::CONSTRAINTS_ACTIVE_DEFAULT;
        bool _distanceConstActivated = defaults::CONSTRAINTS_ACTIVE_DEFAULT;

        size_t _shakeMaxIter  = defaults::SHAKE_MAX_ITER_DEFAULT;
        size_t _rattleMaxIter = defaults::RATTLE_MAX_ITER_DEFAULT;

        double _shakeTolerance  = defaults::SHAKE_TOLERANCE_DEFAULT;
        double _rattleTolerance = defaults::RATTLE_TOLERANCE_DEFAULT;
        double _startTime       = 0.0;

        std::vector<BondConstraint>     _bondConstraints;
        std::vector<DistanceConstraint> _distanceConstraints;

       public:
        std::shared_ptr<Constraints> clone() const;

        void calculateConstraintBondRefs(const pq::SimBox &simulationBox);

        void initMShake();

        void applyShake(pq::SimBox &simulationBox);
        void _applyShake(pq::SimBox &simulationBox);
        void _applyMShake(pq::SimBox &simulationBox);

        void applyRattle(pq::SimBox &simulationBox);
        void _applyRattle();
        void _applyMRattle(pq::SimBox &simulationBox);

        void applyDistanceConstraints(
            const pq::SimBox &,
            pq::PhysicalData &,
            const double
        );

        /*****************************
         * standard activate methods *
         *****************************/

        void activateShake() { _shakeActivated = true; }
        void deactivateShake() { _shakeActivated = false; }
        void activateMShake() { _mShakeActivated = true; }
        void deactivateMShake() { _mShakeActivated = false; }
        void activateDistanceConstraints();
        void deactivateDistanceConstraints();

        [[nodiscard]] bool isShakeActive() const;
        [[nodiscard]] bool isMShakeActive() const;
        [[nodiscard]] bool isShakeLikeActive() const;
        [[nodiscard]] bool isDistanceConstraintsActive() const;
        [[nodiscard]] bool isActive() const;

        /************************
         * standard add methods *
         ************************/

        void addBondConstraint(const BondConstraint &bondConstraint);
        void addDistanceConstraint(const DistanceConstraint &distanceConst);
        void addMShakeReference(const MShakeReference &mShakeReference);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] const pq::BondConstraintsVec &getBondConstraints() const;
        [[nodiscard]] const pq::DistConstraintsVec &getDistConstraints() const;
        [[nodiscard]] const pq::MShakeReferenceVec &getMShakeReferences() const;

        [[nodiscard]] size_t getNumberOfBondConstraints() const;
        [[nodiscard]] size_t getNumberOfMShakeConstraints(pq::SimBox &) const;
        [[nodiscard]] size_t getNumberOfDistanceConstraints() const;

        [[nodiscard]] size_t getShakeMaxIter() const;
        [[nodiscard]] size_t getRattleMaxIter() const;
        [[nodiscard]] double getShakeTolerance() const;
        [[nodiscard]] double getRattleTolerance() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setShakeMaxIter(const size_t shakeMaxIter);
        void setRattleMaxIter(const size_t rattleMaxIter);
        void setShakeTolerance(const double shakeTolerance);
        void setRattleTolerance(const double rattleTolerance);

        void setStartTime(const double startTime) { _startTime = startTime; }
    };

}   // namespace constraints

#endif   // _CONSTRAINTS_HPP_
