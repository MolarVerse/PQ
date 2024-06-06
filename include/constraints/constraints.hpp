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
#include "mShakeReference.hpp"      // for MShakeReference
#include "matrix.hpp"               // for Matrix
#include "physicalData.hpp"         // for PhysicalData
#include "timer.hpp"                // for Timer

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

/**
 * @brief namespace for all constraints
 */
namespace constraints
{

    using SimBox       = simulationBox::SimulationBox;
    using PhysicalData = physicalData::PhysicalData;

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
        bool _shakeActivated         = defaults::_CONSTRAINTS_ACTIVE_DEFAULT_;
        bool _mShakeActivated        = defaults::_CONSTRAINTS_ACTIVE_DEFAULT_;
        bool _distanceConstActivated = defaults::_CONSTRAINTS_ACTIVE_DEFAULT_;

        size_t _shakeMaxIter  = defaults::_SHAKE_MAX_ITER_DEFAULT_;
        size_t _rattleMaxIter = defaults::_RATTLE_MAX_ITER_DEFAULT_;

        double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;
        double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;
        double _startTime       = 0.0;

        std::vector<BondConstraint>      _bondConstraints;
        std::vector<DistanceConstraint>  _distanceConstraints;
        std::vector<MShakeReference>     _mShakeReferences;
        std::vector<std::vector<double>> _mShakeRSquaredRefs;

        std::vector<linearAlgebra::Matrix<double>> _mShakeMatrices;
        std::vector<linearAlgebra::Matrix<double>> _mShakeInvMatrices;

       public:
        void calculateConstraintBondRefs(const SimBox &simulationBox);

        void initMShake();

        void applyShake(const SimBox &simulationBox);
        void applyRattle();
        void applyDistanceConstraints(
            const SimBox &simBox,
            PhysicalData &data,
            const double  dt
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

        [[nodiscard]] const std::vector<BondConstraint> &getBondConstraints(
        ) const;
        [[nodiscard]] const std::vector<DistanceConstraint> &getDistanceConstraints(
        ) const;
        [[nodiscard]] const std::vector<MShakeReference> &getMShakeReferences(
        ) const;

        [[nodiscard]] size_t getNumberOfBondConstraints() const;
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
