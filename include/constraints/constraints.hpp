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

#include "bondConstraint.hpp"
#include "defaults.hpp"
#include "distanceConstraint.hpp"

#include <cstddef>
#include <vector>

namespace simulationBox
{
    class SimulationBox;   // forward declaration
}

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
     * @details it performs the shake and rattle algorithm on all bond constraints
     */
    class Constraints
    {
      private:
        bool _shakeActivated               = defaults::_CONSTRAINTS_ARE_ACTIVE_DEFAULT_;
        bool _distanceConstraintsActivated = defaults::_CONSTRAINTS_ARE_ACTIVE_DEFAULT_;

        size_t _shakeMaxIter  = defaults::_SHAKE_MAX_ITER_DEFAULT_;
        size_t _rattleMaxIter = defaults::_RATTLE_MAX_ITER_DEFAULT_;

        double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;
        double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;

        std::vector<BondConstraint>     _bondConstraints;
        std::vector<DistanceConstraint> _distanceConstraints;

      public:
        void calculateConstraintBondRefs(const simulationBox::SimulationBox &simulationBox);

        void applyShake(const simulationBox::SimulationBox &simulationBox);
        void applyRattle();
        void applyDistanceConstraints(const simulationBox::SimulationBox &simulationBox);

        /*****************************
         * standard activate methods *
         *****************************/

        void               activateShake() { _shakeActivated = true; }
        void               deactivateShake() { _shakeActivated = false; }
        void               activateDistanceConstraints() { _distanceConstraintsActivated = true; }
        void               deactivateDistanceConstraints() { _distanceConstraintsActivated = false; }
        [[nodiscard]] bool isShakeActive() const { return _shakeActivated; }
        [[nodiscard]] bool isDistanceConstraintsActive() const { return _distanceConstraintsActivated; }
        [[nodiscard]] bool isActive() const { return _shakeActivated || _distanceConstraintsActivated; }

        /************************
         * standard add methods *
         ************************/

        void addBondConstraint(const BondConstraint &bondConstraint) { _bondConstraints.push_back(bondConstraint); }
        void addDistanceConstraint(const DistanceConstraint &distanceConstraint)
        {
            _distanceConstraints.push_back(distanceConstraint);
        }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] const std::vector<BondConstraint>     &getBondConstraints() const { return _bondConstraints; }
        [[nodiscard]] const std::vector<DistanceConstraint> &getDistanceConstraints() const { return _distanceConstraints; }

        [[nodiscard]] size_t getNumberOfBondConstraints() const { return _bondConstraints.size(); }
        [[nodiscard]] size_t getNumberOfDistanceConstraints() const { return _distanceConstraints.size(); }

        [[nodiscard]] size_t getShakeMaxIter() const { return _shakeMaxIter; }
        [[nodiscard]] size_t getRattleMaxIter() const { return _rattleMaxIter; }
        [[nodiscard]] double getShakeTolerance() const { return _shakeTolerance; }
        [[nodiscard]] double getRattleTolerance() const { return _rattleTolerance; }

        /***************************
         * standard setter methods *
         ***************************/

        void setShakeMaxIter(const size_t shakeMaxIter) { _shakeMaxIter = shakeMaxIter; }
        void setRattleMaxIter(const size_t rattleMaxIter) { _rattleMaxIter = rattleMaxIter; }
        void setShakeTolerance(const double shakeTolerance) { _shakeTolerance = shakeTolerance; }
        void setRattleTolerance(const double rattleTolerance) { _rattleTolerance = rattleTolerance; }
    };

}   // namespace constraints

#endif   // _CONSTRAINTS_HPP_