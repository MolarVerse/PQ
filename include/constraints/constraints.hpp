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
        bool _activated = defaults::_CONSTRAINTS_ARE_ACTIVE_DEFAULT_;

        size_t _shakeMaxIter  = defaults::_SHAKE_MAX_ITER_DEFAULT_;
        size_t _rattleMaxIter = defaults::_RATTLE_MAX_ITER_DEFAULT_;

        double _shakeTolerance  = defaults::_SHAKE_TOLERANCE_DEFAULT_;
        double _rattleTolerance = defaults::_RATTLE_TOLERANCE_DEFAULT_;

        std::vector<BondConstraint> _bondConstraints;

      public:
        void calculateConstraintBondRefs(const simulationBox::SimulationBox &simulationBox);

        void applyShake(const simulationBox::SimulationBox &simulationBox);
        void applyRattle();

        /*****************************
         * standard activate methods *
         *****************************/

        void               activate() { _activated = true; }
        void               deactivate() { _activated = false; }
        [[nodiscard]] bool isActive() const { return _activated; }

        /************************
         * standard add methods *
         ************************/

        void addBondConstraint(const BondConstraint &bondConstraint) { _bondConstraints.push_back(bondConstraint); }

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] const std::vector<BondConstraint> &getBondConstraints() const { return _bondConstraints; }
        [[nodiscard]] size_t                             getNumberOfBondConstraints() const { return _bondConstraints.size(); }

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