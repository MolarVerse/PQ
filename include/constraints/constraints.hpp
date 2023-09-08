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