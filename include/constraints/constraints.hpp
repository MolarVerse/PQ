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

#include <cstddef>
#include <vector>

#include "bondConstraint.hpp"
#include "defaults.hpp"
#include "distanceConstraint.hpp"
#include "mShakeReference.hpp"

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

/**
 * @brief namespace for all constraints
 */
namespace constraints
{
    class MShake;   // forward declaration

    /**
     * @class Constraints
     *
     * @brief class containing all constraints
     *
     * @details it performs the shake and rattle algorithm on all bond
     * constraints
     */
    class Constraints
    {
       private:
        std::unique_ptr<MShake> _mShake;

        size_t _shakeMaxIter  = defaults::SHAKE_MAX_ITER_DEFAULT;
        size_t _rattleMaxIter = defaults::RATTLE_MAX_ITER_DEFAULT;

        double _shakeTolerance  = defaults::SHAKE_TOLERANCE_DEFAULT;
        double _rattleTolerance = defaults::RATTLE_TOLERANCE_DEFAULT;
        double _startTime       = 0.0;

        std::vector<BondConstraint>     _bondConstraints;
        std::vector<DistanceConstraint> _distanceConstraints;

       public:
        Constraints();
        ~Constraints();

        void calculateConstraintBondRefs(
            const molsys::SimulationBox &simulationBox
        );

        void initMShake();

        void applyShake(molsys::SimulationBox &simulationBox);
        void applyRattle(molsys::SimulationBox &simulationBox);
        void applyDistanceConstraints(
            const molsys::SimulationBox &,
            physicalData::PhysicalData &,
            double
        );

        /************************
         * standard add methods *
         ************************/

        void addBondConstraint(const BondConstraint &bondConstraint);
        void addDistanceConstraint(const DistanceConstraint &distanceConst);
        void addMShakeReference(const MShakeReference &mShakeReference);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]]
        const std::vector<BondConstraint> &getBondConstraints() const;
        [[nodiscard]]
        const std::vector<DistanceConstraint> &getDistConstraints() const;
        [[nodiscard]]
        const std::vector<MShakeReference> &getMShakeReferences() const;

        [[nodiscard]] size_t getNumberOfBondConstraints() const;
        [[nodiscard]] size_t getNumberOfMShakeConstraints(
            molsys::SimulationBox &
        ) const;
        [[nodiscard]] size_t getNumberOfDistanceConstraints() const;

        [[nodiscard]] size_t getShakeMaxIter() const;
        [[nodiscard]] size_t getRattleMaxIter() const;
        [[nodiscard]] double getShakeTolerance() const;
        [[nodiscard]] double getRattleTolerance() const;

        /***************************
         * standard setter methods *
         ***************************/

        void setShakeMaxIter(size_t shakeMaxIter);
        void setRattleMaxIter(size_t rattleMaxIter);
        void setShakeTolerance(double shakeTolerance);
        void setRattleTolerance(double rattleTolerance);

        void setStartTime(const double startTime) { _startTime = startTime; }

       private:
        void _applyShake(molsys::SimulationBox &simulationBox);
        void _applyMShake(molsys::SimulationBox &simulationBox);

        void _applyRattle();
        void _applyMRattle(molsys::SimulationBox &simulationBox);
    };

}   // namespace constraints

#endif   // _CONSTRAINTS_HPP_
