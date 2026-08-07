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

#ifndef _OPTIMIZER_HPP_

#define _OPTIMIZER_HPP_

#include <cstddef>   // for size_t
#include <deque>
#include <memory>

#include "convergence.hpp"   // for Convergence
#include "physicalData.hpp"
#include "simulationBox.hpp"
#include "vector3d.hpp"

namespace opt
{
    /**
     * @class Optimizer
     *
     * @brief Base class for all optimizers
     *
     */
    class Optimizer
    {
       protected:
        size_t _nEpochs = 0;

        opt::Convergence _convergence;

        std::shared_ptr<simulationBox::SimulationBox>
            _simulationBox;   // TODO(97gamjak): remove this via pimpl
        std::shared_ptr<physicalData::PhysicalData>
            _physicalData;   // TODO(97gamjak): remove this via pimpl
        std::shared_ptr<physicalData::PhysicalData>
            _physicalDataOld;   // TODO(97gamjak): remove this via pimpl

        std::deque<double>                            _energyHistory;
        std::deque<double>                            _maxForceHistory;
        std::deque<double>                            _rmsForceHistory;
        std::deque<std::vector<linearAlgebra::Vec3D>> _forceHistory;
        std::deque<std::vector<linearAlgebra::Vec3D>> _positionHistory;

       public:
        explicit Optimizer(const size_t);

        Optimizer()          = default;
        virtual ~Optimizer() = default;

        [[nodiscard]]
        virtual std::shared_ptr<Optimizer> clone() const = 0;
        virtual void update(const double, const size_t)  = 0;
        [[nodiscard]]
        virtual size_t maxHistoryLength() const = 0;

        void               updateHistory();
        [[nodiscard]] bool hasConverged();

        /***************************
         * standard setter methods *
         ***************************/

        void setConvergence(const opt::Convergence);

        void setSimulationBox(
            const std::shared_ptr<simulationBox::SimulationBox>
        );

        void setPhysicalData(const std::shared_ptr<physicalData::PhysicalData>);

        void setPhysicalDataOld(
            const std::shared_ptr<physicalData::PhysicalData>
        );

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNEpochs() const;

        [[nodiscard]] size_t getHistoryIndex(int offset) const;

        [[nodiscard]] double getEnergy() const;
        [[nodiscard]] double getEnergy(const int) const;

        [[nodiscard]] double getRMSForce() const;
        [[nodiscard]] double getRMSForce(const int) const;

        [[nodiscard]] double getMaxForce() const;
        [[nodiscard]] double getMaxForce(const int) const;

        [[nodiscard]]
        std::vector<linearAlgebra::Vec3D> getForces() const;
        [[nodiscard]]
        std::vector<linearAlgebra::Vec3D> getForces(const int) const;

        [[nodiscard]]
        std::vector<linearAlgebra::Vec3D> getPositions() const;
        [[nodiscard]]
        std::vector<linearAlgebra::Vec3D> getPositions(const int) const;

        [[nodiscard]] opt::Convergence &getConvergence();
        [[nodiscard]] opt::Convergence  getConvergence() const;
    };

}   // namespace opt

#endif   // _OPTIMIZER_HPP_
