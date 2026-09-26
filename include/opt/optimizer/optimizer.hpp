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
#include "vector3d.hpp"

namespace molsys
{
    class SimulationBox;   // forward declaration
}   // namespace molsys

namespace physicalData
{
    class PhysicalData;   // forward declaration
}   // namespace physicalData

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
       private:
        struct Impl;
        std::unique_ptr<Impl> _impl;

       protected:
        size_t _nEpochs = 0;

        opt::Convergence _convergence;

        std::deque<double>                     _energyHistory;
        std::deque<double>                     _maxForceHistory;
        std::deque<double>                     _rmsForceHistory;
        std::deque<std::vector<linalg::Vec3D>> _forceHistory;
        std::deque<std::vector<linalg::Vec3D>> _positionHistory;

       public:
        explicit Optimizer(size_t);

        Optimizer() = default;
        virtual ~Optimizer();

        virtual void update(double, size_t) = 0;
        [[nodiscard]]
        virtual size_t maxHistoryLength() const = 0;

        void               updateHistory();
        [[nodiscard]] bool hasConverged();

        /***************************
         * standard setter methods *
         ***************************/

        void setConvergence(opt::Convergence);

        void setSimulationBox(const std::shared_ptr<molsys::SimulationBox>&);
        void setPhysicalData(
            const std::shared_ptr<physicalData::PhysicalData>&
        );
        void setPhysicalDataOld(
            const std::shared_ptr<physicalData::PhysicalData>&
        );

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getNEpochs() const;
        [[nodiscard]] size_t getHistoryIndex(int offset) const;

        [[nodiscard]] double getEnergy() const;
        [[nodiscard]] double getEnergy(int) const;

        [[nodiscard]] double getRMSForce() const;
        [[nodiscard]] double getRMSForce(int) const;

        [[nodiscard]] double getMaxForce() const;
        [[nodiscard]] double getMaxForce(int) const;

        [[nodiscard]]
        std::vector<linalg::Vec3D> getForces() const;
        [[nodiscard]]
        std::vector<linalg::Vec3D> getForces(int) const;

        [[nodiscard]]
        std::vector<linalg::Vec3D> getPositions() const;
        [[nodiscard]]
        std::vector<linalg::Vec3D> getPositions(int) const;

        [[nodiscard]] opt::Convergence& getConvergence();
        [[nodiscard]] opt::Convergence  getConvergence() const;

       protected:
        [[nodiscard]]
        molsys::SimulationBox& _getSimulationBox() const;
    };

}   // namespace opt

#endif   // _OPTIMIZER_HPP_
