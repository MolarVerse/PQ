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

#ifndef OPT__OPTIMIZER__OPTIMIZER_HPP
#define OPT__OPTIMIZER__OPTIMIZER_HPP

#include <cstddef>
#include <deque>
#include <memory>

#include "convergence.hpp"
#include "unit_types.hpp"

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
        opt::Convergence _convergence;

        std::deque<EnergyPerMol>         _energyHistory;
        std::deque<Force>                _maxForceHistory;
        std::deque<Force>                _rmsForceHistory;
        std::deque<std::vector<Force3D>> _forceHistory;
        std::deque<std::vector<Pos3D>>   _positionHistory;

        struct Impl;
        std::unique_ptr<Impl> _impl;

       public:
        Optimizer();
        virtual ~Optimizer();

        virtual void update(double, size_t) = 0;
        [[nodiscard]]
        virtual size_t maxHistoryLength() const = 0;

        void               updateHistory(const EnergyPerMol& totalEnergy);
        [[nodiscard]] bool hasConverged();

        /***************************
         * standard setter methods *
         ***************************/

        void setConvergence(opt::Convergence);

        void setSimulationBox(const std::shared_ptr<molsys::SimulationBox>&);

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] size_t getHistoryIndex(int offset) const;

        [[nodiscard]] const EnergyPerMol& getEnergy() const;
        [[nodiscard]] const EnergyPerMol& getEnergy(int) const;

        [[nodiscard]] const Force& getRMSForce() const;
        [[nodiscard]] const Force& getRMSForce(int) const;

        [[nodiscard]] const Force& getMaxForce() const;
        [[nodiscard]] const Force& getMaxForce(int) const;

        [[nodiscard]]
        const std::vector<Force3D>& getForces() const;
        [[nodiscard]]
        const std::vector<Force3D>& getForces(int) const;

        [[nodiscard]]
        const std::vector<Pos3D>& getPositions() const;
        [[nodiscard]]
        const std::vector<Pos3D>& getPositions(int) const;

        [[nodiscard]] opt::Convergence& getConvergence();
        [[nodiscard]] opt::Convergence  getConvergence() const;

       protected:
        [[nodiscard]]
        molsys::SimulationBox& _getSimulationBox() const;
    };

}   // namespace opt

#endif   // OPT__OPTIMIZER__OPTIMIZER_HPP
