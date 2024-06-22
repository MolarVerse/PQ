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
#include <memory>    // for shared_ptr

#include "convergenceSettings.hpp"   // for ConvergenceSettings
#include "typeAliases.hpp"           // for SharedSimulationBox

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

        pq::SharedSimBox       _simulationBox;
        pq::SharedPhysicalData _physicalData;
        pq::SharedPhysicalData _physicalDataOld;

        bool _enableEnergyConv   = true;
        bool _enableMaxForceConv = true;
        bool _enableRMSForceConv = true;

        double _relEnergyConv;
        double _relMaxForceConv;
        double _relRMSForceConv;

        double _absEnergyConv;
        double _absMaxForceConv;
        double _absRMSForceConv;

        settings::ConvStrategy _energyConvStrategy;
        settings::ConvStrategy _forceConvStrategy;

        std::deque<double> _energyHistory;
        std::deque<double> _maxForceHistory;
        std::deque<double> _rmsForceHistory;
        pq::Vec3DVecDeque  _forceHistory;
        pq::Vec3DVecDeque  _positionHistory;

       public:
        explicit Optimizer(const size_t);
        explicit Optimizer(
            const size_t,
            const double,
            const double,
            const double,
            const double,
            const double,
            const double
        );

        Optimizer()          = default;
        virtual ~Optimizer() = default;

        virtual pq::SharedOptimizer clone() const                     = 0;
        virtual void                update(const double learningRate) = 0;
        virtual size_t              maxHistoryLength() const          = 0;

        void updateHistory();

        [[nodiscard]] bool hasConverged() const;
        [[nodiscard]] bool hasPropertyConv(
            const bool,
            const bool,
            const settings::ConvStrategy
        ) const;

        /***************************
         * standard setter methods *
         ***************************/

        void setSimulationBox(const pq::SharedSimBox);
        void setPhysicalData(const pq::SharedPhysicalData);
        void setPhysicalDataOld(const pq::SharedPhysicalData);

        void setEnableEnergyConv(const bool);
        void setEnableMaxForceConv(const bool);
        void setEnableRMSForceConv(const bool);

        void setRelEnergyConv(const double);
        void setRelMaxForceConv(const double);
        void setRelRMSForceConv(const double);

        void setAbsEnergyConv(const double);
        void setAbsMaxForceConv(const double);
        void setAbsRMSForceConv(const double);

        void setEnergyConvStrategy(const settings::ConvStrategy);
        void setForceConvStrategy(const settings::ConvStrategy);

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

        [[nodiscard]] pq::Vec3DVec getForces() const;
        [[nodiscard]] pq::Vec3DVec getForces(const int) const;

        [[nodiscard]] pq::Vec3DVec getPositions() const;
        [[nodiscard]] pq::Vec3DVec getPositions(const int) const;
    };

}   // namespace opt

#endif   // _OPTIMIZER_HPP_
