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
namespace engine
{
    class OptEngine;   // forward declaration

}   // namespace engine

namespace simulationBox
{
    class SimulationBox;   // forward declaration

}   // namespace simulationBox

namespace opt
{
    using SharedSimulationBox = std::shared_ptr<simulationBox::SimulationBox>;

    /**
     * @class Optimizer
     *
     * @brief Base class for all optimizers
     *
     */
    class Optimizer
    {
       protected:
        size_t _nEpochs;

        SharedSimulationBox _simulationBox;

        bool _enableEnergyConv;
        bool _enableMaxForceConv;
        bool _enableRMSForceConv;

        double _relEnergyConv;
        double _relMaxForceConv;
        double _relRMSForceConv;

        double _absEnergyConv;
        double _absMaxForceConv;
        double _absRMSForceConv;

        settings::ConvStrategy _energyConvStrategy;
        settings::ConvStrategy _forceConvStrategy;

       public:
        explicit Optimizer(const size_t);

        Optimizer()          = default;
        virtual ~Optimizer() = default;

        virtual void update(const double learningRate) = 0;

        /***************************
         * standard setter methods *
         ***************************/

        void setSimulationBox(const SharedSimulationBox);

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
    };

}   // namespace opt

#endif   // _OPTIMIZER_HPP_
