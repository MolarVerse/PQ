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

#include "optimizer.hpp"

#include <memory>   // for std::shared_ptr

#include "simulationBox.hpp"

using namespace opt;

/**
 * @brief Construct a new Optimizer object
 *
 * @param nEpochs
 * @param initialLearningRate
 */
Optimizer::Optimizer(const size_t nEpochs) : _nEpochs(nEpochs) {}

/**
 * @brief check if the optimizer has converged
 *
 */
bool Optimizer::hasConverged() const
{
    const auto rmsForceOld = _simulationBox->calculateRMSForceOld();
    const auto rmsForceNew = _simulationBox->calculateRMSForce();

    const auto maxForceOld = _simulationBox->calculateMaxForceOld();
    const auto maxForceNew = _simulationBox->calculateMaxForce();

    return false;
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set simulation box shared pointer
 *
 * @param simulationBox
 */
void Optimizer::setSimulationBox(
    const std::shared_ptr<simulationBox::SimulationBox> simulationBox
)
{
    _simulationBox = simulationBox;
}

/**
 * @brief set physical data shared pointer
 *
 * @param physicalData
 */
void Optimizer::setPhysicalData(
    const std::shared_ptr<physicalData::PhysicalData> physicalData
)
{
    _physicalData = physicalData;
}

/**
 * @brief set physical data old shared pointer
 *
 * @param physicalDataOld
 */
void Optimizer::setPhysicalDataOld(
    const std::shared_ptr<physicalData::PhysicalData> physicalDataOld
)
{
    _physicalDataOld = physicalDataOld;
}

/**
 * @brief set energy convergence flag
 *
 * @param enableEnergyConv
 */
void Optimizer::setEnableEnergyConv(const bool enableEnergyConv)
{
    _enableEnergyConv = enableEnergyConv;
}

/**
 * @brief set max force convergence flag
 *
 * @param enableMaxForceConv
 */
void Optimizer::setEnableMaxForceConv(const bool enableMaxForceConv)
{
    _enableMaxForceConv = enableMaxForceConv;
}

/**
 * @brief set RMS force convergence flag
 *
 * @param enableRMSForceConv
 */
void Optimizer::setEnableRMSForceConv(const bool enableRMSForceConv)
{
    _enableRMSForceConv = enableRMSForceConv;
}

/**
 * @brief set relative energy convergence
 *
 * @param relEnergyConv
 */
void Optimizer::setRelEnergyConv(const double relEnergyConv)
{
    _relEnergyConv = relEnergyConv;
}

/**
 * @brief set relative max force convergence
 *
 * @param relMaxForceConv
 */
void Optimizer::setRelMaxForceConv(const double relMaxForceConv)
{
    _relMaxForceConv = relMaxForceConv;
}

/**
 * @brief set relative RMS force convergence
 *
 * @param relRMSForceConv
 */
void Optimizer::setRelRMSForceConv(const double relRMSForceConv)
{
    _relRMSForceConv = relRMSForceConv;
}

/**
 * @brief set absolute energy convergence
 *
 * @param absEnergyConv
 */
void Optimizer::setAbsEnergyConv(const double absEnergyConv)
{
    _absEnergyConv = absEnergyConv;
}

/**
 * @brief set absolute max force convergence
 *
 * @param absMaxForceConv
 */
void Optimizer::setAbsMaxForceConv(const double absMaxForceConv)
{
    _absMaxForceConv = absMaxForceConv;
}

/**
 * @brief set absolute RMS force convergence
 *
 * @param absRMSForceConv
 */
void Optimizer::setAbsRMSForceConv(const double absRMSForceConv)
{
    _absRMSForceConv = absRMSForceConv;
}

/**
 * @brief set energy convergence strategy
 *
 * @param energyConvStrategy
 */
void Optimizer::setEnergyConvStrategy(
    const settings::ConvStrategy energyConvStrategy
)
{
    _energyConvStrategy = energyConvStrategy;
}

/**
 * @brief set force convergence strategy
 *
 * @param forceConvStrategy
 */
void Optimizer::setForceConvStrategy(
    const settings::ConvStrategy forceConvStrategy
)
{
    _forceConvStrategy = forceConvStrategy;
}