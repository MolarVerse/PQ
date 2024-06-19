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

#include <iostream>   // for std::cout
#include <memory>     // for std::shared_ptr

#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox

using namespace opt;
using namespace physicalData;
using namespace simulationBox;
using namespace settings;

/**
 * @brief Construct a new Optimizer object
 *
 * @param nEpochs
 * @param initialLearningRate
 */
Optimizer::Optimizer(const size_t nEpochs)
    : Optimizer(nEpochs, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
{
}

/**
 * @brief Construct a new Optimizer object
 *
 */
Optimizer::Optimizer(
    const size_t nEpochs,
    const double relEnergyConv,
    const double relMaxForceConv,
    const double relRMSForceConv,
    const double absEnergyConv,
    const double absMaxForceConv,
    const double absRMSForceConv
)
    : _nEpochs(nEpochs),
      _relEnergyConv(relEnergyConv),
      _relMaxForceConv(relMaxForceConv),
      _relRMSForceConv(relRMSForceConv),
      _absEnergyConv(absEnergyConv),
      _absMaxForceConv(absMaxForceConv),
      _absRMSForceConv(absRMSForceConv)
{
    _energyConvStrategy = ConvStrategy::RIGOROUS;
    _forceConvStrategy  = ConvStrategy::RIGOROUS;
}

void Optimizer::updateHistory()
{
    _energyHistory.push(_physicalData->getTotalEnergy());
    _forceHistory.push(_simulationBox->getForces());
    _positionHistory.push(_simulationBox->getPositions());

    if (_energyHistory.size() > maxHistoryLength())
    {
        _energyHistory.pop();
        _forceHistory.pop();
        _positionHistory.pop();
    }
}

/**
 * @brief check if the optimizer has converged
 *
 * @return true/false if the optimizer has converged
 */
bool Optimizer::hasConverged() const
{
    const auto energyOld = _physicalDataOld->getTotalEnergy();
    const auto energyNew = _physicalData->getTotalEnergy();

    const auto rmsForceOld = _simulationBox->calculateRMSForceOld();
    const auto rmsForceNew = _simulationBox->calculateRMSForce();

    const auto maxForceOld = _simulationBox->calculateMaxForceOld();
    const auto maxForceNew = _simulationBox->calculateMaxForce();

    auto converged = true;

    if (_enableEnergyConv)
    {
        const auto absDeviation = std::abs(energyOld - energyNew);
        const auto relDeviation = absDeviation / std::abs(energyOld);

        const auto isAbsConv = absDeviation < _absEnergyConv;
        const auto isRelConv = relDeviation < _relEnergyConv;

        const auto isEnergyConverged =
            hasPropertyConv(isAbsConv, isRelConv, _energyConvStrategy);

        converged = converged && isEnergyConverged;

        std::cout << "energyOld: " << energyOld << std::endl;
        std::cout << "energyNew: " << energyNew << std::endl;
        std::cout << "absDeviation: " << absDeviation << std::endl;
        std::cout << "relDeviation: " << relDeviation << std::endl << std::endl;
    }

    if (_enableMaxForceConv)
    {
        const auto absDeviation = std::abs(maxForceOld - maxForceNew);
        const auto relDeviation = absDeviation / std::abs(maxForceOld);

        const auto isAbsConv = absDeviation < _absMaxForceConv;
        const auto isRelConv = relDeviation < _relMaxForceConv;

        const auto isMaxForceConverged =
            hasPropertyConv(isAbsConv, isRelConv, _forceConvStrategy);

        converged = converged && isMaxForceConverged;

        std::cout << "maxForceOld: " << maxForceOld << std::endl;
        std::cout << "maxForceNew: " << maxForceNew << std::endl;
        std::cout << "absDeviation: " << absDeviation << std::endl;
        std::cout << "relDeviation: " << relDeviation << std::endl << std::endl;
    }

    if (_enableRMSForceConv)
    {
        const auto absDeviation = std::abs(rmsForceOld - rmsForceNew);
        const auto relDeviation = absDeviation / std::abs(rmsForceOld);

        const auto isAbsConv = absDeviation < _absRMSForceConv;
        const auto isRelConv = relDeviation < _relRMSForceConv;

        const auto isRMSForceConverged =
            hasPropertyConv(isAbsConv, isRelConv, _forceConvStrategy);

        converged = converged && isRMSForceConverged;

        std::cout << "rmsForceOld: " << rmsForceOld << std::endl;
        std::cout << "rmsForceNew: " << rmsForceNew << std::endl;
        std::cout << "absDeviation: " << absDeviation << std::endl;
        std::cout << "relDeviation: " << relDeviation << std::endl << std::endl;
    }

    return converged;
}

/**
 * @brief check if a property of the optimizer has converged
 * according to the convergence strategy set
 *
 * @param absConv
 * @param relConv
 * @param convStrategy
 * @return true
 * @return false
 */
bool Optimizer::hasPropertyConv(
    const bool         absConv,
    const bool         relConv,
    const ConvStrategy convStrategy
) const
{
    if (convStrategy == ConvStrategy::RIGOROUS)
        return absConv && relConv;
    else if (convStrategy == ConvStrategy::LOOSE)
        return absConv || relConv;
    else if (convStrategy == ConvStrategy::ABSOLUTE)
        return absConv;
    else if (convStrategy == ConvStrategy::RELATIVE)
        return relConv;
    else
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
    const std::shared_ptr<SimulationBox> simulationBox
)
{
    _simulationBox = simulationBox;
}

/**
 * @brief set physical data shared pointer
 *
 * @param physicalData
 */
void Optimizer::setPhysicalData(const std::shared_ptr<PhysicalData> physicalData
)
{
    _physicalData = physicalData;
}

/**
 * @brief set old physical data shared pointer
 *
 * @param physicalData
 */
void Optimizer::setPhysicalDataOld(
    const std::shared_ptr<PhysicalData> physicalData
)
{
    _physicalDataOld = physicalData;
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
void Optimizer::setEnergyConvStrategy(const ConvStrategy energyConvStrategy)
{
    _energyConvStrategy = energyConvStrategy;
}

/**
 * @brief set force convergence strategy
 *
 * @param forceConvStrategy
 */
void Optimizer::setForceConvStrategy(const ConvStrategy forceConvStrategy)
{
    _forceConvStrategy = forceConvStrategy;
}

/***************************
 *                         *
 * standard getter methods *
 *                         *
 ***************************/

/**
 * @brief get the number of epochs
 *
 * @return size_t
 */
size_t Optimizer::getNEpochs() const { return _nEpochs; }
