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

#include "exceptions.hpp"      // for OptException
#include "physicalData.hpp"    // for PhysicalData
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for max, rms

using namespace opt;
using namespace physicalData;
using namespace simulationBox;
using namespace settings;
using namespace customException;

/**
 * @brief Construct a new Optimizer object
 *
 * @param nEpochs
 * @param initialLearningRate
 */
Optimizer::Optimizer(const size_t nEpochs) : _nEpochs(nEpochs) {}

/**
 * @brief update the optimizer history
 *
 * @param learningRate
 */
void Optimizer::updateHistory()
{
    _energyHistory.push_back(_physicalData->getTotalEnergy());
    _forceHistory.push_back(_simulationBox->getForces());
    _positionHistory.push_back(_simulationBox->getPositions());

    const auto rmsForce = rms(_simulationBox->getForces());
    const auto maxForce = max(_simulationBox->getForces());

    _rmsForceHistory.push_back(rmsForce);
    _maxForceHistory.push_back(maxForce);

    if (_energyHistory.size() > maxHistoryLength())
    {
        _energyHistory.pop_front();
        _forceHistory.pop_front();
        _positionHistory.pop_front();
        _rmsForceHistory.pop_front();
        _maxForceHistory.pop_front();
    }
}

/**
 * @brief check if the optimizer has converged
 *
 * @return true/false if the optimizer has converged
 */
bool Optimizer::hasConverged()
{
    const auto energyOld = getEnergy(-2);
    const auto energyNew = getEnergy(-1);

    const auto rmsForceNew = getRMSForce(-1);
    const auto maxForceNew = getMaxForce(-1);

    _convergence.calcEnergyConvergence(energyOld, energyNew);
    _convergence.calcForceConvergence(maxForceNew, rmsForceNew);

    return _convergence.checkConvergence();
}

/***************************
 *                         *
 * standard setter methods *
 *                         *
 ***************************/

/**
 * @brief set convergence member
 *
 * @param convergence
 */
void Optimizer::setConvergence(const Convergence convergence)
{
    _convergence = convergence;
}

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
void Optimizer::setPhysicalData(
    const std::shared_ptr<PhysicalData> physicalData
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

/**
 * @brief get history index
 *
 * @return size_t
 */
size_t Optimizer::getHistoryIndex(const int offset) const
{
    if (offset >= 0)
        throw OptException(
            "Offset must be negative to access history in the past"
        );

    const auto size  = int(_energyHistory.size());
    const auto index = size_t(size + offset);

    return index;
}

/**
 * @brief get the last energy in the history
 *
 * @return double
 */
double Optimizer::getEnergy() const { return _energyHistory.back(); }

/**
 * @brief get the energy in history with negative index offset
 *
 * @param offset
 *
 */
double Optimizer::getEnergy(const int offset) const
{
    const auto index = getHistoryIndex(offset);

    return _energyHistory[index];
}

/**
 * @brief get the last RMS force in the history
 *
 * @return double
 */
double Optimizer::getRMSForce() const { return _rmsForceHistory.back(); }

/**
 * @brief get the RMS force in history with negative index offset
 *
 * @param offset
 *
 */
double Optimizer::getRMSForce(const int offset) const
{
    const auto index = getHistoryIndex(offset);

    return _rmsForceHistory[index];
}

/**
 * @brief get the last max force in the history
 *
 * @return double
 */
double Optimizer::getMaxForce() const { return _maxForceHistory.back(); }

/**
 * @brief get the max force in history with negative index offset
 *
 * @param offset
 *
 */
double Optimizer::getMaxForce(const int offset) const
{
    const auto index = getHistoryIndex(offset);

    return _maxForceHistory[index];
}

/**
 * @brief get the last force in the history
 *
 * @return std::vector<pq::Vec3D>
 */
std::vector<linearAlgebra::Vec3D> Optimizer::getForces() const
{
    return _forceHistory.back();
}

/**
 * @brief get the force in history with negative index offset
 *
 * @param offset
 *
 */
std::vector<linearAlgebra::Vec3D> Optimizer::getForces(const int offset) const
{
    const auto index = getHistoryIndex(offset);

    return _forceHistory[index];
}

/**
 * @brief get the last position in the history
 *
 * @return std::vector<pq::Vec3D>
 */
std::vector<linearAlgebra::Vec3D> Optimizer::getPositions() const
{
    return _positionHistory.back();
}

/**
 * @brief get the position in history with negative index offset
 *
 * @param offset
 *
 */
std::vector<linearAlgebra::Vec3D> Optimizer::getPositions(
    const int offset
) const
{
    const auto index = getHistoryIndex(offset);

    return _positionHistory[index];
}

/**
 * @brief get the convergence member
 *
 * @return opt::Convergence
 */
Convergence Optimizer::getConvergence() const { return _convergence; }

/**
 * @brief get the convergence member
 *
 * @return opt::Convergence
 */
Convergence &Optimizer::getConvergence() { return _convergence; }
