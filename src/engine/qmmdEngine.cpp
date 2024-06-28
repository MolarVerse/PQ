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

#include "qmmdEngine.hpp"

#include "dftbplusRunner.hpp"    // for DFTBPlusRunner
#include "integrator.hpp"        // for Integrator
#include "manostat.hpp"          // for Manostat
#include "physicalData.hpp"      // for PhysicalData
#include "pyscfRunner.hpp"       // for PySCFRunner
#include "resetKinetics.hpp"     // for ResetKinetics
#include "settings.hpp"          // for Settings
#include "thermostat.hpp"        // for Thermostat
#include "timingsSettings.hpp"   // for TimingsSettings
#include "turbomoleRunner.hpp"   // for TurbomoleRunner

#ifdef WITH_ASE
#include "maceRunner.hpp"   // for MaceRunner
#endif

using engine::QMMDEngine;
using namespace settings;

/**
 * @brief Takes one step in a QM MD simulation.
 *
 * @details The step is taken in the following order:
 * - First step of the integrator
 * - Apply thermostat half step
 * - Run QM calculations
 * - Apply thermostat on forces
 * - Second step of the integrator
 * - Apply thermostat
 * - Calculate kinetic energy and momentum
 * - Apply manostat
 * - Reset temperature and momentum
 *
 */
void QMMDEngine::takeStep()
{
    _thermostat->applyThermostatHalfStep(*_simulationBox, *_physicalData);

    _integrator->firstStep(*_simulationBox);

    _constraints->applyShake(*_simulationBox);

    _qmRunner->run(*_simulationBox, *_physicalData);

    _constraints->applyDistanceConstraints(
        *_simulationBox,
        *_physicalData,
        calculateTotalSimulationTime()
    );

    _constraints->calculateConstraintBondRefs(*_simulationBox);

    _thermostat->applyThermostatOnForces(*_simulationBox);

    _integrator->secondStep(*_simulationBox);

    _constraints->applyRattle(*_simulationBox);

    _thermostat->applyThermostat(*_simulationBox, *_physicalData);

    _physicalData->calculateKinetics(*_simulationBox);

    _manostat->applyManostat(*_simulationBox, *_physicalData);

    _resetKinetics.reset(_step, *_physicalData, *_simulationBox);

    _thermostat->applyTemperatureRamping();

    _physicalData->setNumberOfQMAtoms(_simulationBox->getNumberOfQMAtoms());
}

/**
 * @brief Set the QMRunner object based on the QM method.
 *
 * @param method
 */
void QMMDEngine::setQMRunner(const QMMethod method)
{
    if (method == QMMethod::DFTBPLUS)
        _qmRunner = std::make_shared<QM::DFTBPlusRunner>();

    else if (method == QMMethod::PYSCF)
        _qmRunner = std::make_shared<QM::PySCFRunner>();

    else if (method == QMMethod::TURBOMOLE)
        _qmRunner = std::make_shared<QM::TurbomoleRunner>();

#ifdef WITH_ASE
    else if (method == QMMethod::MACE)
    {
        const auto maceModelType = string(QMSettings::getMaceModelType());
        auto       maceModel     = string(QMSettings::getMaceModelSize());
        const auto modelPath     = QMSettings::getMaceModelPath();
        const auto fpType        = Settings::getFloatingPointPybindString();

        if (!modelPath.empty())
            maceModel = modelPath;

        _qmRunner = std::make_shared<QM::MaceRunner>(
            maceModelType,
            maceModel,
            fpType,
            false
        );
    }
#endif

    else
        throw customException::InputFileException(
            "A qm based jobtype was requested but no external "
            "program via \"qm_prog\" provided"
        );
}

/**
 * @brief Get the QMRunner object.
 *
 * @return QM::QMRunner *
 */
QM::QMRunner* QMMDEngine::getQMRunner() const { return _qmRunner.get(); }