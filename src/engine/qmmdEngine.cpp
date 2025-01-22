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
#include "aseDftbRunner.hpp"    // for aseDftbRunner
#include "fairchemRunner.hpp"   // for FairchemRunner
#include "maceRunner.hpp"       // for MaceRunner
#endif

using engine::QMMDEngine;
using namespace settings;
using namespace customException;
using namespace QM;
using std::make_shared;

/**
 * @brief calculate QM forces
 *
 */
void QMMDEngine::calculateForces()
{
    _qmRunner->run(*_simulationBox, *_physicalData);
}

/**
 * @brief Set the QMRunner object based on the QM method.
 *
 * @param method
 */
void QMMDEngine::setQMRunner(const QMMethod method)
{
    using enum QMMethod;

    if (method == DFTBPLUS)
        _qmRunner = make_shared<DFTBPlusRunner>();

    else if (method == ASEDFTBPLUS)
        setAseDftbRunner();

    else if (method == PYSCF)
        _qmRunner = make_shared<PySCFRunner>();

    else if (method == TURBOMOLE)
        _qmRunner = make_shared<TurbomoleRunner>();

    else if (method == MACE)
        setMaceQMRunner();

    else if (method == FAIRCHEM)
        setFairchemRunner();

    else
        throw InputFileException(
            "A qm based jobtype was requested but no external "
            "program via \"qm_prog\" provided"
        );
}

/**
 * @brief sets the QMRunner object for mace type qm methods.
 *
 * @throws InputFileException if ASE was not enabled at compile
 * time.
 *
 */
void QMMDEngine::setMaceQMRunner()
{
#ifdef WITH_ASE
    const auto modelType = string(QMSettings::getMaceModelType());
    const auto modelPath = QMSettings::getMaceModelPath();
    const auto useDFTD   = QMSettings::useDispersionCorr();
    const auto fpType    = Settings::getFloatingPointPybindString();

    auto maceModel = string(QMSettings::getMaceModelSize());

    if (!modelPath.empty())
        maceModel = modelPath;

    _qmRunner = make_shared<MaceRunner>(modelType, maceModel, fpType, useDFTD);
#else
    throw CompileTimeException(
        "A mace type qm method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use mace type "
        "qm methods using: -DBUILD_WITH_ASE=ON"
    );
#endif
}
/**
 * @brief sets the QMRunner object for FAIR-Chem type qm methods.
 *
 * @throws py::error_already_set if the import of the FAIR-Chem module fails
 */
void QMMDEngine::setFairchemRunner()
{
#ifdef WITH_ASE
    const auto modelName = QMSettings::getFairchemModelName();

    _qmRunner = make_shared<FairchemRunner>(modelName);
#else
    throw CompileTimeException(
        "The FAIR-Chem qm method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use mace type "
        "qm methods using: -DBUILD_WITH_ASE=ON"
    );
#endif
}
/**
 * @brief sets the QMRunner object for ase dftbplus type qm methods.
 *
 * @throws InputFileException if ASE was not enabled at compile
 * time.
 *
 */
void QMMDEngine::setAseDftbRunner()
{
#ifdef WITH_ASE
    const auto slakosPath    = QMSettings::getSlakosPath();
    const auto useThirdOrder = QMSettings::useThirdOrderDftb();
    const auto hubbardDerivs = QMSettings::getHubbardDerivs();
    const auto dispersion    = QMSettings::useDispersionCorr();

    _qmRunner = make_shared<AseDftbRunner>(
        slakosPath,
        useThirdOrder,
        hubbardDerivs,
        dispersion
    );
#else
    throw CompileTimeException(
        "The ASE DFTB+ qm method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use mace type "
        "qm methods using: -DBUILD_WITH_ASE=ON"
    );
#endif
}

/**
 * @brief Get the QMRunner object.
 *
 * @return QMRunner *
 */
QMRunner* QMMDEngine::getQMRunner() const { return _qmRunner.get(); }