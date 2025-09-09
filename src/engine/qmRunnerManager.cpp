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

#include "qmRunnerManager.hpp"

#include "dftbplusRunner.hpp"    // for DFTBPlusRunner
#include "exceptions.hpp"        // for InputFileException, CompileTimeException
#include "pyscfRunner.hpp"       // for PySCFRunner
#include "qmSettings.hpp"        // for QMSettings
#include "settings.hpp"          // for Settings
#include "turbomoleRunner.hpp"   // for TurbomoleRunner

#ifdef WITH_ASE
#include "aseDftbRunner.hpp"   // for AseDftbRunner
#include "aseXtbRunner.hpp"    // for AseXtbRunner
#include "maceRunner.hpp"      // for MaceRunner
#endif

using namespace engine;
using namespace settings;
using namespace customException;
using namespace QM;

using std::make_shared;
using std::shared_ptr;

/**
 * @brief Create a QM runner based on the specified method
 *
 * @param method The QM method to use
 * @return shared_ptr<QMRunner> Shared pointer to the created QM runner
 * @throws InputFileException if the method is not supported
 */
shared_ptr<QMRunner> QMRunnerManager::createQMRunner(const QMMethod method)
{
    using enum QMMethod;

    switch (method)
    {
        case DFTBPLUS: return make_shared<DFTBPlusRunner>();

        case ASEDFTBPLUS: return createAseDftbRunner();

        case ASEXTB: return createAseXtbRunner();

        case PYSCF: return make_shared<PySCFRunner>();

        case TURBOMOLE: return make_shared<TurbomoleRunner>();

        case MACE: return createMaceQMRunner();

        default:
            throw InputFileException(
                "A QM based jobtype was requested but no valid external "
                "program via \"qm_prog\" provided"
            );
    }
}

/**
 * @brief Create a MACE QM runner
 *
 * @return shared_ptr<QMRunner> Shared pointer to the MACE runner
 * @throws CompileTimeException if ASE was not enabled at compile time
 */
shared_ptr<QMRunner> QMRunnerManager::createMaceQMRunner()
{
#ifdef WITH_ASE
    const auto modelType = string(QMSettings::getMaceModelType());
    const auto modelPath = QMSettings::getMaceModelPath();
    const auto useDFTD   = QMSettings::useDispersionCorr();
    const auto fpType    = Settings::getFloatingPointPybindString();

    auto maceModel = string(QMSettings::getMaceModelSize());

    if (!modelPath.empty())
        maceModel = modelPath;

    return make_shared<QM::MaceRunner>(modelType, maceModel, fpType, useDFTD);
#else
    throw CompileTimeException(
        "A MACE type QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use MACE type "
        "QM methods using: -DBUILD_WITH_ASE=ON"
    );
#endif
}

/**
 * @brief Create an ASE DFTB+ QM runner
 *
 * @return shared_ptr<QMRunner> Shared pointer to the ASE DFTB+ runner
 * @throws CompileTimeException if ASE was not enabled at compile time
 */
shared_ptr<QMRunner> QMRunnerManager::createAseDftbRunner()
{
#ifdef WITH_ASE
    const auto slakosPath    = QMSettings::getSlakosPath();
    const auto useThirdOrder = QMSettings::useThirdOrderDftb();
    const auto hubbardDerivs = QMSettings::getHubbardDerivs();
    const auto dispersion    = QMSettings::useDispersionCorr();

    return make_shared<QM::AseDftbRunner>(
        slakosPath,
        useThirdOrder,
        hubbardDerivs,
        dispersion
    );
#else
    throw CompileTimeException(
        "The ASE DFTB+ QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use ASE DFTB+ type "
        "QM methods using: -DBUILD_WITH_ASE=ON"
    );
#endif
}

/**
 * @brief Create an ASE xTB QM runner
 *
 * @return shared_ptr<QMRunner> Shared pointer to the ASE xTB runner
 * @throws CompileTimeException if ASE was not enabled at compile time
 */
shared_ptr<QMRunner> QMRunnerManager::createAseXtbRunner()
{
#ifdef WITH_ASE
    const auto xtbMethod = string(QMSettings::getXtbMethod());

    return make_shared<QM::AseXtbRunner>(xtbMethod);
#else
    throw CompileTimeException(
        "The ASE xTB QM method was requested but ASE was not enabled at "
        "compile time. Please recompile with ASE enabled to use the ASE xTB "
        "type QM method using: -DBUILD_WITH_ASE=ON"
    );
#endif
}
