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

#include <memory>
#include <utility>

#include "dftbplusRunner.hpp"   // for DFTBPlusRunner
#include "enums/qm.hpp"
#include "exceptions.hpp"   // for InputFileException, exc::CompileTimeException
#include "pyscfRunner.hpp"       // for PySCFRunner
#include "qmSettings.hpp"        // for settings::QMSettings
#include "settings.hpp"          // for Settings
#include "turbomoleRunner.hpp"   // for TurbomoleRunner

#ifdef WITH_ASE
#include "aseDftbRunner.hpp"     // for AseDftbRunner
#include "aseFennolRunner.hpp"   // for AseFennolRunner
#include "aseMaceRunner.hpp"     // for AseMaceRunner
#include "aseXtbRunner.hpp"      // for AseXtbRunner
#endif

namespace engine
{

    /**
     * @brief Create a QM runner based on the specified method
     *
     * @param method The QM method to use
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the created QM
     * runner
     * @throws InputFileException if the method is not supported
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createQMRunner(
        QMMethod method
    )
    {
        using enum QMMethod;

        switch (method)
        {
            case DFTBPLUS: return std::make_shared<QM::DFTBPlusRunner>();

            case ASE_DFTBPLUS: return createAseDftbRunner();

            case ASE_XTB: return createAseXtbRunner();

            case PYSCF: return std::make_shared<QM::PySCFRunner>();

            case TURBOMOLE: return std::make_shared<QM::TurbomoleRunner>();

            case MACE: return createAseMaceRunner();

            case FENNOL: return createAseFennolRunner();

            case NONE:
                throw exc::InputFileException(
                    "A QM based jobtype was requested but no valid external "
                    "program via \"qm_prog\" provided"
                );
        }

        std::unreachable();
    }

    /**
     * @brief Create a MACE QM runner
     *
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the MACE runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseMaceRunner()
    {
#ifdef WITH_ASE
        const auto modelType = MaceModelTypeMeta::toString(
            settings::QMSettings::getMaceModelType()
        );
        const auto modelPath = settings::QMSettings::getMaceModelPath();
        const auto useDFTD   = settings::QMSettings::useDispersionCorr();
        const auto fpType = settings::Settings::getFloatingPointPybindString();
        const auto useCueq =
            settings::QMSettings::getMaceMode() == MaceMode::FAST;

        auto maceModel =
            MaceModelMeta::toString(settings::QMSettings::getMaceModel());

        if (!modelPath.empty())
            maceModel = modelPath;

        return std::make_shared<QM::AseMaceRunner>(
            modelType,
            maceModel,
            fpType,
            useDFTD,
            useCueq
        );
#else
        throw exc::CompileTimeException(
            "A MACE type QM method was requested but ASE was not enabled at "
            "compile time. Please recompile with ASE enabled to use MACE type "
            "QM methods using: -DBUILD_WITH_ASE=ON"
        );
#endif
    }

    /**
     * @brief Create an ASE DFTB+ QM runner
     *
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the ASE DFTB+
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseDftbRunner()
    {
#ifdef WITH_ASE
        const auto slakosPath    = settings::QMSettings::getSlakosPath();
        const auto useThirdOrder = settings::QMSettings::useThirdOrderDftb();
        const auto hubbardDerivs = settings::QMSettings::getHubbardDerivs();
        const auto dispersion    = settings::QMSettings::useDispersionCorr();

        return std::make_shared<QM::AseDftbRunner>(
            slakosPath,
            useThirdOrder,
            hubbardDerivs,
            dispersion
        );
#else
        throw exc::CompileTimeException(
            "The ASE DFTB+ QM method was requested but ASE was not enabled at "
            "compile time. Please recompile with ASE enabled to use ASE DFTB+ "
            "type "
            "QM methods using: -DBUILD_WITH_ASE=ON"
        );
#endif
    }

    /**
     * @brief Create an ASE xTB QM runner
     *
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the ASE xTB
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseXtbRunner()
    {
#ifdef WITH_ASE
        const auto xtbMethod =
            XtbMethodMeta::toString(settings::QMSettings::getXtbMethod());

        return std::make_shared<QM::AseXtbRunner>(xtbMethod);
#else
        throw exc::CompileTimeException(
            "The ASE xTB QM method was requested but ASE was not enabled at "
            "compile time. Please recompile with ASE enabled to use the ASE "
            "xTB "
            "type QM method using: -DBUILD_WITH_ASE=ON"
        );
#endif
    }

    /**
     * @brief Create an ASE FeNNol QM runner
     *
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the ASE FeNNol
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseFennolRunner()
    {
#ifdef WITH_ASE
        using enum FPType;

        const auto modelPath = settings::QMSettings::getFennolModelPath();
        const auto gpuPreprocessing =
            settings::QMSettings::useGPUPreprocessing();
        const bool useFloat64 =
            settings::Settings::getFloatingPointType() == DOUBLE;

        return std::make_shared<QM::AseFennolRunner>(
            modelPath,
            gpuPreprocessing,
            useFloat64
        );
#else
        throw exc::CompileTimeException(
            "The ASE FeNNol QM method was requested but ASE was not enabled at "
            "compile time. Please recompile with ASE enabled to use the ASE "
            "FeNNol "
            "type QM method using: -DBUILD_WITH_ASE=ON"
        );
#endif
    }

}   // namespace engine
