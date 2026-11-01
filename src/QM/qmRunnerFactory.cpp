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

#include "qmRunnerFactory.hpp"

#include <memory>
#include <utility>

#include "enums/qm.hpp"
#include "external/dftbplusRunner.hpp"
#include "external/pyscfRunner.hpp"
#include "external/turbomoleRunner.hpp"
#include "generalSettings.hpp"
#include "qmSettings.hpp"

#ifdef WITH_ASE
#include "ase/aseDftbRunner.hpp"
#include "ase/aseFennolRunner.hpp"
#include "ase/aseMaceRunner.hpp"
#include "ase/aseXtbRunner.hpp"
#else
#include "exceptions.hpp"
#endif

namespace QM
{

    /**
     * @brief Create a QM runner based on the specified method
     *
     * @param settings The QMSettings object containing the relevant settings
     * for configuring the QM runner
     * @return std::unique_ptr<QMRunner> Shared pointer to the created QM
     * runner
     * @throws InputFileException if the method is not supported
     */
    std::unique_ptr<QMRunner> QMRunnerFactory::createQMRunner(
        const QMSettings& settings
    )
    {
        using enum QMMethod;

        if (!settings.isQMMethodSet())
        {
            throw exc::InputFileException(
                "A QM based jobtype was requested but no valid external "
                "program via \"qm_prog\" provided"
            );
        }

        switch (settings.getQMMethod())
        {
            case DFTBPLUS: return std::make_unique<DFTBPlusRunner>(settings);

            case ASE_DFTBPLUS: return _createAseDftbRunner(settings);

            case ASE_XTB: return _createAseXtbRunner(settings);

            case PYSCF: return std::make_unique<PySCFRunner>(settings);

            case TURBOMOLE: return std::make_unique<TurbomoleRunner>(settings);

            case MACE: return _createAseMaceRunner(settings);

            case FENNOL: return _createAseFennolRunner(settings);
        }

        std::unreachable();
    }

    /**
     * @brief Create a MACE QM runner
     *
     * @return std::unique_ptr<QMRunner> Shared pointer to the MACE runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::unique_ptr<QMRunner> QMRunnerFactory::_createAseMaceRunner(
        const QMSettings& settings
    )
    {
#ifdef WITH_ASE
        const auto modelType =
            MaceModelTypeMeta::toString(settings.getMaceModelType());
        const auto modelPath = settings.getMaceModelPath();
        const auto useDFTD   = settings.useDispersionCorr();
        const auto fpType =
            settings::GeneralSettings::getFloatingPointPybindString();
        const auto useCueq = settings.getMaceMode() == MaceMode::FAST;

        auto maceModel = MaceModelMeta::toString(settings.getMaceModel());

        if (!modelPath.empty())
            maceModel = modelPath;

        return std::make_unique<AseMaceRunner>(
            settings,
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
     * @return std::unique_ptr<QMRunner> Shared pointer to the ASE DFTB+
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::unique_ptr<QMRunner> QMRunnerFactory::_createAseDftbRunner(
        const QMSettings& settings
    )
    {
#ifdef WITH_ASE
        const auto slakosPath    = settings.getSlakosPath();
        const auto useThirdOrder = settings.useThirdOrderDftb();
        const auto hubbardDerivs = settings.getHubbardDerivs();
        const auto dispersion    = settings.useDispersionCorr();

        return std::make_unique<AseDftbRunner>(
            settings,
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
     * @return std::unique_ptr<QMRunner> Shared pointer to the ASE xTB
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::unique_ptr<QMRunner> QMRunnerFactory::_createAseXtbRunner(
        const QMSettings& settings
    )
    {
#ifdef WITH_ASE
        const auto xtbMethod = XtbMethodMeta::toString(settings.getXtbMethod());

        return std::make_unique<AseXtbRunner>(settings, xtbMethod);
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
     * @return std::unique_ptr<QMRunner> Shared pointer to the ASE FeNNol
     * runner
     * @throws exc::CompileTimeException if ASE was not enabled at compile time
     */
    std::unique_ptr<QMRunner> QMRunnerFactory::_createAseFennolRunner(
        const QMSettings& settings
    )
    {
#ifdef WITH_ASE
        using enum FPType;

        const auto modelPath        = settings.getFennolModelPath();
        const auto gpuPreprocessing = settings.useGPUPreprocessing();
        const bool useFloat64 =
            settings::GeneralSettings::getFloatingPointType() == DOUBLE;

        return std::make_unique<AseFennolRunner>(
            settings,
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

}   // namespace QM
