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

#include "dftbplusRunner.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "generalSettings.hpp"
#include "pyscfRunner.hpp"
#include "qmSettings.hpp"
#include "turbomoleRunner.hpp"

#ifdef WITH_ASE
#include "aseDftbRunner.hpp"
#include "aseFennolRunner.hpp"
#include "aseMaceRunner.hpp"
#include "aseXtbRunner.hpp"
#endif

namespace engine
{

    /**
     * @brief Create a QM runner based on the specified method
     *
     * @param settings The QMSettings object containing the configuration for
     * the QM runner
     * @return std::shared_ptr<QM::QMRunner> Shared pointer to the created QM
     * runner
     * @throws InputFileException if the method is not supported
     */
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createQMRunner(
        const QMSettings   &qmSettings,
        const MaceSettings &maceSettings
    )
    {
        using enum QMMethod;

        switch (qmSettings.getQMMethod())
        {
            case DFTBPLUS:
                return std::make_shared<QM::DFTBPlusRunner>(qmSettings);

            case ASE_DFTBPLUS: return createAseDftbRunner(qmSettings);

            case ASE_XTB: return createAseXtbRunner(qmSettings);

            case PYSCF: return std::make_shared<QM::PySCFRunner>(qmSettings);

            case TURBOMOLE:
                return std::make_shared<QM::TurbomoleRunner>(qmSettings);

            case MACE: return createAseMaceRunner(qmSettings, maceSettings);

            case FENNOL: return createAseFennolRunner(qmSettings);

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
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseMaceRunner(
        const QMSettings   &qmSettings,
        const MaceSettings &maceSettings
    )
    {
#ifdef WITH_ASE
        return std::make_shared<QM::AseMaceRunner>(qmSettings, maceSettings);
#else
        // to silence unused parameter warnings when ASE is not enabled
        (void) qmSettings;
        (void) maceSettings;
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
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseDftbRunner(
        const QMSettings &settings
    )
    {
#ifdef WITH_ASE
        const auto slakosPath    = QMSettings::getSlakosPath();
        const auto useThirdOrder = QMSettings::useThirdOrderDftb();
        const auto hubbardDerivs = QMSettings::getHubbardDerivs();
        const auto dispersion    = QMSettings::useDispersionCorr();

        return std::make_shared<QM::AseDftbRunner>(
            settings,
            slakosPath,
            useThirdOrder,
            hubbardDerivs,
            dispersion
        );
#else
        // to silence unused parameter warnings when ASE is not enabled
        (void) settings;
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
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseXtbRunner(
        const QMSettings &settings
    )
    {
#ifdef WITH_ASE
        const auto xtbMethod =
            XtbMethodMeta::toString(QMSettings::getXtbMethod());

        return std::make_shared<QM::AseXtbRunner>(settings, xtbMethod);
#else
        // to silence unused parameter warnings when ASE is not enabled
        (void) settings;
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
    std::shared_ptr<QM::QMRunner> QMRunnerManager::createAseFennolRunner(
        const QMSettings &settings
    )
    {
#ifdef WITH_ASE
        using enum FPType;

        const auto modelPath        = QMSettings::getFennolModelPath();
        const auto gpuPreprocessing = QMSettings::useGPUPreprocessing();
        const bool useFloat64 =
            settings::GeneralSettings::getFloatingPointType() == DOUBLE;

        return std::make_shared<QM::AseFennolRunner>(
            settings,
            modelPath,
            gpuPreprocessing,
            useFloat64
        );
#else
        // to silence unused parameter warnings when ASE is not enabled
        (void) settings;
        throw exc::CompileTimeException(
            "The ASE FeNNol QM method was requested but ASE was not enabled at "
            "compile time. Please recompile with ASE enabled to use the ASE "
            "FeNNol "
            "type QM method using: -DBUILD_WITH_ASE=ON"
        );
#endif
    }

}   // namespace engine
