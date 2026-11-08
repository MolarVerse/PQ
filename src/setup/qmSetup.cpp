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

#include "qmSetup.hpp"

#include <format>
#include <string_view>

#include "engine.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "externalQMRunner.hpp"
#include "generalSettings.hpp"
#include "potentialSettings.hpp"
#include "qmCapableEngine.hpp"
#include "qmSettings.hpp"
#include "stdoutOutput.hpp"
#include "units.hpp"

namespace setup
{

    /**
     * @brief constructor
     *
     * @param qmCapableEngine The QM-capable engine used to perform QM
     * calculations
     * @param qmSettings The QM settings used to configure the QM runner, which
     * may include method-specific parameters and computational options
     * @param maceSettings The Mace settings used to configure the MACE runner,
     * which may include model-specific parameters and computational options
     * @param fennolSettings The Fennol settings used to configure the Fennol
     * runner, which may include model-specific parameters and computational
     * options
     * @param dftbSettings The DFTB settings used to configure the DFTB runner,
     * which may include method-specific parameters and computational options
     * @param xtbSettings The XTB settings used to configure the XTB runner,
     * which may include method-specific parameters and computational options
     * @param externalQMSettings The External QM settings used to configure the
     * external QM runner, which may include method-specific parameters and
     * computational options
     *
     */
    QMSetup::QMSetup(
        engine::QMCapableEngine  &qmCapableEngine,
        const QMSettings         &qmSettings,
        const MaceSettings       &maceSettings,
        const FennolSettings     &fennolSettings,
        const DftbSettings       &dftbSettings,
        const XtbSettings        &xtbSettings,
        const ExternalQMSettings &externalQMSettings
    )
        : _qmCapableEngine(qmCapableEngine),
          _qmSettings(qmSettings),
          _maceSettings(maceSettings),
          _fennolSettings(fennolSettings),
          _dftbSettings(dftbSettings),
          _xtbSettings(xtbSettings),
          _externalQMSettings(externalQMSettings)
    {
    }

    /**
     * @brief wrapper to build QMSetup object and call setup
     *
     * @param engine
     * TODO(97gamjak): remove references as soon as qmsettings is in place
     */
    void setupQM(engine::Engine &engine)
    {
        if (!settings::GeneralSettings::isQMActivated())
            return;

        out::StdoutOutput::writeSetup("QM runner");
        engine.getLogOutput().writeSetup("QM runner");

        // Try to cast to QMCapableEngine first (covers both QMMDEngine and
        // QMMMMDEngine)
        if (auto *qmCapableEngine =
                dynamic_cast<engine::QMCapableEngine *>(&engine))
        {
            const auto &settings = engine.getSettings();
            QMSetup     qmSetup(
                *qmCapableEngine,
                settings.qmSettings,
                settings.maceSettings,
                settings.fennolSettings,
                settings.dftbSettings,
                settings.xtbSettings,
                settings.externalQMSettings
            );
            qmSetup.setup();
        }
        else
        {
            throw exc::InputFileException(
                "QM setup requested but engine does not support QM capabilities"
            );
        }
    }

    /**
     * @brief setup QM-MD for all subtypes
     *
     */
    void QMSetup::setup()
    {
        setupQMMethod();

        setupCoulombRadiusCutOff();

        setupWriteInfo();
    }

    /**
     * @brief setup the "QM" method of the system
     *
     */
    void QMSetup::setupQMMethod()
    {
        _qmCapableEngine.setQMRunner(
            _qmSettings,
            _maceSettings,
            _fennolSettings,
            _dftbSettings,
            _xtbSettings,
            _externalQMSettings
        );
    }

    /**
     * @brief set coulomb radius cutoff to 0.0 for QM-MD, QM-RPMD
     *
     */
    void QMSetup::setupCoulombRadiusCutOff()
    {
        using enum JobType;

        const auto jobType = settings::GeneralSettings::getJobtype();

        if (jobType == QM_MD || jobType == RING_POLYMER_QM_MD)
            settings::PotentialSettings::setCoulombRadiusCutOff(0.0);
    }

    /**
     * @brief write info about the QM setup
     *
     */
    void QMSetup::setupWriteInfo() const
    {
        using enum QMMethod;

        // Cast QMCapableEngine to Engine to access output methods
        auto &engine    = dynamic_cast<engine::Engine &>(_qmCapableEngine);
        auto &logOutput = engine.getLogOutput();

        const auto qmMethod = _qmSettings->getQMMethod();
        const auto qmRunnerMessage =
            std::format("QM runner: {}", QMMethodMeta::toString(qmMethod));

        logOutput.writeSetupInfo(qmRunnerMessage);
        logOutput.writeEmptyLine();

        if (_qmSettings->isExternalQMRunner())
        {
            const auto qmScript        = _externalQMSettings->getQMScript();
            const auto qmScriptMessage = std::format("QM script: {}", qmScript);

            logOutput.writeSetupInfo(qmScriptMessage);
        }

        if (qmMethod == MACE)
        {
            const auto modelType = _maceSettings->getMaceModelType();
            const auto modelSize = _maceSettings->getMaceModel();
            const auto modelPath = _maceSettings->getMaceModelPath();
            const auto floatingPointStr =
                settings::GeneralSettings::getFloatingPointPybindString();
            const auto        maceMode = _maceSettings->getMaceMode();
            const auto *const useDisp =
                _qmSettings->useDispersionCorr() ? "on" : "off";

            const auto modelTypeMsg = std::format(
                "Model type:            {}",
                MaceModelTypeMeta::toString(modelType)
            );
            const auto modelSizeMsg = std::format(
                "Model size:            {}",
                MaceModelMeta::toString(modelSize)
            );
            const auto modelPathMsg =
                std::format("Model path:            {}", modelPath);
            const auto fpMsg =
                std::format("Floating point type:   {}", floatingPointStr);
            const auto dispCorrMsg =
                std::format("Dispersion Correction: {}", useDisp);
            const auto modeMsg = std::format(
                "Evaluation mode:       {}",
                MaceModeMeta::toString(maceMode)
            );

            logOutput.writeSetupInfo(modelTypeMsg);
            logOutput.writeSetupInfo(modelSizeMsg);

            if (modelSize == MaceModel::CUSTOM)
                logOutput.writeSetupInfo(modelPathMsg);

            logOutput.writeSetupInfo(fpMsg);
            logOutput.writeSetupInfo(dispCorrMsg);
            logOutput.writeSetupInfo(modeMsg);

            if (maceMode == MaceMode::FAST)
            {
                logOutput.writeSetupInfo(
                    std::format(
                        "                       cuequivariance-accelerated "
                        "kernels; "
                        "results are not bit-identical to the e3nn reference "
                        "(use mace_mode = accurate for the exact reference)"
                    )
                );
            }
        }

        if (qmMethod == FENNOL)
        {
            using enum FPType;

            const auto modelPath = _fennolSettings->getFennolModelPath();
            const auto useGPUPreprocessing =
                _fennolSettings->useGPUPreprocessing();
            const bool useFloat64 =
                settings::GeneralSettings::getFloatingPointType() == DOUBLE;

            const auto modelPathMsg =
                std::format("Model path:               {}", modelPath);
            const auto gpuPreprocMsg = std::format(
                "Using GPU pre-processing: {}",
                useGPUPreprocessing
            );
            const auto fpMsg =
                std::format("Using float64:            {}", useFloat64);

            logOutput.writeSetupInfo(modelPathMsg);
            logOutput.writeSetupInfo(gpuPreprocMsg);
            logOutput.writeSetupInfo(fpMsg);
        }

        if (qmMethod == ASE_DFTBPLUS)
        {
            const auto slakosType    = _dftbSettings->getSlakosType();
            const auto slakosPath    = _dftbSettings->getSlakosPath();
            const auto thirdOrder    = _dftbSettings->useThirdOrderDftb();
            const auto hubbardDerivs = _dftbSettings->getHubbardDerivs();
            const auto dispersion    = _qmSettings->useDispersionCorr();

            const auto slakosTypeMsg = std::format(
                "DFTB approach:        {}",
                SlakosTypeMeta::toString(slakosType)
            );
            const auto slakosPathMsg =
                std::format("sk file path:         {}", slakosPath);
            const auto dispersionMsg = std::format(
                "Dispersion is turned: {}",
                dispersion ? "on" : "off"
            );
            const auto thirdOrderMsg = std::format(
                "3rd order is turned:  {}",
                thirdOrder ? "on" : "off"
            );
            const auto threeOBThirdOrderMsg = std::format(
                "3ob approach has been chosen while disabling 3rd order DFTB. "
                "This setup is not recommended."
            );
            const auto threeOBHubbardDerivsMsg = std::format(
                "3ob approach has been chosen while setting custom Hubbard "
                "derivatives. This setup is not recommended."
            );

            logOutput.writeSetupInfo(slakosTypeMsg);
            logOutput.writeSetupInfo(slakosPathMsg);
            logOutput.writeSetupInfo(dispersionMsg);
            logOutput.writeSetupInfo(thirdOrderMsg);
            if (_dftbSettings->isHubbardDerivsSetByUser() &&
                hubbardDerivs.has_value())
            {
                const auto hubbardDerivsMsg = std::format(
                    "Hubbard derivatives:  {}",
                    string(hubbardDerivs.value())
                );
                logOutput.writeSetupInfo(hubbardDerivsMsg);
            }

            // Warnings for non-recommended setups
            if (slakosType == SlakosType::THREEOB && !thirdOrder)
            {
                logOutput.writeEmptyLine();
                logOutput.writeSetupWarning(threeOBThirdOrderMsg);
                out::StdoutOutput::writeSetupWarning(threeOBThirdOrderMsg);
            }

            if (slakosType == SlakosType::THREEOB &&
                _dftbSettings->isHubbardDerivsSetByUser())
            {
                logOutput.writeEmptyLine();
                logOutput.writeSetupWarning(threeOBHubbardDerivsMsg);
                out::StdoutOutput::writeSetupWarning(threeOBHubbardDerivsMsg);
            }
        }

        if (qmMethod == ASE_XTB)
        {
            const auto xtbMethod = _xtbSettings->getXtbMethod();

            const auto xtbMethodMsg = std::format(
                "xTB Parametrization:   {}",
                XtbMethodMeta::toString(xtbMethod)
            );

            logOutput.writeSetupInfo(xtbMethodMsg);
        }

        const auto qm_loop_time_limit = _qmSettings->getQMLoopTimeLimit();

        const auto qmLoopTimeLimitMsg = std::format(
            "QM looptime limit: {}",
            qm_loop_time_limit > 0 ? std::format("{}", qm_loop_time_limit.in(s))
                                   : "unlimited"
        );

        logOutput.writeEmptyLine();
        logOutput.writeSetupInfo(qmLoopTimeLimitMsg);
        logOutput.writeEmptyLine();
    }

}   // namespace setup
