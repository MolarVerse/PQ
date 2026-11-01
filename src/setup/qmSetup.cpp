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
#include "generalSettings.hpp"
#include "potentialSettings.hpp"
#include "qmCapableEngine.hpp"
#include "qmSettings.hpp"
#include "references.hpp"
#include "referencesOutput.hpp"
#include "stdoutOutput.hpp"

namespace setup
{

    /**
     * @brief constructor
     *
     * @param qmCapableEngine
     */
    QMSetup::QMSetup(engine::QMCapableEngine &qmCapableEngine)
        : _qmCapableEngine(qmCapableEngine)
    {
    }

    /**
     * @brief wrapper to build QMSetup object and call setup
     *
     * @param engine
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
            QMSetup qmSetup(*qmCapableEngine);
            qmSetup.setup(engine.getSettings().qmSettings);
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
    void QMSetup::setup(const QMSettings &qmSettings)
    {
        setupQMMethod(qmSettings);

        setupQMMethodAseXtb(qmSettings);

        setupCoulombRadiusCutOff();

        setupWriteInfo(qmSettings);
    }

    /**
     * @brief setup the "QM" method of the system
     *
     */
    void QMSetup::setupQMMethod(const QMSettings &qmSettings)
    {
        _qmCapableEngine.setQMRunner(qmSettings);
    }

    /**
     * @brief setup the ASE DFTB+ method of the system
     *
     */
    void QMSetup::setupQMMethodAseXtb(const QMSettings &qmSettings)
    {
        if (!(qmSettings.getQMMethod() == QMMethod::ASE_XTB))
            return;

        const auto xtbMethod = qmSettings.getXtbMethod();
        if (xtbMethod == XtbMethod::GFN1)
        {
            references::ReferencesOutput::addReferenceFile(
                references::GFN1_FILE
            );
        }
        else if (xtbMethod == XtbMethod::GFN2)
        {
            references::ReferencesOutput::addReferenceFile(
                references::GFN2_FILE
            );
        }
        else if (xtbMethod == XtbMethod::IPEA1)
        {
            references::ReferencesOutput::addReferenceFile(
                references::IPEA1_FILE
            );
        }
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
    void QMSetup::setupWriteInfo(const QMSettings &qmSettings) const
    {
        using enum QMMethod;

        // Cast QMCapableEngine to Engine to access output methods
        auto &engine    = dynamic_cast<engine::Engine &>(_qmCapableEngine);
        auto &logOutput = engine.getLogOutput();

        const auto qmMethod = qmSettings.getQMMethod();
        const auto qmRunnerMessage =
            std::format("QM runner: {}", QMMethodMeta::toString(qmMethod));

        logOutput.writeSetupInfo(qmRunnerMessage);
        logOutput.writeEmptyLine();

        if (qmSettings.isExternalQMRunner())
        {
            const auto qmScriptMessage =
                std::format("QM script: {}", qmSettings.getQMScript());

            logOutput.writeSetupInfo(qmScriptMessage);
        }

        if (qmMethod == MACE)
        {
            const auto modelType = qmSettings.getMaceModelType();
            const auto modelSize = qmSettings.getMaceModel();
            const auto modelPath = qmSettings.getMaceModelPath();
            const auto floatingPointStr =
                settings::GeneralSettings::getFloatingPointPybindString();
            const auto        maceMode = qmSettings.getMaceMode();
            const auto *const useDisp =
                qmSettings.useDispersionCorr() ? "on" : "off";

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

            const auto modelPath           = qmSettings.getFennolModelPath();
            const auto useGPUPreprocessing = qmSettings.useGPUPreprocessing();
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
            const auto slakosType         = qmSettings.getSlakosType();
            const auto slakosPath         = qmSettings.getSlakosPath();
            const auto thirdOrder         = qmSettings.useThirdOrderDftb();
            const auto hubbardDerivs      = qmSettings.getHubbardDerivs();
            const auto ishubbardDerivsSet = qmSettings.isHubbardDerivsSet();
            const auto dispersion         = qmSettings.useDispersionCorr();

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
            const auto hubbardDerivsMsg =
                std::format("Hubbard derivatives:  {}", string(hubbardDerivs));
            const auto threeOBHubbardDerivsMsg = std::format(
                "3ob approach has been chosen while setting custom Hubbard "
                "derivatives. This setup is not recommended."
            );

            logOutput.writeSetupInfo(slakosTypeMsg);
            logOutput.writeSetupInfo(slakosPathMsg);
            logOutput.writeSetupInfo(dispersionMsg);
            logOutput.writeSetupInfo(thirdOrderMsg);
            if (ishubbardDerivsSet)
                logOutput.writeSetupInfo(hubbardDerivsMsg);

            // Warnings for non-recommended setups
            if (slakosType == SlakosType::THREEOB && !thirdOrder)
            {
                logOutput.writeEmptyLine();
                logOutput.writeSetupWarning(threeOBThirdOrderMsg);
                out::StdoutOutput::writeSetupWarning(threeOBThirdOrderMsg);
            }

            if (slakosType == SlakosType::THREEOB && ishubbardDerivsSet)
            {
                logOutput.writeEmptyLine();
                logOutput.writeSetupWarning(threeOBHubbardDerivsMsg);
                out::StdoutOutput::writeSetupWarning(threeOBHubbardDerivsMsg);
            }
        }

        if (qmMethod == ASE_XTB)
        {
            const auto xtbMethod = qmSettings.getXtbMethod();

            const auto xtbMethodMsg = std::format(
                "xTB Parametrization:   {}",
                XtbMethodMeta::toString(xtbMethod)
            );

            logOutput.writeSetupInfo(xtbMethodMsg);
        }

        const auto qm_loop_time_limit = qmSettings.getQMLoopTimeLimit();

        const auto qmLoopTimeLimitMsg = std::format(
            "QM looptime limit: {}",
            qm_loop_time_limit > 0 ? std::format("{} s", qm_loop_time_limit)
                                   : "unlimited"
        );

        logOutput.writeEmptyLine();
        logOutput.writeSetupInfo(qmLoopTimeLimitMsg);
        logOutput.writeEmptyLine();
    }

}   // namespace setup
