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
#include "stringUtilities.hpp"

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
     *
     */
    QMSetup::QMSetup(
        engine::QMCapableEngine &qmCapableEngine,
        const QMSettings        &qmSettings,
        const MaceSettings      &maceSettings,
        const FennolSettings    &fennolSettings
    )
        : _qmCapableEngine(qmCapableEngine),
          _qmSettings(qmSettings),
          _maceSettings(maceSettings),
          _fennolSettings(fennolSettings)
    {
    }

    /**
     * @brief wrapper to build QMSetup object and call setup
     *
     * @param engine
     * @param references Reference settings object used to manage and add
     * necessary reference files for the specified QM method.
     * TODO(97gamjak): remove references as soon as qmsettings is in place
     */
    void setupQM(engine::Engine &engine, ReferenceSettings &references)
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
                settings.fennolSettings
            );
            qmSetup.setup(references);
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
    void QMSetup::setup(ReferenceSettings &references)
    {
        setupQMMethod();

        _setupQMMethodAseDftbPlus();

        _setupQMMethodAseXtb(references);

        if (_qmSettings->isExternalQMRunner())
            setupQMScript();

        setupCoulombRadiusCutOff();

        setupWriteInfo();
    }

    /**
     * @brief setup the "QM" method of the system
     *
     */
    void QMSetup::setupQMMethod()
    {
        _qmCapableEngine
            .setQMRunner(_qmSettings, _maceSettings, _fennolSettings);
    }

    /**
     * @brief setup the ASE DFTB+ method of the system
     *
     */
    void QMSetup::_setupQMMethodAseDftbPlus() const
    {
        if (!(_qmSettings->getQMMethod() == QMMethod::ASE_DFTBPLUS))
            return;

        if (QMSettings::getSlakosType() == SlakosType::THREEOB &&
            !QMSettings::isThirdOrderDftbSet())
            QMSettings::setUseThirdOrderDftb(true);
    }

    /**
     * @brief setup the ASE DFTB+ method of the system
     *
     */
    void QMSetup::_setupQMMethodAseXtb(ReferenceSettings &references) const
    {
        if (!(_qmSettings->getQMMethod() == QMMethod::ASE_XTB))
            return;

        if (QMSettings::getXtbMethod() == XtbMethod::GFN1)
        {
            references.addReferenceFile(ReferenceFiles::GFN1_FILE);
        }
        else if (QMSettings::getXtbMethod() == XtbMethod::GFN2)
        {
            references.addReferenceFile(ReferenceFiles::GFN2_FILE);
        }
        else if (QMSettings::getXtbMethod() == XtbMethod::IPEA1)
        {
            references.addReferenceFile(ReferenceFiles::IPEA1_FILE);
        }
    }

    /**
     * @brief checks if a singularity or static build is used and sets the
     * qm_script accordingly
     *
     * @details if a singularity or static build is used the qm_script is set to
     * the qm_script_full_path and the script path is set to the empty string to
     * avoid errors. This is necessary because the script can not be accessed
     * from inside the container. Therefore the user has to provide the script
     * somewhere else and give the full or relative path to it. For more
     * information please refer to the documentation.
     *
     */
    void QMSetup::setupQMScript() const
    {
        auto &qmRunner         = *_qmCapableEngine.getQMRunner();
        auto &externalQMRunner = dynamic_cast<QM::ExternalQMRunner &>(qmRunner);

        const auto singularityString = QM::ExternalQMRunner::getSingularity();
        const auto staticBuildString = QM::ExternalQMRunner::getStaticBuild();

        const auto singularity =
            utilities::toLowerCopy(singularityString) == "on";
        const auto staticBuild =
            utilities::toLowerCopy(staticBuildString) == "on";

        const auto qmScript        = QMSettings::getQMScript();
        const auto isQMScriptEmpty = qmScript.empty();

        const auto qmScriptFullPath        = QMSettings::getQMScriptFullPath();
        const auto isQMScriptFullPathEmpty = qmScriptFullPath.empty();

        if (singularity || staticBuild)
        {
            if (isQMScriptFullPathEmpty)
            {
                throw exc::QMRunnerException(
                    "You are using at least one of these settings: i) "
                    "singularity "
                    "build or/and ii) static build of PQ. Therefore the "
                    "general "
                    "setting with 'qm_script' to set only the name of the "
                    "executable is not applicable. Please use "
                    "'qm_script_full_path' instead and provide the full path "
                    "to "
                    "the executable. For singularity builds the script can not "
                    "be "
                    "accessed from inside the container. In case of a static "
                    "build "
                    "the binary may be shipped without the source code and "
                    "again "
                    "PQ might therefore not be able to locate the executable "
                    "qm "
                    "script. Therefore you have to provide the script "
                    "somewhere "
                    "else and give the full/relative path to it. For more "
                    "information please refer to the documentation."
                );
            }

            if (!isQMScriptEmpty)
            {
                throw exc::QMRunnerException(
                    "You have set both 'qm_script' and 'qm_script_full_path' "
                    "in "
                    "the input file. Please use only one the full path option "
                    "as "
                    "you are working either with a singularity build or a "
                    "static "
                    "build. For more information please refer to the "
                    "documentation."
                );
            }

            // setting script path to empty string to avoid errors
            externalQMRunner.setScriptPath("");

            // overwriting qm_script with full path
            QMSettings::setQMScript(QMSettings::getQMScriptFullPath());
        }
        else if (isQMScriptEmpty && isQMScriptFullPathEmpty)
        {
            throw exc::InputFileException(
                "No qm_script provided. Please provide a qm_script in the "
                "input "
                "file."
            );
        }
        else if (!isQMScriptFullPathEmpty && isQMScriptEmpty)
        {
            // setting script path to empty string to avoid errors
            externalQMRunner.setScriptPath("");

            // overwriting qm_script with full path
            QMSettings::setQMScript(QMSettings::getQMScriptFullPath());
        }
        else if (!isQMScriptFullPathEmpty && !isQMScriptEmpty)
        {
            throw exc::InputFileException(
                "You have set both 'qm_script' and 'qm_script_full_path' in "
                "the "
                "input file. They are mutually exclusive. Please use only one "
                "of "
                "them. For more information please refer to the documentation."
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
            const auto qmScript        = QMSettings::getQMScript();
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
                QMSettings::useDispersionCorr() ? "on" : "off";

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
            const auto slakosType         = QMSettings::getSlakosType();
            const auto slakosPath         = QMSettings::getSlakosPath();
            const auto thirdOrder         = QMSettings::useThirdOrderDftb();
            const auto hubbardDerivs      = QMSettings::getHubbardDerivs();
            const auto ishubbardDerivsSet = QMSettings::isHubbardDerivsSet();
            const auto dispersion         = QMSettings::useDispersionCorr();

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
            const auto xtbMethod = QMSettings::getXtbMethod();

            const auto xtbMethodMsg = std::format(
                "xTB Parametrization:   {}",
                XtbMethodMeta::toString(xtbMethod)
            );

            logOutput.writeSetupInfo(xtbMethodMsg);
        }

        const auto qm_loop_time_limit = QMSettings::getQMLoopTimeLimit();

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
