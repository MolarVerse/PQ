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

#include <string_view>   // for string_view

#include "dftbplusRunner.hpp"      // for DFTBPlusRunner
#include "exceptions.hpp"          // for InputFileException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "pyscfRunner.hpp"         // for PySCFRunner
#include "qmSettings.hpp"          // for QMMethod, QMSettings
#include "qmmdEngine.hpp"          // for QMMDEngine
#include "settings.hpp"            // for Settings
#include "stdoutOutput.hpp"        // for StdoutOutput
#include "stringUtilities.hpp"     // for toLowerCopy
#include "turbomoleRunner.hpp"     // for TurbomoleRunner

using setup::QMSetup;
using namespace settings;
using namespace engine;
using namespace QM;
using namespace utilities;
using namespace customException;

/**
 * @brief wrapper to build QMSetup object and call setup
 *
 * @param engine
 */
void setup::setupQM(Engine &engine)
{
    if (!Settings::isQMActivated())
        return;

    engine.getStdoutOutput().writeSetup("QM runner");
    engine.getLogOutput().writeSetup("QM runner");

    QMSetup qmSetup(dynamic_cast<QMMDEngine &>(engine));
    qmSetup.setup();
}

/**
 * @brief constructor
 *
 * @param engine
 */
QMSetup::QMSetup(QMMDEngine &engine) : _engine(engine) {}

/**
 * @brief setup QM-MD for all subtypes
 *
 */
void QMSetup::setup()
{
    setupQMMethod();

    setupQMMethodAseDftbPlus();

    if (QMSettings::isExternalQMRunner())
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
    _engine.setQMRunner(QMSettings::getQMMethod());
}

/**
 * @brief setup the ASE DFTB+ method of the system
 *
 */
void QMSetup::setupQMMethodAseDftbPlus()
{
    if (!(QMSettings::getQMMethod() == QMMethod::ASEDFTBPLUS))
        return;

    if (QMSettings::getSlakosType() == SlakosType::THREEOB &&
        !QMSettings::isThirdOrderDftbSet())
        QMSettings::setUseThirdOrderDftb(true);

    if (!QMSettings::useThirdOrderDftb() && QMSettings::isHubbardDerivsSet())
        throw InputFileException(
            "You have set custom Hubbard derivatives but disabled 3rd order "
            "DFTB. "
            "This setup is invalid."
        );
}

/**
 * @brief checks if a singularity or static build is used and sets the qm_script
 * accordingly
 *
 * @details if a singularity or static build is used the qm_script is set to the
 * qm_script_full_path and the script path is set to the empty string to avoid
 * errors. This is necessary because the script can not be accessed from inside
 * the container. Therefore the user has to provide the script somewhere else
 * and give the full or relative path to it. For more information please refer
 * to the documentation.
 *
 */
void QMSetup::setupQMScript() const
{
    auto &qmRunner         = *_engine.getQMRunner();
    auto &externalQMRunner = dynamic_cast<ExternalQMRunner &>(qmRunner);

    const auto singularityString = externalQMRunner.getSingularity();
    const auto staticBuildString = externalQMRunner.getStaticBuild();

    const auto singularity = toLowerCopy(singularityString) == "on";
    const auto staticBuild = toLowerCopy(staticBuildString) == "on";

    const auto qmScript        = QMSettings::getQMScript();
    const auto isQMScriptEmpty = qmScript.empty();

    const auto qmScriptFullPath        = QMSettings::getQMScriptFullPath();
    const auto isQMScriptFullPathEmpty = qmScriptFullPath.empty();

    if (singularity || staticBuild)
    {
        if (isQMScriptFullPathEmpty)

            throw QMRunnerException(
                "You are using at least one of these settings: i) singularity "
                "build or/and ii) static build of PQ. Therefore the general "
                "setting with 'qm_script' to set only the name of the "
                "executable is not applicable. Please use "
                "'qm_script_full_path' instead and provide the full path to "
                "the executable. For singularity builds the script can not be "
                "accessed from inside the container. In case of a static build "
                "the binary may be shipped without the source code and again "
                "PQ might therefore not be able to locate the executable qm "
                "script. Therefore you have to provide the script somewhere "
                "else and give the full/relative path to it. For more "
                "information please refer to the documentation."
            );

        else if (!isQMScriptEmpty)

            throw QMRunnerException(
                "You have set both 'qm_script' and 'qm_script_full_path' in "
                "the input file. Please use only one the full path option as "
                "you are working either with a singularity build or a static "
                "build. For more information please refer to the "
                "documentation."
            );

        else
        {
            // setting script path to empty string to avoid errors
            externalQMRunner.setScriptPath("");

            // overwriting qm_script with full path
            QMSettings::setQMScript(QMSettings::getQMScriptFullPath());
        }
    }
    else if (isQMScriptEmpty && isQMScriptFullPathEmpty)

        throw InputFileException(
            "No qm_script provided. Please provide a qm_script in the input "
            "file."
        );

    else if (!isQMScriptFullPathEmpty && isQMScriptEmpty)
    {
        // setting script path to empty string to avoid errors
        externalQMRunner.setScriptPath("");

        // overwriting qm_script with full path
        QMSettings::setQMScript(QMSettings::getQMScriptFullPath());
    }
    else if (!isQMScriptFullPathEmpty && !isQMScriptEmpty)

        throw InputFileException(
            "You have set both 'qm_script' and 'qm_script_full_path' in the "
            "input file. They are mutually exclusive. Please use only one of "
            "them. For more information please refer to the documentation."
        );
}

/**
 * @brief set coulomb radius cutoff to 0.0 for QM-MD, QM-RPMD
 *
 */
void QMSetup::setupCoulombRadiusCutOff() const
{
    using enum JobType;

    const auto jobType = Settings::getJobtype();

    if (jobType == QM_MD || jobType == RING_POLYMER_QM_MD)
        PotentialSettings::setCoulombRadiusCutOff(0.0);
}

/**
 * @brief write info about the QM setup
 *
 */
void QMSetup::setupWriteInfo() const
{
    using enum QMMethod;

    auto &logOutput = _engine.getLogOutput();
    auto &stdOut    = _engine.getStdoutOutput();

    const auto qmMethod        = QMSettings::getQMMethod();
    const auto qmRunnerMessage = std::format("QM runner: {}", string(qmMethod));

    logOutput.writeSetupInfo(qmRunnerMessage);
    logOutput.writeEmptyLine();

    if (QMSettings::isExternalQMRunner())
    {
        const auto qmScript        = QMSettings::getQMScript();
        const auto qmScriptMessage = std::format("QM script: {}", qmScript);

        logOutput.writeSetupInfo(qmScriptMessage);
    }

    if (qmMethod == MACE)
    {
        const auto modelType = QMSettings::getMaceModelType();
        const auto modelSize = QMSettings::getMaceModelSize();
        const auto fp        = Settings::getFloatingPointPybindString();
        const auto useDisp   = QMSettings::useDispersionCorr() ? "on" : "off";

        // clang-format off
        const auto modelTypeMsg = std::format("Model type:            {}", string(modelType));
        const auto modelSizeMsg = std::format("Model size:            {}", string(modelSize));
        const auto fpMsg        = std::format("Floating point type:   {}", fp);
        const auto dispCorrMsg  = std::format("Dispersion Correction: {}", useDisp);
        // clang-format on

        logOutput.writeSetupInfo(modelTypeMsg);
        logOutput.writeSetupInfo(modelSizeMsg);
        logOutput.writeSetupInfo(fpMsg);
        logOutput.writeSetupInfo(dispCorrMsg);
    }

    if (qmMethod == FAIRCHEM)
    {
        const auto modelName = QMSettings::getFairchemModelName();
        const auto modelPath = QMSettings::getFairchemModelPath();
        const auto cpu       = QMSettings::useFairchemOnCPU() ? "on" : "off";

        // clang-format off
        const auto modelTypeMsg     = std::format("Model name:           {}", modelName);
        const auto modelPathMsg     = std::format("Model path:           {}", modelPath);
        const auto cpuMsg           = std::format("Run on CPU:           {}", cpu);
        // clang-format on

        logOutput.writeSetupInfo(modelTypeMsg);
    }

    if (qmMethod == ASEDFTBPLUS)
    {
        const auto slakosType         = QMSettings::getSlakosType();
        const auto slakosPath         = QMSettings::getSlakosPath();
        const auto thirdOrder         = QMSettings::useThirdOrderDftb();
        const auto hubbardDerivs      = QMSettings::getHubbardDerivs();
        const auto ishubbardDerivsSet = QMSettings::isHubbardDerivsSet();
        const auto dispersion         = QMSettings::useDispersionCorr();

        // clang-format off
        const auto slakosTypeMsg           = std::format("DFTB approach:        {}", string(slakosType));
        const auto slakosPathMsg           = std::format("sk file path:         {}", slakosPath);
        const auto dispersionMsg           = std::format("Dispersion is turned: {}", dispersion ? "on" : "off");
        const auto thirdOrderMsg           = std::format("3rd order is turned:  {}", thirdOrder ? "on" : "off");
        const auto threeOBThirdOrderMsg    = std::format("3ob approach has been chosen while disabling 3rd order DFTB. This setup is not recommended.");
        const auto hubbardDerivsMsg        = std::format("Hubbard derivatives:  {}", string(hubbardDerivs));
        const auto threeOBHubbardDerivsMsg = std::format("3ob approach has been chosen while setting custom Hubbard derivatives. This setup is not recommended.");
        // clang-format on

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
            stdOut.writeSetupWarning(threeOBThirdOrderMsg);
        }

        if (slakosType == SlakosType::THREEOB && ishubbardDerivsSet)
        {
            logOutput.writeEmptyLine();
            logOutput.writeSetupWarning(threeOBHubbardDerivsMsg);
            stdOut.writeSetupWarning(threeOBHubbardDerivsMsg);
        }
    }

    logOutput.writeEmptyLine();
}
