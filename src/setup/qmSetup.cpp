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

#include "dftbplusRunner.hpp"      // for DFTBPlusRunner
#include "exceptions.hpp"          // for InputFileException
#include "potentialSettings.hpp"   // for PotentialSettings
#include "pyscfRunner.hpp"         // for PySCFRunner
#include "qmSettings.hpp"          // for QMMethod, QMSettings
#include "qmmdEngine.hpp"          // for QMMDEngine
#include "settings.hpp"            // for Settings
#include "stringUtilities.hpp"     // for toLowerCopy
#include "turbomoleRunner.hpp"     // for TurbomoleRunner

#include <string_view>   // for string_view

using setup::QMSetup;

/**
 * @brief wrapper to build QMSetup object and call setup
 *
 * @param engine
 */
void setup::setupQM(engine::QMMDEngine &engine)
{
    engine.getStdoutOutput().writeSetup("QM runner");
    engine.getLogOutput().writeSetup("QM runner");

    QMSetup qmSetup(engine);
    qmSetup.setup();

    engine.getLogOutput().writeSetupInfo("QM runner: " + string(settings::QMSettings::getQMMethod()));
    engine.getLogOutput().writeSetupInfo("QM script: " + settings::QMSettings::getQMScript());
}

/**
 * @brief setup QM-MD for all subtypes
 *
 */
void QMSetup::setup()
{

    setupQMMethod();
    setupQMScript();
    setupCoulombRadiusCutOff();
}

/**
 * @brief setup the "QM" method of the system
 *
 */
void QMSetup::setupQMMethod()
{

    const auto method = settings::QMSettings::getQMMethod();

    if (method == settings::QMMethod::DFTBPLUS)
        _engine.setQMRunner(QM::DFTBPlusRunner());

    else if (method == settings::QMMethod::PYSCF)
        _engine.setQMRunner(QM::PySCFRunner());

    else if (method == settings::QMMethod::TURBOMOLE)
        _engine.setQMRunner(QM::TurbomoleRunner());

    else
        throw customException::InputFileException(
            "A qm based jobtype was requested but no external program via \"qm_prog\" provided");
}

/**
 * @brief checks if a singularity or static build is used and sets the qm_script accordingly
 *
 * @details if a singularity or static build is used the qm_script is set to the qm_script_full_path
 * and the script path is set to the empty string to avoid errors. This is necessary because
 * the script can not be accessed from inside the container. Therefore the user has to provide
 * the script somewhere else and give the full or relative path to it. For more information please
 * refer to the documentation.
 *
 */
void QMSetup::setupQMScript() const
{
    const bool singularity = utilities::toLowerCopy(_engine.getQMRunner()->getSingularity()) == "on";
    const bool staticBuild = utilities::toLowerCopy(_engine.getQMRunner()->getStaticBuild()) == "on";

    if (singularity || staticBuild)
    {
        if (settings::QMSettings::getQMScriptFullPath().empty())
            throw customException::QMRunnerException(
                R"(
You are using at least one of these settings: i) singularity build or/and ii) static build of PQ.
Therefore the general setting with "qm_script" to set only the name of the executable is not
applicable. Please use "qm_script_full_path" instead and provide the full path to the executable. For
singularity builds the script can not be accessed from inside the container. In case of a static build
the binary may be shipped without the source code and again PQ might therefore not be able to 
locate the executable qm script. Therefore you have to provide the script somewhere else and give the
full/relative path to it. For more information please refer to the documentation.
)");
        else if (!settings::QMSettings::getQMScript().empty())
            throw customException::QMRunnerException(
                R"(
You have set both "qm_script" and "qm_script_full_path" in the input file. Please use only one the full
path option as you are working either with a singularity build or a static build. For more information
please refer to the documentation.
)");
        else
        {
            _engine.getQMRunner()->setScriptPath("");   // setting script path to empty string to avoid errors
            settings::QMSettings::setQMScript(
                settings::QMSettings::getQMScriptFullPath());   // overwriting qm_script with full path
        }
    }
    else if (settings::QMSettings::getQMScript().empty() && settings::QMSettings::getQMScriptFullPath().empty())
        throw customException::InputFileException("No qm_script provided. Please provide a qm_script in the input file.");
    else if (!settings::QMSettings::getQMScriptFullPath().empty() && settings::QMSettings::getQMScript().empty())
    {
        _engine.getQMRunner()->setScriptPath("");   // setting script path to empty string to avoid errors
        settings::QMSettings::setQMScript(settings::QMSettings::getQMScriptFullPath());   // overwriting qm_script with full path
    }
    else if (!settings::QMSettings::getQMScriptFullPath().empty() && !settings::QMSettings::getQMScript().empty())
        throw customException::InputFileException(
            R"(
You have set both "qm_script" and "qm_script_full_path" in the input file. They are mutually exclusive.
Please use only one of them. For more information please refer to the documentation.
            )");
}

/**
 * @brief set coulomb radius cutoff to 0.0 for QM-MD, QM-RPMD
 *
 */
void QMSetup::setupCoulombRadiusCutOff() const
{
    const auto jobType = settings::Settings::getJobtype();

    if (jobType == settings::JobType::QM_MD || jobType == settings::JobType::RING_POLYMER_QM_MD)
        settings::PotentialSettings::setCoulombRadiusCutOff(0.0);
}
