/*****************************************************************************
<GPL_HEADER>

    PIMD-QMCF
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
 * @brief set coulomb radius cutoff to 0.0 for QM-MD, QM-RPMD
 *
 */
void QMSetup::setupCoulombRadiusCutOff() const
{
    const auto jobType = settings::Settings::getJobtype();

    if (jobType == settings::JobType::QM_MD || jobType == settings::JobType::RING_POLYMER_QM_MD)
        settings::PotentialSettings::setCoulombRadiusCutOff(0.0);
}