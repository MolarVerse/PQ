#include "qmSetup.hpp"

#include "dftbplusRunner.hpp"   // for DFTBPlusRunner
#include "exceptions.hpp"       // for InputFileException
#include "qmSettings.hpp"       // for QMMethod, QMSettings
#include "qmmdEngine.hpp"       // for QMMDEngine

#include <string_view>   // for string_view

using setup::QMSetup;

/**
 * @brief wrapper to build QMSetup object and call setup
 *
 * @param engine
 */
void setup::setupQM(engine::QMMDEngine &engine)
{
    QMSetup qmSetup(engine);
    qmSetup.setup();
}

/**
 * @brief setup the "QM" of the system
 *
 */
void QMSetup::setup()
{
    const auto method = settings::QMSettings::getQMMethod();

    if (method == settings::QMMethod::DFTBPLUS)
        _engine.setQMRunner(QM::DFTBPlusRunner());
    else
        throw customException::InputFileException(
            "A qm based jobtype was requested but no external program via \"qm_prog\" provided");
}