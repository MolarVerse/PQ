#include "qmSetup.hpp"

#include "dftbplusRunner.hpp"
#include "exceptions.hpp"
#include "qmSettings.hpp"
#include "qmmdEngine.hpp"

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