#include "outputFilesSetup.hpp"

#include "engine.hpp"
#include "outputFileSettings.hpp"

using setup::OutputFilesSetup;

/**
 * @brief wrapper function to setup output files
 *
 */
void setup::setupOutputFiles(engine::Engine &engine)
{
    OutputFilesSetup outputFilesSetup(engine);
    outputFilesSetup.setup();
}

/**
 * @brief setup output files
 *
 */
void OutputFilesSetup::setup()
{
    _engine.getRstFileOutput().setFilename(settings::OutputFileSettings::getRestartFileName());
    _engine.getEnergyOutput().setFilename(settings::OutputFileSettings::getEnergyFileName());
    _engine.getXyzOutput().setFilename(settings::OutputFileSettings::getTrajectoryFileName());
    _engine.getLogOutput().setFilename(settings::OutputFileSettings::getLogFileName());
    _engine.getInfoOutput().setFilename(settings::OutputFileSettings::getInfoFileName());
    _engine.getVelOutput().setFilename(settings::OutputFileSettings::getVelocityFileName());
    _engine.getForceOutput().setFilename(settings::OutputFileSettings::getForceFileName());
    _engine.getChargeOutput().setFilename(settings::OutputFileSettings::getChargeFileName());
    _engine.getMomentumOutput().setFilename(settings::OutputFileSettings::getMomentumFileName());

    _engine.getRingPolymerRstFileOutput().setFilename(settings::OutputFileSettings::getRingPolymerRestartFileName());
    _engine.getRingPolymerXyzOutput().setFilename(settings::OutputFileSettings::getRingPolymerTrajectoryFileName());
    _engine.getRingPolymerVelOutput().setFilename(settings::OutputFileSettings::getRingPolymerVelocityFileName());
    _engine.getRingPolymerForceOutput().setFilename(settings::OutputFileSettings::getRingPolymerForceFileName());
    _engine.getRingPolymerChargeOutput().setFilename(settings::OutputFileSettings::getRingPolymerChargeFileName());

    _engine.getLogOutput().writeHeader();
    _engine.getStdoutOutput().writeHeader();
}