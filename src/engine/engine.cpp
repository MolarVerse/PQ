#include "engine.hpp"
#include "output.hpp"

Engine::Engine()
{
    _energyOutput.setFilename("default.en");
    _xyzOutput.setFilename("default.xyz");
    _velOutput.setFilename("default.vel");
    _forceOutput.setFilename("default.force");
    _logOutput.setFilename("default.log");
    _rstFileOutput.setFilename("default.rst");
    _chargeOutput.setFilename("default.chg");
    _infoOutput.setFilename("default.info");
}

Engine::Engine(const Engine &engine)
{
    _settings = engine._settings;
    _jobType = engine._jobType;
    _energyOutput = engine._energyOutput;
    _xyzOutput = engine._xyzOutput;
    _velOutput = engine._velOutput;
    _forceOutput = engine._forceOutput;
    _logOutput = engine._logOutput;
    _stdoutOutput = engine._stdoutOutput;
    _rstFileOutput = engine._rstFileOutput;
    _chargeOutput = engine._chargeOutput;
    _infoOutput = engine._infoOutput;
    _simulationBox = engine._simulationBox;
    _integrator = engine._integrator;
}