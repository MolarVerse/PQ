#include "ringPolymerEngine.hpp"

#include "constants.hpp"             // for _RPMD_PREFACTOR_
#include "output.hpp"                // for Output
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "thermostatSettings.hpp"    // for ThermostatSettings

using engine::Engine;
using engine::RingPolymerEngine;

void RingPolymerEngine::writeOutput()
{
    Engine::writeOutput();

    if (0 == _step % output::Output::getOutputFrequency())
    {
        const auto step0 = _timings.getStepCount();

        _engineOutput.writeRingPolymerRstFile(_ringPolymerBeads, step0 + _step);
        _engineOutput.writeRingPolymerXyzFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerVelFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerForceFile(_ringPolymerBeads);
        _engineOutput.writeRingPolymerChargeFile(_ringPolymerBeads);
    }
}

void RingPolymerEngine::coupleRingPolymerBeads()
{
    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();
    const auto temperature   = settings::ThermostatSettings::getTargetTemperature();
    const auto rpmd_factor   = constants::_RPMD_PREFACTOR_ * numberOfBeads * numberOfBeads * temperature * temperature;

    for (size_t i = 0; i < numberOfBeads; ++i)
    {
        auto &bead1 = _ringPolymerBeads[i];
        auto &bead2 = _ringPolymerBeads[(i + 1) % numberOfBeads];

        for (size_t j = 0; j < bead1.getNumberOfAtoms(); ++j)
        {
            auto &atom1 = bead1.getAtom(j);
            auto &atom2 = bead2.getAtom(j);

            const auto forceConstant = rpmd_factor * atom1.getMass();
            const auto force         = forceConstant * (atom2.getPosition() - atom1.getPosition());

            atom1.addForce(force);
            atom2.addForce(-force);
        }
    }
}

void RingPolymerEngine::combineBeads()
{
    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();

    std::ranges::for_each(_simulationBox.getAtoms(), [](auto &atom) { atom->setPosition({0.0, 0.0, 0.0}); });

    auto addCoordinates = [this, numberOfBeads](auto &bead)
    {
        for (size_t i = 0; i < bead.getNumberOfAtoms(); ++i)
        {
            auto &atom = bead.getAtom(i);

            _simulationBox.getAtom(i).addPosition(atom.getPosition() / double(numberOfBeads));
        }
    };

    std::ranges::for_each(_ringPolymerBeads, addCoordinates);
}