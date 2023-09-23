#include "ringPolymerEngine.hpp"

#include "constants.hpp"             // for _RPMD_PREFACTOR_
#include "output.hpp"                // for Output
#include "outputFileSettings.hpp"    // for OutputFileSettings
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "thermostatSettings.hpp"    // for ThermostatSettings

using engine::Engine;
using engine::RingPolymerEngine;

void RingPolymerEngine::writeOutput()
{
    Engine::writeOutput();

    if (0 == _step % settings::OutputFileSettings::getOutputFrequency())
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

    std::vector<double> ringPolymerEnergy(numberOfBeads, 0.0);

    for (size_t i = 0; i < numberOfBeads; ++i)
    {
        auto &bead1 = _ringPolymerBeads[i];
        auto &bead2 = _ringPolymerBeads[(i + 1) % numberOfBeads];

        for (size_t j = 0, numberOfAtoms = bead1.getNumberOfAtoms(); j < numberOfAtoms; ++j)
        {
            auto &atom1 = bead1.getAtom(j);
            auto &atom2 = bead2.getAtom(j);

            const auto deltaPosition = atom2.getPosition() - atom1.getPosition();

            const auto forceConstant = rpmd_factor * atom1.getMass();
            const auto force         = forceConstant * deltaPosition;

            ringPolymerEnergy[i] += 0.5 * forceConstant * normSquared(deltaPosition);

            atom1.addForce(force);
            atom2.addForce(-force);
        }
    }

    _physicalData.setRingPolymerEnergy(ringPolymerEnergy);
}

void RingPolymerEngine::combineBeads()
{
    const auto numberOfBeads = settings::RingPolymerSettings::getNumberOfBeads();

    std::ranges::for_each(_simulationBox.getAtoms(),
                          [](auto &atom)
                          {
                              atom->setPosition({0.0, 0.0, 0.0});
                              atom->setVelocity({0.0, 0.0, 0.0});
                              atom->setForce({0.0, 0.0, 0.0});
                          });

    auto addCoordinates = [this, numberOfBeads](auto &bead)
    {
        for (size_t i = 0; i < bead.getNumberOfAtoms(); ++i)
        {
            auto &atom = bead.getAtom(i);

            _simulationBox.getAtom(i).addPosition(atom.getPosition() / double(numberOfBeads));
            _simulationBox.getAtom(i).addVelocity(atom.getVelocity() / double(numberOfBeads));
            _simulationBox.getAtom(i).addForce(atom.getForce() / double(numberOfBeads));
        }
    };

    std::ranges::for_each(_ringPolymerBeads, addCoordinates);
}