#include "maxwellBoltzmann.hpp"

#include "constants/conversionFactors.hpp"           // for _AMU_TO_KG_
#include "constants/internalConversionFactors.hpp"   // for _VELOCITY_UNIT_TO_SI_
#include "constants/natureConstants.hpp"             // for _BOLTZMANN_CONSTANT_
#include "resetKinetics.hpp"                         // for ResetKinetics
#include "simulationBox.hpp"                         // for SimulationBox
#include "thermostatSettings.hpp"                    // for ThermostatSettings

#include <algorithm>    // for __for_each_fn
#include <cmath>        // for sqrt
#include <functional>   // for identity

using maxwellBoltzmann::MaxwellBoltzmann;

/**
 * @brief generate boltzmann distributed velocities for all atoms in the simulation box
 *
 * @details using a standard deviation of sqrt(kb*T/m) for each component of the velocity vector
 *
 * @param simBox
 */
void MaxwellBoltzmann::initializeVelocities(simulationBox::SimulationBox &simBox)
{
    auto generateVelocities = [this](auto &atom)
    {
        const auto mass = atom->getMass() * constants::_AMU_TO_KG_;

        const auto stddev =
            ::sqrt(constants::_BOLTZMANN_CONSTANT_ * settings::ThermostatSettings::getTargetTemperature() / mass) /
            constants::_VELOCITY_UNIT_TO_SI_;

        std::normal_distribution<double> distribution{0.0, stddev};

        atom->setVelocity({distribution(_generator), distribution(_generator), distribution(_generator)});
    };

    std::ranges::for_each(simBox.getAtoms(), generateVelocities);

    auto resetKinetics = resetKinetics::ResetKinetics();
    resetKinetics.setMomentum(simBox.calculateMomentum());
    resetKinetics.resetMomentum(simBox);
    resetKinetics.resetAngularMomentum(simBox);
    resetKinetics.resetTemperature(simBox);
}