#include "energyOutput.hpp"

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData
#include "settings.hpp"             // for Settings
#include "thermostatSettings.hpp"   // for ThermostatSettings

#include <format>    // for format
#include <ostream>   // for basic_ostream, ofstream
#include <string>    // for operator<<

using namespace output;

/**
 * @brief Write the energy output
 *
 * @details
 * - Coulomb and Non-Coulomb energies contain the intra and inter energies.
 * - Bond, Angle, Dihedral and Improper energies are only available if the force field is active.
 * - qm energy is only available if qm is active.
 * - coulomb and non-coulomb energies are only available if mm is active.
 * - volume and density are only available if manostat is active.
 * - nose hoover momentum and friction energies are only available if nose hoover thermostat is active.
 *
 * @param step
 * @param data
 */
void EnergyOutput::write(const size_t step, const double loopTime, const physicalData::PhysicalData &data)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.12f}\t", data.getTemperature());
    _fp << std::format("{:20.12f}\t", data.getPressure());
    _fp << std::format("{:20.12f}\t", data.getTotalEnergy());

    if (settings::Settings::isQMActivated())
    {
        _fp << std::format("{:20.12f}\t", data.getQMEnergy());
        _fp << std::format("{:20.12f}\t", 0.0);   // TODO: implement
    }

    _fp << std::format("{:20.12f}\t", data.getKineticEnergy());
    _fp << std::format("{:20.12f}\t", data.getIntraEnergy());

    if (settings::Settings::isMMActivated())
    {
        _fp << std::format("{:20.12f}\t", data.getCoulombEnergy());
        _fp << std::format("{:20.12f}\t", data.getNonCoulombEnergy());
    }

    if (settings::ForceFieldSettings::isActive())
    {
        _fp << std::format("{:20.12f}\t", data.getBondEnergy());
        _fp << std::format("{:20.12f}\t", data.getAngleEnergy());
        _fp << std::format("{:20.12f}\t", data.getDihedralEnergy());
        _fp << std::format("{:20.12f}\t", data.getImproperEnergy());
    }

    if (settings::ManostatSettings::getManostatType() != settings::ManostatType::NONE)
    {
        _fp << std::format("{:20.12f}\t", data.getVolume());
        _fp << std::format("{:20.12f}\t", data.getDensity());
    }

    if (settings::ThermostatSettings::getThermostatType() == settings::ThermostatType::NOSE_HOOVER)
    {
        _fp << std::format("{:20.12f}\t", data.getNoseHooverMomentumEnergy());
        _fp << std::format("{:20.12f}\t", data.getNoseHooverFrictionEnergy());
    }

    _fp << std::format("{:20.5e}\t", norm(data.getMomentum()));
    _fp << std::format("{:12.5f}\n", loopTime);
}