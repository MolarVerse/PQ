#include "energyOutput.hpp"

#include "forceFieldSettings.hpp"   // for ForceFieldSettings
#include "manostatSettings.hpp"     // for ManostatSettings
#include "physicalData.hpp"         // for PhysicalData

#include <format>    // for format
#include <ostream>   // for basic_ostream, ofstream
#include <string>    // for operator<<

using namespace output;

/**
 * @brief Write the energy output
 *
 * @details Coulomb and Non-Coulomb energies contain the intra and inter energies. Bond, Angle, Dihedral and Improper energies are
 * only available if the force field is active.
 *
 * @param step
 * @param data
 */
void EnergyOutput::write(const size_t step, const double loopTime, const physicalData::PhysicalData &data)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.12f}\t", data.getTemperature());
    _fp << std::format("{:20.12f}\t", data.getPressure());
    _fp << std::format("{:20.12f}\t", data.getPotentialEnergy());
    _fp << std::format("{:20.12f}\t", data.getKineticEnergy());
    _fp << std::format("{:20.12f}\t", data.getIntraEnergy());
    _fp << std::format("{:20.12f}\t", data.getCoulombEnergy());
    _fp << std::format("{:20.12f}\t", data.getNonCoulombEnergy());

    if (settings::ForceFieldSettings::isActive())
    {
        _fp << std::format("{:20.12f}\t", data.getBondEnergy());
        _fp << std::format("{:20.12f}\t", data.getAngleEnergy());
        _fp << std::format("{:20.12f}\t", data.getDihedralEnergy());
        _fp << std::format("{:20.12f}\t", data.getImproperEnergy());
    }

    if (settings::ManostatSettings::getManostatType() != "none")
    {
        _fp << std::format("{:20.12f}\t", data.getVolume());
        _fp << std::format("{:20.12f}\t", data.getDensity());
    }

    _fp << std::format("{:20.5e}\t", data.getMomentum());
    _fp << std::format("{:12.5f}\n", loopTime);
}