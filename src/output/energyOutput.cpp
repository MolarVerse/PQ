#include "energyOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData

#include <format>    // for format
#include <ostream>   // for basic_ostream, operator<<

using namespace output;

/**
 * @brief Write the energy output
 *
 * @param step
 * @param data
 */
void EnergyOutput::write(const size_t step, const physicalData::PhysicalData &data)
{
    _fp << std::format("{:10d}\t", step);
    _fp << std::format("{:20.12f}\t", data.getTemperature());
    _fp << std::format("{:20.12f}\t", data.getPressure());
    _fp << std::format("{:20.12f}\t", 0.0);
    _fp << std::format("{:20.12f}\t", data.getKineticEnergy());
    _fp << std::format("{:20.12f}\t", 0.0);
    _fp << std::format("{:20.12f}\t", data.getCoulombEnergy());
    _fp << std::format("{:20.12f}\t", data.getNonCoulombEnergy());
    _fp << std::format("{:20.12f}\t", 0.0);
    _fp << std::format("{:20.12f}\t", 0.0);
    _fp << std::format("{:20.12e}\t", data.getMomentum());
    _fp << std::format("{:20.12f}\n", 0.0);
}