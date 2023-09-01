#include "energyOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData

#include <iomanip>
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
    _fp << std::right;
    _fp << std::setw(10);
    _fp << step;

    _fp << std::fixed;
    _fp << std::setprecision(12);

    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getTemperature();

    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getPressure();

    _fp << std::right;
    _fp << std::setw(20);
    _fp << 0.0;

    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getKineticEnergy();

    _fp << std::right;
    _fp << std::setw(20);
    _fp << 0.0;

    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getCoulombEnergy();

    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getNonCoulombEnergy();

    _fp << std::right;
    _fp << std::setw(20);
    _fp << 0.0;

    _fp << std::right;
    _fp << std::setw(20);
    _fp << 0.0;

    _fp << std::scientific;
    _fp << std::right;
    _fp << std::setw(20);
    _fp << data.getMomentum();

    _fp << std::fixed;
    _fp << std::right;
    _fp << std::setw(20);
    _fp << 0.0;
    _fp << '\n';
}