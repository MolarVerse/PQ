#include "energyOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData

#include <iomanip>
#include <ostream>   // for basic_ostream, operator<<

using namespace std;
using namespace physicalData;
using namespace output;

/**
 * @brief Write the energy output
 *
 * @param step
 * @param data
 */
void EnergyOutput::write(const size_t step, const PhysicalData &data)
{
    _fp << right;
    _fp << setw(10);
    _fp << step;

    _fp << fixed;
    _fp << setprecision(12);

    _fp << right;
    _fp << setw(20);
    _fp << data.getTemperature();

    _fp << right;
    _fp << setw(20);
    _fp << data.getPressure();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << data.getKineticEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << data.getCoulombEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << data.getNonCoulombEnergy();

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << right;
    _fp << setw(20);
    _fp << 0.0;

    _fp << scientific;
    _fp << right;
    _fp << setw(20);
    _fp << data.getMomentum();

    _fp << fixed;
    _fp << right;
    _fp << setw(20);
    _fp << 0.0;
    _fp << '\n';
}