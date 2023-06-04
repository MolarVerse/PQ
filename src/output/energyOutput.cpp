#include "energyOutput.hpp"

#include <iomanip>

using namespace std;

void EnergyOutput::write(const size_t step, const PhysicalData &data)
{
    _fp << step << "\t";
    _fp << fixed;
    _fp << setprecision(12);
    _fp << data.getTemperature() << "\t";
    _fp << data.getPressure() << "\t";
    _fp << 0.0 << "\t";
    _fp << data.getKineticEnergy() << "\t";
    _fp << 0.0 << "\t";
    _fp << data.getCoulombEnergy() << "\t";
    _fp << data.getNonCoulombEnergy() << "\t";
    _fp << 0.0 << "\t";
    _fp << 0.0 << "\t";

    _fp << scientific;
    _fp << data.getMomentum() << "\t";

    _fp << fixed;
    _fp << 0.0 << "\t";
    _fp << endl;
}