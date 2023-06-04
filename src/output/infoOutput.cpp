#include "infoOutput.hpp"

#include <iomanip>

using namespace std;

void InfoOutput::write(const double simulationTime, const PhysicalData &data)
{
    _fp.close();

    _fp.open(_filename);

    _fp << "-----------------------------------------------" << endl;
    _fp << "|                                             |" << endl;
    _fp << "-----------------------------------------------" << endl;

    _fp << std::setw(8) << std::setfill(' ');
    _fp << "|    ";
    _fp << "SIMULATION TIME " << simulationTime << " ps";
    _fp << "\t\t";
    _fp << "TEMPERATURE " << data.getTemperature() << " K";
    _fp << "    |";

    _fp << endl;

    _fp << std::setw(8) << std::setfill(' ');
    _fp << "|    ";
    _fp << "PRESSURE    " << data.getPressure() << " bar";
    _fp << "\t\t";
    _fp << "TEMPERATURE " << data.getTemperature() << " K";
    _fp << "    |";

    _fp << endl;
}