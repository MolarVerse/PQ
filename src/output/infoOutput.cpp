#include "infoOutput.hpp"

#include "physicalData.hpp"   // for PhysicalData

#include <iomanip>
#include <ostream>   // for operator<<, basic_ostream, ostream, endl

using namespace std;
using namespace physicalData;
using namespace output;

/**
 * @brief write info file
 *
 * @param simulationTime
 * @param data
 */
void InfoOutput::write(const double simulationTime, const PhysicalData &data)
{
    _fp.close();

    _fp.open(_fileName);

    writeHeader();

    writeLeft(simulationTime, "SIMULATION TIME", "ps", fixed, 5);
    writeRight(data.getTemperature(), "TEMPERATURE", "K", fixed, 5);

    writeLeft(data.getPressure(), "PRESSURE", "bar", fixed, 5);
    writeRight(0.0, "E(TOT)", "kcal/mol", fixed, 5);

    writeLeft(data.getKineticEnergy(), "E(KIN)", "kcal/mol", fixed, 5);
    writeRight(0.0, "E(INTRA)", "kcal/mol", fixed, 5);

    writeLeft(data.getCoulombEnergy(), "E(COUL)", "kcal/mol", fixed, 5);
    writeRight(data.getNonCoulombEnergy(), "E(NON-COUL)", "kcal/mol", fixed, 5);

    writeLeft(data.getMomentum(), "MOMENTUM", "amuA/fs", scientific, 1);
    writeRight(0.0, "LOOPTIME", "s", fixed, 5);

    _fp << setw(86);
    _fp << setfill('-');
    _fp << right;
    _fp << "\n"
        << "\n";

    _fp.flush();
}

/**
 * @brief write header of info file
 *
 */
void InfoOutput::writeHeader()
{
    _fp << setw(86);
    _fp << setfill('-');
    _fp << "\n";

    _fp << setfill(' ');
    _fp << "|";
    _fp << setw(51);
    _fp << "PIMD-QMCF info file";
    _fp << setw(34);
    _fp << right << "|";
    _fp << "\n";

    _fp << setfill('-');
    _fp << setw(86);
    _fp << "\n";

    _fp << setfill(' ');
}

/**
 * @brief write left column of info file
 *
 * @param value
 * @param name
 * @param unit
 * @param formatter
 * @param precision
 */
void InfoOutput::writeLeft(const double       value,
                           const string_view &name,
                           const string_view &unit,
                           ios_base &(*formatter)(ios_base &),
                           const size_t precision)
{
    _fp << "|   ";
    _fp << left << setw(16) << name;
    _fp << right << setw(15) << formatter << setprecision(precision) << value;
    _fp << " " << left << setw(9) << unit;
}

/**
 * @brief write right column of info file
 *
 * @param value
 * @param name
 * @param unit
 * @param formatter
 * @param precision
 */
void InfoOutput::writeRight(const double       value,
                            const string_view &name,
                            const string_view &unit,
                            ios_base &(*formatter)(ios_base &),
                            const size_t precision)
{
    _fp << left << setw(12) << name;
    _fp << right << setw(15) << setfill(' ') << formatter << setprecision(precision) << value;
    _fp << " " << left << setw(9) << unit;
    _fp << "   |";

    _fp << "\n";
}
