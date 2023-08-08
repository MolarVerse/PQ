#include "rstFileOutput.hpp"

#include <iomanip>

using namespace output;
using namespace std;
using namespace simulationBox;

/**
 * @brief Write the restart file
 *
 * @param simBox
 * @param step
 */
void RstFileOutput::write(SimulationBox &simBox, const size_t step)
{
    _fp.close();

    _fp.open(_filename);

    _fp << "Step " << step << endl;

    _fp << "Box ";
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << endl;
    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();
        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << left;
            _fp << setw(5);
            _fp << i + 1;

            _fp << left;
            _fp << setw(5);
            _fp << molecule.getMoltype();

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[2];

            _fp << scientific;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[0];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[1];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[2];

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[2];

            _fp << endl;
        }
    }
}