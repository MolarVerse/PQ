#include "rstFileOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for operator<<

#include <iomanip>   // for operator<<, setw, setprecision, left
#include <ostream>   // for basic_ostream, operator<<, flush, std
#include <string>    // for char_traits, operator<<
#include <vector>    // for vector

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

    _fp.open(_fileName);

    _fp << "Step " << step << '\n' << flush;

    _fp << "Box ";
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << '\n' << flush;
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

            _fp << '\n' << flush;
        }
    }
}