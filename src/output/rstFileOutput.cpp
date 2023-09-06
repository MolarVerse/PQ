#include "rstFileOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for operator<<

#include <format>    // for format
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

    _fp << "Box   " << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << '\n' << flush;

    for (const auto &molecule : simBox.getMolecules())
    {
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << format("{:<5}\t", molecule.getAtomName(i));
            _fp << format("{:<5}\t", i + 1);
            _fp << format("{:<5}\t", molecule.getMoltype());

            _fp << format("{:15.8f}\t{:15.8f}\t{:15.8f}\t",
                          molecule.getAtomPosition(i)[0],
                          molecule.getAtomPosition(i)[1],
                          molecule.getAtomPosition(i)[2]);

            _fp << format("{:19.8e}\t{:19.8e}\t{:19.8e}\t",
                          molecule.getAtomVelocity(i)[0],
                          molecule.getAtomVelocity(i)[1],
                          molecule.getAtomVelocity(i)[2]);

            _fp << format("{:15.8f}\t{:15.8f}\t{:15.8f}",
                          molecule.getAtomForce(i)[0],
                          molecule.getAtomForce(i)[1],
                          molecule.getAtomForce(i)[2]);

            _fp << '\n' << flush;
        }
    }
}