#include "trajectoryOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <cstddef>   // for size_t
#include <iomanip>
#include <ostream>   // for ofstream, basic_ostream, operator<<
#include <vector>    // for vector

using namespace std;
using namespace simulationBox;
using namespace output;

/**
 * @brief Write xyz file
 *
 * @param simBox
 */
void TrajectoryOutput::writeXyz(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << '\n';

    _fp << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomPosition(i)[2];

            _fp << '\n';
        }
    }
}

/**
 * @brief Write velocities file
 *
 * @param simBox
 */
void TrajectoryOutput::writeVelocities(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << '\n';

    _fp << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << scientific;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[0];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[1];

            _fp << setw(20);
            _fp << molecule.getAtomVelocity(i)[2];

            _fp << '\n';
        }
    }
}

/**
 * @brief Write forces file
 *
 * @param simBox
 */
void TrajectoryOutput::writeForces(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << '\n';

    _fp << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);
            _fp << right;

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[0];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[1];

            _fp << setw(15);
            _fp << molecule.getAtomForce(i)[2];

            _fp << '\n';
        }
    }
}

/**
 * @brief Write charges file
 *
 * @param simBox
 */
void TrajectoryOutput::writeCharges(SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms();
    _fp << "  ";
    _fp << simBox.getBoxDimensions();
    _fp << "  ";
    _fp << simBox.getBoxAngles();
    _fp << '\n';

    _fp << '\n';

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << left;
            _fp << setw(5);
            _fp << molecule.getAtomName(i);

            _fp << fixed;
            _fp << setprecision(8);

            _fp << setw(15);
            _fp << right;
            _fp << molecule.getPartialCharge(i);

            _fp << '\n';
        }
    }
}