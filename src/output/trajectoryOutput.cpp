#include "trajectoryOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <cstddef>   // for size_t
#include <iomanip>
#include <ostream>   // for ofstream, basic_ostream, operator<<
#include <string>    // for operator<<
#include <vector>    // for vector

using namespace output;

/**
 * @brief Write the header of a trajectory files
 *
 * @param simBox
 */
void TrajectoryOutput::writeHeader(const simulationBox::SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms() << "  " << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << "\n\n";
}

/**
 * @brief Write xyz file
 *
 * @param simBox
 */
void TrajectoryOutput::writeXyz(simulationBox::SimulationBox &simBox)
{
    writeHeader(simBox);

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << std::left;
            _fp << std::setw(5);
            _fp << molecule.getAtomName(i);

            _fp << std::fixed;
            _fp << std::setprecision(8);
            _fp << std::right;

            _fp << std::setw(15);
            _fp << molecule.getAtomPosition(i)[0];

            _fp << std::setw(15);
            _fp << molecule.getAtomPosition(i)[1];

            _fp << std::setw(15);
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
void TrajectoryOutput::writeVelocities(simulationBox::SimulationBox &simBox)
{
    writeHeader(simBox);

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << std::left;
            _fp << std::setw(5);
            _fp << molecule.getAtomName(i);

            _fp << std::scientific;
            _fp << std::setprecision(8);
            _fp << std::right;

            _fp << std::setw(20);
            _fp << molecule.getAtomVelocity(i)[0];

            _fp << std::setw(20);
            _fp << molecule.getAtomVelocity(i)[1];

            _fp << std::setw(20);
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
void TrajectoryOutput::writeForces(simulationBox::SimulationBox &simBox)
{
    writeHeader(simBox);

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << std::left;
            _fp << std::setw(5);
            _fp << molecule.getAtomName(i);

            _fp << std::fixed;
            _fp << std::setprecision(8);
            _fp << std::right;

            _fp << std::setw(15);
            _fp << molecule.getAtomForce(i)[0];

            _fp << std::setw(15);
            _fp << molecule.getAtomForce(i)[1];

            _fp << std::setw(15);
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
void TrajectoryOutput::writeCharges(simulationBox::SimulationBox &simBox)
{
    writeHeader(simBox);

    for (const auto &molecule : simBox.getMolecules())
    {
        const auto numberOfAtoms = molecule.getNumberOfAtoms();

        for (size_t i = 0; i < numberOfAtoms; ++i)
        {
            _fp << std::left;
            _fp << std::setw(5);
            _fp << molecule.getAtomName(i);

            _fp << std::fixed;
            _fp << std::setprecision(8);
            _fp << std::right;

            _fp << std::setw(15);
            _fp << std::right;
            _fp << molecule.getPartialCharge(i);

            _fp << '\n';
        }
    }
}