#include "trajectoryOutput.hpp"

#include "molecule.hpp"        // for Molecule
#include "simulationBox.hpp"   // for SimulationBox
#include "vector3d.hpp"        // for Vec3D

#include <cstddef>   // for size_t
#include <format>    // for format
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
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << std::format("{:<5}\t", molecule.getAtomName(i));

            _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(i)[0]);
            _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(i)[1]);
            _fp << std::format("{:15.8f}\n", molecule.getAtomPosition(i)[2]);

            _fp << std::flush;
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
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << std::format("{:<5}\t", molecule.getAtomName(i));

            _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(i)[0]);
            _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(i)[1]);
            _fp << std::format("{:20.8e}\n", molecule.getAtomVelocity(i)[2]);

            _fp << std::flush;
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
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << std::format("{:<5}\t", molecule.getAtomName(i));

            _fp << std::format("{:15.8f}\t", molecule.getAtomForce(i)[0]);
            _fp << std::format("{:15.8f}\t", molecule.getAtomForce(i)[1]);
            _fp << std::format("{:15.8f}\n", molecule.getAtomForce(i)[2]);

            _fp << std::flush;
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
        for (size_t i = 0, numberOfAtoms = molecule.getNumberOfAtoms(); i < numberOfAtoms; ++i)
        {
            _fp << std::format("{:<5}\t", molecule.getAtomName(i));

            _fp << std::format("{:15.8f}\n", molecule.getPartialCharge(i));

            _fp << std::flush;
        }
}