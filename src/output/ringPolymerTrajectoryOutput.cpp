#include "ringPolymerTrajectoryOutput.hpp"

#include "molecule.hpp"              // for Molecule
#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox
#include "vector3d.hpp"              // for Vec3D, operator<<

#include <algorithm>    // for __for_each_fn, for_each
#include <format>       // for format
#include <functional>   // for identity
#include <ostream>      // for basic_ostream, ofstream, operator<<
#include <stddef.h>     // for size_t
#include <string>       // for operator<<, char_traits

using output::RingPolymerTrajectoryOutput;

/**
 * @brief write the header of the beads trajectory file
 *
 * @details number of atoms is multiplied by the number of beads - box dimensions and angles are the same for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeHeader(const simulationBox::SimulationBox &simBox)
{
    _fp << simBox.getNumberOfAtoms() * settings::RingPolymerSettings::getNumberOfBeads() << "  ";
    _fp << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << '\n';
}

/**
 * @brief write the xyz file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeXyz(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(j)[0]);
                _fp << std::format("{:15.8f}\t", molecule.getAtomPosition(j)[1]);
                _fp << std::format("{:15.8f}\n", molecule.getAtomPosition(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the velocity file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeVelocities(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(j)[0]);
                _fp << std::format("{:20.8e}\t", molecule.getAtomVelocity(j)[1]);
                _fp << std::format("{:20.8e}\n", molecule.getAtomVelocity(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the force file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeForces(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);

    auto totalForce = 0.0;
    std::ranges::for_each(beads, [&totalForce](auto &bead) { totalForce += bead.calculateTotalForce(); });

    _fp << std::format("# Total force = {:.5e} kcal/mol/Angstrom\n", totalForce);

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\t", molecule.getAtomForce(j)[0]);
                _fp << std::format("{:15.8f}\t", molecule.getAtomForce(j)[1]);
                _fp << std::format("{:15.8f}\n", molecule.getAtomForce(j)[2]);

                _fp << std::flush;
            }
}

/**
 * @brief write the charge file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeCharges(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);
    _fp << '\n';

    for (size_t i = 0; i < settings::RingPolymerSettings::getNumberOfBeads(); ++i)
        for (const auto &molecule : beads[i].getMolecules())
            for (size_t j = 0, numberOfAtoms = molecule.getNumberOfAtoms(); j < numberOfAtoms; ++j)
            {
                _fp << std::format("{:>5}{}\t", molecule.getAtomName(j), i + 1);

                _fp << std::format("{:15.8f}\n", molecule.getPartialCharge(j));

                _fp << std::flush;
            }
}