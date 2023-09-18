#include "ringPolymerTrajectoryOutput.hpp"

#include "ringPolymerSettings.hpp"   // for RingPolymerSettings
#include "simulationBox.hpp"         // for SimulationBox

#include <format>   // for format

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
    _fp << simBox.getBoxDimensions() << "  " << simBox.getBoxAngles() << "\n\n";
}

/**
 * @brief write the xyz file for all beads
 *
 * @param beads
 */
void RingPolymerTrajectoryOutput::writeXyz(std::vector<simulationBox::SimulationBox> &beads)
{
    writeHeader(beads[0]);

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