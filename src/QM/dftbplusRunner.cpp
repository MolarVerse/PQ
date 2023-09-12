#include "dftbplusRunner.hpp"

#include "simulationBox.hpp"   // for SimulationBox

#include <format>    // for format
#include <fstream>   // for ofstream
#include <string>    // for string

using QM::DFTBPlusRunner;

void DFTBPlusRunner::writeCoordsFile(simulationBox::SimulationBox &box)
{
    const std::string fileName = "coords";
    std::ofstream     coordsFile(fileName);

    coordsFile << box.getNumberOfQMAtoms() << "  S\n";

    const auto uniqueAtomNames = box.getUniqueQMAtomNames();

    for (const auto &atomName : uniqueAtomNames)
        coordsFile << atomName << "  ";

    coordsFile << "\n";

    for (size_t i = 0, numberOfAtoms = box.getNumberOfQMAtoms(); i < numberOfAtoms; ++i)
    {
        const auto &atom = box.getQMAtom(i);

        const auto iter   = std::ranges::find(uniqueAtomNames, atom.getName());
        const auto atomId = std::ranges::distance(uniqueAtomNames.begin(), iter) + 1;

        coordsFile << std::format("{:5d} {:5d}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                                  i + 1,
                                  atomId,
                                  atom.getPosition()[0],
                                  atom.getPosition()[1],
                                  atom.getPosition()[2]);
    }

    coordsFile << std::format("{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n", "", 0.0, 0.0, 0.0);
    coordsFile << std::format("{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n", "", box.getBoxDimensions()[0], 0.0, 0.0);
    coordsFile << std::format("{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n", "", 0.0, box.getBoxDimensions()[1], 0.0);
    coordsFile << std::format("{:11}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n", "", 0.0, 0.0, box.getBoxDimensions()[2]);

    coordsFile.close();
}

void DFTBPlusRunner::execute() {}