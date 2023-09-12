#include "dftbplusRunner.hpp"

#include "simulationBox.hpp"   // for SimulationBox

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

    for (size_t i, numberOfAtoms = box.getNumberOfQMAtoms(); i < numberOfAtoms; ++i)
    {
        const auto &atom = box.getQMAtom(i);

        const auto iter   = std::ranges::find(uniqueAtomNames, atom.getName());
        const auto atomId = std::ranges::distance(uniqueAtomNames.begin(), iter) + 1;

        coordsFile << i << "  " << atomId << "  " << atom.getPosition()[0] << "  " << atom.getPosition()[1] << "  "
                   << atom.getPosition()[2] << "\n";
    }

    coordsFile << "0.0 0.0 0.0\n";
    coordsFile << box.getBoxDimensions()[0] << " 0.0 0.0\n";
    coordsFile << "0.0 " << box.getBoxDimensions()[1] << " 0.0\n";
    coordsFile << "0.0 0.0 " << box.getBoxDimensions()[2] << "\n";

    coordsFile.close();
}