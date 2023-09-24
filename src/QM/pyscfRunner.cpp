#include "pyscfRunner.hpp"

#include "qmSettings.hpp"        // for QMSettings
#include "simulationBox.hpp"     // for SimulationBox
#include "stringUtilities.hpp"   // for fileExists

#include <format>    // for format
#include <fstream>   // for operator<<, basic_ostream, endl, ostream

using QM::PySCFRunner;

/**
 * @brief writes the coords file in order to run the external qm program
 *
 * @param box
 */
void PySCFRunner::writeCoordsFile(simulationBox::SimulationBox &box)
{
    const std::string fileName = "coords.xyz";
    std::ofstream     coordsFile(fileName);

    coordsFile << box.getNumberOfQMAtoms() << "\n\n";

    for (size_t i = 0, numberOfAtoms = box.getNumberOfQMAtoms(); i < numberOfAtoms; ++i)
    {
        const auto &atom = box.getQMAtom(i);

        coordsFile << std::format("{:5s}\t{:16.12f}\t{:16.12f}\t{:16.12f}\n",
                                  atom.getName(),
                                  atom.getPosition()[0],
                                  atom.getPosition()[1],
                                  atom.getPosition()[2]);
    }

    coordsFile.close();
}

/**
 * @brief executes the qm script of the external program
 *
 */
void PySCFRunner::execute()
{
    const auto scriptFileName = _scriptPath + settings::QMSettings::getQMScript();

    if (!utilities::fileExists(scriptFileName))
        throw customException::InputFileException(std::format("PySCF script file \"{}\" does not exist.", scriptFileName));

    const auto command = std::format("python {} > pyscf.out", scriptFileName);
    ::system(command.c_str());
}