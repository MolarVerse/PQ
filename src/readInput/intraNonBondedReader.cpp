#include "intraNonBondedReader.hpp"

#include "stringUtilities.hpp"

using namespace readInput;
using namespace std;
using namespace utilities;

/**
 * @brief construct IntraNonBondedReader object and read the file
 *
 * @param engine
 */
void readInput::readIntraNonBondedFile(engine::Engine &engine)
{
    IntraNonBondedReader reader(engine.getSettings().getIntraNonBondedFilename(), engine);
    reader.read();
}

/**
 * @brief reads the intra non bonded interactions from the intraNonBonded file
 *
 */
void IntraNonBondedReader::read()
{
    if (!isNeeded())
        return;

    string         line;
    vector<string> lineElements;

    _lineNumber = 0;

    while (getline(_fp, line))
    {
        line         = removeComments(line, "#");
        lineElements = splitString(line);

        ++_lineNumber;

        if (lineElements.empty())
            continue;

        optional<size_t> moleculeTypeFromString = _engine.getSimulationBox().findMoleculeTypeByString(lineElements[0]);
        size_t           moleculeType           = 0;

        if (moleculeTypeFromString == nullopt)
        {
            try
            {
                const auto moleculeTypeFromSizeT = stoul(lineElements[0]);
                const bool moleculeTypeExists    = _engine.getSimulationBox().moleculeTypeExists(moleculeTypeFromSizeT);

                if (moleculeTypeExists)
                    moleculeType = moleculeTypeFromSizeT;
                else
                    throw;
            }
            catch (...)
            {
                throw customException::IntraNonBondedException(
                    format(R"(ERROR: could not find molecule type "{}" in line {} in file {})",
                           lineElements[0],
                           _lineNumber,
                           _filename));
            }
        }
        else
            moleculeType = moleculeTypeFromString.value();
    }
}

void IntraNonBondedReader::processMolecule(const size_t moleculeType)
{
    string line;
    auto   endedNormal = false;

    const auto          numberOfAtoms = _engine.getSimulationBox().findMoleculeType(moleculeType).getNumberOfAtoms();
    vector<vector<int>> atomIndices(numberOfAtoms, vector<int>(0));

    while (getline(_fp, line))
    {
        line                    = utilities::removeComments(line, "#");
        const auto lineElements = utilities::splitString(line);

        ++_lineNumber;
        if (lineElements.empty())
        {
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
        {
            ++_lineNumber;
            endedNormal = true;
            break;
        }

        ++_lineNumber;
    }

    if (!endedNormal)
        throw customException::IntraNonBondedException(
            format(R"(ERROR: could not find "END" in line {} in file {})", _lineNumber, _filename));

    _engine.getIntraNonBonded().addIntraNonBondedContainer(intraNonBonded::IntraNonBondedContainer(moleculeType, atomIndices));
}