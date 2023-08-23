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
 * @TODO: error handling for filename
 *
 */
void IntraNonBondedReader::read()
{
    if (!isNeeded())
        return;

    if (_filename.empty())
        throw customException::IntraNonBondedException("Intra non bonded file needed for requested simulation setup");

    if (!filesystem::exists(_filename))
        throw customException::IntraNonBondedException("Intra non bonded file \"" + _filename + "\"" + " File not found");

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

        processMolecule(moleculeType);
    }
}

/**
 * @brief processes the intra nonBonded interactions for a given molecule type
 *
 * @details the atomIndices vector is a vector of vectors. The first index is the reference atom index. The second index is the
 * atom index that interacts with the reference atom. The sign of the atom index indicates the type of interaction. If the sign is
 * negative then the interaction is a 1-4 interaction and has to be scaled accordingly.
 *
 * @param moleculeType
 */
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

        const auto referenceAtomIndex = size_t(stoi(lineElements[0]) - 1);

        if (referenceAtomIndex >= numberOfAtoms)
            throw customException::IntraNonBondedException(
                format(R"(ERROR: reference atom index "{}" in line {} in file {} is out of range)",
                       lineElements[0],
                       _lineNumber,
                       _filename));

        for (size_t i = 1; i < lineElements.size(); ++i)
        {
            const auto atomIndex = (::abs(stoi(lineElements[i])) - 1) * utilities::sign(stoi(lineElements[i]));

            if (::abs(atomIndex) >= int(numberOfAtoms))
                throw customException::IntraNonBondedException(format(
                    R"(ERROR: atom index "{}" in line {} in file {} is out of range)", lineElements[i], _lineNumber, _filename));

            atomIndices[referenceAtomIndex].push_back(atomIndex);
        }

        ++_lineNumber;
    }

    if (!endedNormal)
        throw customException::IntraNonBondedException(
            format(R"(ERROR: could not find "END" in line {} in file {})", _lineNumber, _filename));

    _engine.getIntraNonBonded().addIntraNonBondedContainer(intraNonBonded::IntraNonBondedContainer(moleculeType, atomIndices));
}