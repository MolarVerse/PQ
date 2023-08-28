#include "intraNonBondedReader.hpp"

#include "exceptions.hpp"                // for IntraNonBondedException
#include "intraNonBonded.hpp"            // for IntraNonBonded
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "mathUtilities.hpp"             // for sign, utilities
#include "molecule.hpp"                  // for Molecule
#include "settings.hpp"                  // for Settings
#include "simulationBox.hpp"             // for SimulationBox
#include "stringUtilities.hpp"           // for removeComments, splitString

#include <algorithm>     // for for_each
#include <cstdlib>       // for abs, size_t
#include <filesystem>    // for exists
#include <format>        // for format
#include <istream>       // for basic_istream, ifstream, std
#include <optional>      // for operator==, optional, nullopt
#include <ranges>        // for drop
#include <string_view>   // for string_view
#include <vector>        // for vector

using namespace readInput::intraNonBonded;

/**
 * @brief construct IntraNonBondedReader object and read the file
 *
 * @param engine
 */
void readInput::intraNonBonded::readIntraNonBondedFile(engine::Engine &engine)
{
    IntraNonBondedReader reader(engine.getSettings().getIntraNonBondedFilename(), engine);
    reader.read();
}

/**
 * @brief reads the intra non bonded interactions from the intraNonBonded file
 *
 * @details The function reads the intra non bonded interactions from the intraNonBonded file. It calls the processMolecule
 * function if a molecule type is found. The molecule type can be given either via the string name or the size_t molecule type.
 *
 * @throws customException::IntraNonBondedException if the intraNonBonded file is not provided by the user
 * @throws customException::IntraNonBondedException if the intraNonBonded file does not exist
 * @throws customException::IntraNonBondedException if the molecule type is not found
 */
void IntraNonBondedReader::read()
{
    if (!isNeeded())
        return;

    if (_fileName.empty())
        throw customException::IntraNonBondedException("Intra non bonded file needed for requested simulation setup");

    if (!std::filesystem::exists(_fileName))
        throw customException::IntraNonBondedException("Intra non bonded file \"" + _fileName + "\"" + " File not found");

    std::string line;

    while (getline(_fp, line))
    {
        line              = utilities::removeComments(line, "#");
        auto lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        const auto moleculeType = findMoleculeType(lineElements[0]);

        processMolecule(moleculeType);

        ++_lineNumber;
    }

    checkDuplicates();
}

/**
 * @brief finds the molecule type either by string or by size_t
 *
 * @param identifier
 * @return size_t
 *
 * @throws customException::IntraNonBondedException if the molecule type is not found
 */
[[nodiscard]] size_t IntraNonBondedReader::findMoleculeType(const std::string &identifier) const
{
    auto moleculeTypeFromString = _engine.getSimulationBox().findMoleculeTypeByString(identifier);

    if (moleculeTypeFromString == std::nullopt)
    {
        auto moleculeTypeFromSizeT = size_t{};
        try
        {
            moleculeTypeFromSizeT = stoul(identifier);
        }
        catch (...)
        {
            throw customException::IntraNonBondedException(format(
                R"(ERROR: could not find molecule type "{}" in line {} in file "{}")", identifier, _lineNumber, _fileName));
        }

        const bool moleculeTypeExists = _engine.getSimulationBox().moleculeTypeExists(moleculeTypeFromSizeT);

        if (moleculeTypeExists)
            return moleculeTypeFromSizeT;
        else
            throw customException::IntraNonBondedException(format(
                R"(ERROR: could not find molecule type "{}" in line {} in file "{}")", identifier, _lineNumber, _fileName));
    }
    else
        return moleculeTypeFromString.value();
}

/**
 * @brief processes the intra nonBonded interactions for a given molecule type
 *
 * @details the atomIndices vector is a vector of vectors. The first index is the reference atom index. The second index is the
 * atom index that interacts with the reference atom. The sign of the atom index indicates the type of interaction. If the sign is
 * negative then the interaction is a 1-4 interaction and has to be scaled accordingly.
 *
 * Each line should have the following format:
 * <reference atom index> <atom index 1> <atom index 2> ... (negative atom index means 1-4 interaction)
 *
 * The molecule section should end with "END" (case insensitive)
 *
 * @param moleculeType
 *
 * @throws customException::IntraNonBondedException if the reference atom index is out of range
 * @throws customException::IntraNonBondedException if the abs(atom index) is out of range
 * @throws customException::IntraNonBondedException if "END" is not found
 */
void IntraNonBondedReader::processMolecule(const size_t moleculeType)
{
    std::string line;
    auto        endedNormal = false;

    const auto                    numberOfAtoms = _engine.getSimulationBox().findMoleculeType(moleculeType).getNumberOfAtoms();
    std::vector<std::vector<int>> atomIndices(numberOfAtoms, std::vector<int>(0));

    ++_lineNumber;

    while (getline(_fp, line))
    {
        line                    = utilities::removeComments(line, "#");
        const auto lineElements = utilities::splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (utilities::toLowerCopy(lineElements[0]) == "end")
        {
            endedNormal = true;
            break;
        }

        const auto referenceAtomIndex = size_t(stoi(lineElements[0]) - 1);

        if (referenceAtomIndex >= numberOfAtoms)
            throw customException::IntraNonBondedException(
                format(R"(ERROR: reference atom index "{}" in line {} in file "{}" is out of range)",
                       lineElements[0],
                       _lineNumber,
                       _fileName));

        auto addAtomIndexToReferenceAtom = [&atomIndices, referenceAtomIndex, numberOfAtoms, this](const auto &lineElement)
        {
            const auto atomIndex = (::abs(stoi(lineElement)) - 1) * utilities::sign(stoi(lineElement));

            if (::abs(atomIndex) >= int(numberOfAtoms))
                throw customException::IntraNonBondedException(format(
                    R"(ERROR: atom index "{}" in line {} in file "{}" is out of range)", lineElement, _lineNumber, _fileName));

            atomIndices[referenceAtomIndex].push_back(atomIndex);
        };

        std::ranges::for_each(lineElements | std::views::drop(1), addAtomIndexToReferenceAtom);

        ++_lineNumber;
    }

    if (!endedNormal)
        throw customException::IntraNonBondedException(
            format(R"(ERROR: could not find "END" for moltype "{}" in file "{}")", moleculeType, _fileName));

    _engine.getIntraNonBonded().addIntraNonBondedContainer(::intraNonBonded::IntraNonBondedContainer(moleculeType, atomIndices));
}

/**
 * @brief checks if a molecule type is defined multiple times
 *
 * @throws customException::IntraNonBondedException if a molecule type is defined multiple times
 *
 */
void IntraNonBondedReader::checkDuplicates() const
{
    const auto nonBondedContainers = _engine.getIntraNonBonded().getIntraNonBondedContainers();

    auto moleculeTypesView =
        nonBondedContainers | std::views::transform([](const auto &container) { return container.getMolType(); });

    std::vector<size_t> moleculeTypes(moleculeTypesView.begin(), moleculeTypesView.end());
    std::ranges::sort(moleculeTypes);
    const auto it = std::ranges::adjacent_find(moleculeTypes);

    if (it != moleculeTypes.end())
        throw customException::IntraNonBondedException(
            format(R"(ERROR: moltype "{}" is defined multiple times in file "{}")", *it, _fileName));
}