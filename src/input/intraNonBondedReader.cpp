/*****************************************************************************
<GPL_HEADER>

    PQ
    Copyright (C) 2023-now  Jakob Gamper

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

<GPL_HEADER>
******************************************************************************/

#include "intraNonBondedReader.hpp"

#include <algorithm>     // for for_each
#include <cstdlib>       // for abs, size_t
#include <filesystem>    // for exists
#include <format>        // for format
#include <istream>       // for basic_istream, ifstream, std
#include <optional>      // for operator==, optional, nullopt
#include <ranges>        // for drop
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "engine.hpp"                    // for Engine
#include "exceptions.hpp"                // for IntraNonBondedException
#include "fileSettings.hpp"              // for FileSettings
#include "intraNonBonded.hpp"            // for IntraNonBonded
#include "intraNonBondedContainer.hpp"   // for IntraNonBondedContainer
#include "mathUtilities.hpp"             // for sign, utilities
#include "molecule.hpp"                  // for Molecule
#include "settings.hpp"                  // for Settings
#include "simulationBox.hpp"             // for SimulationBox
#include "stringUtilities.hpp"           // for removeComments, splitString

using namespace input::intraNonBondedReader;
using namespace engine;
using namespace settings;
using namespace customException;
using namespace utilities;
using namespace intraNonBonded;

using std::views::drop;

/**
 * @brief checks if the intra non bonded interactions are needed
 *
 * @param engine
 * @return bool
 */
bool input::intraNonBondedReader::isNeeded(const Engine &engine)
{
    return engine.isIntraNonBondedActivated();
}

/**
 * @brief construct IntraNonBondedReader object and read the file
 *
 * @param engine
 */
void input::intraNonBondedReader::readIntraNonBondedFile(Engine &engine)
{
    if (!isNeeded(engine))
        return;

    const auto &stdOut = engine.getStdoutOutput();
    auto       &log    = engine.getLogOutput();

    const auto filename = FileSettings::getIntraNonBondedFileName();

    stdOut.writeRead("Intra Non-Bonded File", filename);
    log.writeRead("Intra Non-Bonded File", filename);

    IntraNonBondedReader reader(filename, engine);
    reader.read();
}

/**
 * @brief Construct a new Intra Non Bonded Reader:: Intra Non Bonded Reader
 * object
 *
 * @param fileName
 * @param engine
 */
IntraNonBondedReader::IntraNonBondedReader(
    const std::string &fileName,
    Engine            &engine
)
    : _fileName(fileName), _fp(fileName), _engine(engine){};

/**
 * @brief reads the intra non bonded interactions from the intraNonBonded file
 *
 * @details The function reads the intra non bonded interactions from the
 * intraNonBonded file. It calls the processMolecule function if a molecule type
 * is found. The molecule type can be given either via the string name or the
 * size_t molecule type.
 *
 * @throws IntraNonBondedException if the intraNonBonded file
 * is not provided by the user
 * @throws IntraNonBondedException if the intraNonBonded file
 * does not exist
 * @throws IntraNonBondedException if the molecule type is not
 * found
 */
void IntraNonBondedReader::read()
{
    if (!FileSettings::isIntraNonBondedFileNameSet())
        throw IntraNonBondedException(
            "Intra non bonded file needed for requested simulation setup"
        );

    std::string line;

    while (getline(_fp, line))
    {
        line              = removeComments(line, "#");
        auto lineElements = splitString(line);

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
 * @param id
 * @return size_t
 *
 * @throws IntraNonBondedException if the molecule type is not
 * found
 */
size_t IntraNonBondedReader::findMoleculeType(const std::string &id) const
{
    auto &simBox            = _engine.getSimulationBox();
    auto  molTypeFromString = simBox.findMoleculeTypeByString(id);

    if (molTypeFromString == std::nullopt)
    {
        auto molTypeFromSizeT = size_t{};

        try
        {
            molTypeFromSizeT = stoul(id);
        }
        catch (...)
        {
            throw IntraNonBondedException(format(
                "ERROR: could not find molecule type '{}' in line {} in file "
                "'{}'",
                id,
                _lineNumber,
                _fileName
            ));
        }

        const bool molTypeExists = simBox.moleculeTypeExists(molTypeFromSizeT);

        if (molTypeExists)
            return molTypeFromSizeT;
        else
            throw IntraNonBondedException(format(
                "ERROR: could not find molecule type '{}' in line {} in file "
                "'{}'",
                id,
                _lineNumber,
                _fileName
            ));
    }
    else
        return molTypeFromString.value();
}

/**
 * @brief processes the intra nonBonded interactions for a given molecule type
 *
 * @details the atomIndices vector is a vector of vectors. The first index is
 * the reference atom index. The second index is the atom index that interacts
 * with the reference atom. The sign of the atom index indicates the type of
 * interaction. If the sign is negative then the interaction is a 1-4
 * interaction and has to be scaled accordingly.
 *
 * Each line should have the following format:
 * <reference atom index> <atom index 1> <atom index 2> ... (negative atom index
 * means 1-4 interaction)
 *
 * The molecule section should end with "END" (case insensitive)
 *
 * @param moleculeType
 *
 * @throws IntraNonBondedException if the reference atom index
 * is out of range
 * @throws IntraNonBondedException if the abs(atom index) is
 * out of range
 * @throws IntraNonBondedException if "END" is not found
 */
void IntraNonBondedReader::processMolecule(const size_t moleculeType)
{
    std::string line;
    auto        endedNormal = false;

    auto &molType = _engine.getSimulationBox().findMoleculeType(moleculeType);

    const auto nAtoms = molType.getNumberOfAtoms();

    std::vector<std::vector<int>> atomIndices(nAtoms, std::vector<int>(0));

    ++_lineNumber;

    while (getline(_fp, line))
    {
        line                    = removeComments(line, "#");
        const auto lineElements = splitString(line);

        if (lineElements.empty())
        {
            ++_lineNumber;
            continue;
        }

        if (toLowerCopy(lineElements[0]) == "end")
        {
            endedNormal = true;
            break;
        }

        const auto refAtomIdx = size_t(stoi(lineElements[0]) - 1);

        if (refAtomIdx >= nAtoms)
            throw IntraNonBondedException(format(
                "ERROR: reference atom index '{}' in line {} in file '{}' is "
                "out of range",
                lineElements[0],
                _lineNumber,
                _fileName
            ));

        auto addAtomIdxToRefAtom =
            [&atomIndices, refAtomIdx, nAtoms, this](const auto &lineElement)
        {
            auto atomIndex  = ::abs(stoi(lineElement)) - 1;
            atomIndex      *= sign(stoi(lineElement));

            if (::abs(atomIndex) >= int(nAtoms))
                throw IntraNonBondedException(format(
                    "ERROR: atom index '{}' in line {} in file '{}' is out of "
                    "range",
                    lineElement,
                    _lineNumber,
                    _fileName
                ));

            atomIndices[refAtomIdx].push_back(atomIndex);
        };

        std::ranges::for_each(lineElements | drop(1), addAtomIdxToRefAtom);

        ++_lineNumber;
    }

    if (!endedNormal)
        throw IntraNonBondedException(format(
            "ERROR: could not find 'END' for moltype '{}' in file '{}'",
            moleculeType,
            _fileName
        ));

    const auto container = IntraNonBondedContainer(moleculeType, atomIndices);

    _engine.getIntraNonBonded().addIntraNonBondedContainer(container);
}

/**
 * @brief checks if a molecule type is defined multiple times
 *
 * @throws IntraNonBondedException if a molecule type is
 * defined multiple times
 *
 */
void IntraNonBondedReader::checkDuplicates() const
{
    auto      &intraNonBonded = _engine.getIntraNonBonded();
    const auto nonBondedCont  = intraNonBonded.getIntraNonBondedContainers();

    auto transform = [](const auto &container)
    { return container.getMolType(); };

    auto moleculeTypesView = nonBondedCont | std::views::transform(transform);

    const auto start = moleculeTypesView.begin();
    const auto end   = moleculeTypesView.end();

    std::vector<size_t> moleculeTypes(start, end);
    std::ranges::sort(moleculeTypes);
    const auto it = std::ranges::adjacent_find(moleculeTypes);

    if (it != moleculeTypes.end())
        throw IntraNonBondedException(format(
            "ERROR: moltype '{}' is defined multiple times in file '{}'",
            *it,
            _fileName
        ));
}

/**
 * @brief sets the file name
 *
 * @param fileName
 */
void IntraNonBondedReader::setFileName(const std::string_view &fileName)
{
    _fileName = fileName;
}

/**
 * @brief reinitializes the file pointer
 */
void IntraNonBondedReader::reInitializeFp() { _fp = std::ifstream(_fileName); }