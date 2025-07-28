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

#include "mShakeReader.hpp"

#include <format>   // for format

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for MShakeFileException
#include "fileSettings.hpp"      // for FileSettings
#include "mShakeReference.hpp"   // for MShakeReference
#include "stringUtilities.hpp"   // for removeComments

using namespace input::mShake;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;
using namespace constraints;
using namespace simulationBox;

/**
 * @brief Wrapper to construct MShakeReader and read mShake file
 *
 * @param engine
 */
void input::mShake::readMShake(Engine &engine)
{
    MShakeReader mShakeReader(engine);
    mShakeReader.read();
}

/**
 * @brief constructor
 *
 * @details opens mShake file pointer
 *
 * @param engine
 */
MShakeReader::MShakeReader(Engine &engine) : _engine(engine)
{
    _fileName = FileSettings::getMShakeFileName();
    _fp.open(_fileName);
}

/**
 * @brief reads mShake file
 *
 * @details reads mShake file
 */
void MShakeReader::read()
{
    std::string line;

    _lineNumber = 0;

    while (getline(_fp, line))
    {
        line              = removeComments(line, "#");
        auto lineElements = splitString(line);

        ++_lineNumber;

        const auto nAtoms          = std::stoi(lineElements[0]);
        auto       mShakeReference = MShakeReference();

        getline(_fp, line);
        processCommentLine(line, mShakeReference);

        std::vector<std::string> atomLines;

        for (int i = 0; i < nAtoms; ++i)
        {
            getline(_fp, line);
            atomLines.push_back(line);
        }

        processAtomLines(atomLines, mShakeReference);

        _engine.getConstraints().addMShakeReference(mShakeReference);
    }
}

/**
 * @brief processes comment line
 *
 * @details processes comment line
 *
 * @param line
 * @param mShakeReference (empty)
 *
 * @throws MShakeFileException - if no moltype definition is
 * found in the comment line
 */
void MShakeReader::processCommentLine(
    std::string     &line,
    MShakeReference &mShakeReference
)
{
    auto lineCommands = removeComments(line, "#");

    auto configs = std::vector<std::string>();

    if (lineCommands.empty())
        configs = {};
    else
        configs = getLineCommands(lineCommands, _lineNumber);

    auto foundMolType = false;

    for (auto &config : configs)
    {
        addSpaces(config, "=", _lineNumber);
        auto configElements = splitString(config);

        if (toLowerCopy(configElements[0]) == "moltype")
        {
            const auto molType = std::stoi(configElements[2]);
            try
            {
                auto  simBox       = _engine.getSimulationBox();
                auto &moleculeType = simBox.findMoleculeType(size_t(molType));

                mShakeReference.setMoleculeType(moleculeType);

                foundMolType = true;
                break;
            }
            catch (const RstFileException &e)
            {
                throw MShakeFileException(e.what());
            }
        }
    }

    if (!foundMolType)
        throw MShakeFileException(
            std::format(
                "Unknown command in mShake file at line {}! The M-Shake file "
                "should be in the form a an extended xyz file. Here, the "
                "comment line should contain the molecule type from the "
                "moldescriptor file in the following form: 'MolType = 1;'. "
                "Please note that the syntax parsing works exactly like in the "
                "input file. Thus, it is case insensitive and the commands are "
                "separated by semicolons. Furthermore, the spaces around the "
                "'=' sign can be of arbitrary length (including also no spaces "
                "at all).",
                _lineNumber
            )
        );
}

/**
 * @brief processes atom line
 *
 * @details processes atom line
 *
 * @param lines
 *
 * @throws MShakeFileException - if a line does not contain
 * exactly 4 (1 str and 3 double) arguments
 * @throws MShakeFileException - if the atomnames in the mshake
 * file do not correspond to the atom names in the rst file
 */
void MShakeReader::processAtomLines(
    std::vector<std::string> &lines,
    MShakeReference          &mShakeReference
)
{
    std::vector<std::string> atomNames;
    std::vector<Atom>        atoms;

    for (auto &line : lines)
    {
        line              = removeComments(line, "#");
        auto lineElements = splitString(line);

        if (lineElements.size() != 4)
            throw MShakeFileException(
                std::format(
                    "Wrong number of elements in atom lines in mShake file "
                    "starting at line {}! The M-Shake file should be in the "
                    "form a "
                    "an extended xyz file. Therefore, this line should contain "
                    "the "
                    "atom type and the coordinates of the atom.",
                    _lineNumber
                )
            );

        const auto atomName = lineElements[0];
        const auto x        = std::stod(lineElements[1]);
        const auto y        = std::stod(lineElements[2]);
        const auto z        = std::stod(lineElements[3]);

        auto atom = Atom();

        atom.setName(atomName);
        atom.addPosition({x, y, z});

        atomNames.push_back(atomName);
        atoms.push_back(atom);
    }

    auto  molType      = mShakeReference.getMoleculeType();
    auto &refAtomNames = molType.getAtomNames();

    if (atoms.size() == 1)
    {
        throw MShakeFileException(
            std::format(
                "Molecule type {} has only one atom. M-Shake requires at least "
                "two "
                "atoms.",
                molType.getMoltype()
            )
        );
    }

    if (atomNames != refAtomNames)
        throw MShakeFileException(
            std::format(
                "Atom names in mShake file at line {} do not match the atom "
                "names of the molecule type! The M-Shake file should be in the "
                "form a an extended xyz file. Therefore, the atom names in the "
                "atom lines should match the atom names of the molecule type "
                "from the restart file.",
                _lineNumber
            )
        );

    mShakeReference.setAtoms(atoms);
}

/**
 * @brief getter for file name
 *
 * @return file name
 */
[[nodiscard]] std::string MShakeReader::getFileName() const
{
    return _fileName;
}
