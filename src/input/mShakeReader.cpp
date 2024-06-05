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

#include "exceptions.hpp"        // for MShakeFileException
#include "fileSettings.hpp"      // for FileSettings
#include "stringUtilities.hpp"   // for removeComments

using namespace input::mShake;

/**
 * @brief Wrapper to construct MShakeReader and read mShake file
 *
 * @param engine
 */
void input::mShake::readMShake(engine::Engine &engine)
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
MShakeReader::MShakeReader(engine::Engine &engine) : _engine(engine)
{
    _fileName = settings::FileSettings::getMShakeFileName();
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
        line              = utilities::removeComments(line, "#");
        auto lineElements = utilities::splitString(line);

        ++_lineNumber;

        if (lineElements.empty())
            customException::MShakeFileException(std::format(
                "Empty line in mShake file at line {}! The M-Shake file should "
                "be in the form a an extended xyz file. Therefore, this line "
                "should be the header line of the extended xyz file and "
                "contain the number of atoms.",
                _lineNumber
            ));

        const auto nAtoms = std::stoi(lineElements[0]);

        getline(_fp, line);
        processCommentLine(line);

        for (int i = 0; i < nAtoms; ++i)
        {
            getline(_fp, line);
            processAtomLine(line);
        }
    }
}

/**
 * @brief processes comment line
 *
 * @details processes comment line
 *
 * @param line
 */
void MShakeReader::processCommentLine(std::string &line)
{
    // line         = utilities::removeComments(line, "#");
    // auto configs = utilities::getLineCommands(line, _lineNumber);

    // for (auto &config : configs)
    // {
    //     // check if config contains moltype
    //     if (config.find("moltype") != std::string::npos)
    //     {
    //         auto configElements =
    //             utilities::addSpaces(config, "=", _lineNumber);
    //         // auto configElements = utilities::addSpaces(config, "=",
    //         // _lineNumber);
    //     }
    // }
}

/**
 * @brief processes atom line
 *
 * @details processes atom line
 *
 * @param line
 */
void MShakeReader::processAtomLine(std::string &line)
{
    line              = utilities::removeComments(line, "#");
    auto lineElements = utilities::splitString(line);

    if (lineElements.size() != 4)
        customException::MShakeFileException(std::format(
            "Wrong number of elements in atom line in mShake file at line "
            "{}! "
            "The M-Shake file should be in the form a an extended xyz "
            "file. "
            "Therefore, this line should contain the atom type and the "
            "coordinates of the atom.",
            _lineNumber
        ));

    const auto atomType = lineElements[0];
    const auto x        = std::stod(lineElements[1]);
    const auto y        = std::stod(lineElements[2]);
    const auto z        = std::stod(lineElements[3]);

    // _engine.addAtom(atomType, x, y, z);
}
