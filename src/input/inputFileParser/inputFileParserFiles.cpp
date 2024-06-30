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

#include "inputFileParserFiles.hpp"

#include <cstddef>       // for size_t
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException
#include "fileSettings.hpp"      // for FileSettings
#include "intraNonBonded.hpp"    // for IntraNonBonded
#include "stringUtilities.hpp"   // for fileExists

using namespace input;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser
 * Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) intra-nonBonded_file <string> 2)
 * topology_file <string> 3) parameter_file <string> 4) start_file <string>
 * (required) 5) moldescriptor_file <string> 6) guff_path <string> (deprecated)
 * 7) guff_file <string>
 *
 * @param engine
 */
InputFileParserFiles::InputFileParserFiles(engine::Engine &engine)
    : InputFileParser(engine)
{
    addKeyword(
        std::string("intra-nonBonded_file"),
        bind_front(&InputFileParserFiles::parseIntraNonBondedFile, this),
        false
    );

    addKeyword(
        std::string("topology_file"),
        bind_front(&InputFileParserFiles::parseTopologyFilename, this),
        false
    );

    addKeyword(
        std::string("parameter_file"),
        bind_front(&InputFileParserFiles::parseParameterFilename, this),
        false
    );

    addKeyword(
        std::string("start_file"),
        bind_front(&InputFileParserFiles::parseStartFilename, this),
        true
    );

    addKeyword(
        std::string("rpmd_start_file"),
        bind_front(&InputFileParserFiles::parseRingPolymerStartFilename, this),
        false
    );

    addKeyword(
        std::string("moldescriptor_file"),
        bind_front(&InputFileParserFiles::parseMoldescriptorFilename, this),
        false
    );

    addKeyword(
        std::string("guff_path"),
        bind_front(&InputFileParserFiles::parseGuffPath, this),
        false
    );

    addKeyword(
        std::string("guff_file"),
        bind_front(&InputFileParserFiles::parseGuffDatFilename, this),
        false
    );

    addKeyword(
        std::string("mshake_file"),
        bind_front(&InputFileParserFiles::parseMShakeFilename, this),
        false
    );
}

/**
 * @brief Parse the name of the file containing the intraNonBonded combinations
 *
 * @details Settings this keyword activates the intraNonBonded interactions
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws customException::InputFileException if the file does not exist
 */
void InputFileParserFiles::parseIntraNonBondedFile(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &fileName = lineElements[2];

    if (!utilities::fileExists(fileName))
        throw customException::InputFileException(
            std::format("Intra non bonded file \"{}\" File not found", fileName)
        );

    _engine.getIntraNonBonded().activate();

    settings::FileSettings::setIntraNonBondedFileName(fileName);
    settings::FileSettings::setIsIntraNonBondedFileNameSet();
}

/**
 * @brief parse topology file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws customException::InputFileException if topology filename is empty or
 * file does not exist
 */
void InputFileParserFiles::parseTopologyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(
            std::format("Cannot open topology file - filename = {}", filename)
        );

    settings::FileSettings::setTopologyFileName(filename);
    settings::FileSettings::setIsTopologyFileNameSet();
}

/**
 * @brief parse parameter file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws customException::InputFileException if parameter filename is empty or
 * file does not exist
 */
void InputFileParserFiles::parseParameterFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(
            std::format("Cannot open parameter file - filename = {}", filename)
        );

    settings::FileSettings::setParameterFileName(filename);
    settings::FileSettings::setIsParameterFileNameSet();
}

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserFiles::parseStartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(
            std::format("Cannot open start file - filename = {}", filename)
        );

    settings::FileSettings::setStartFileName(filename);
}

/**
 * @brief parse ring polymer start file of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void InputFileParserFiles::parseRingPolymerStartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(std::format(
            "Cannot open ring polymer start file - filename = {}",
            filename
        ));

    settings::FileSettings::setRingPolymerStartFileName(filename);
    settings::FileSettings::setIsRingPolymerStartFileNameSet();
}

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @details default is moldescriptor.dat
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if file does not exist
 */
void InputFileParserFiles::parseMoldescriptorFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(std::format(
            "Cannot open moldescriptor file - filename = \"{}\" - file not "
            "found",
            filename
        ));

    settings::FileSettings::setMolDescriptorFileName(filename);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @throws customException::InputFileException deprecated keyword
 */
void InputFileParserFiles::parseGuffPath(
    const std::vector<std::string> &,
    const size_t
)
{
    throw customException::InputFileException(std::format(
        "The \"guff_path\" keyword id deprecated. Please use \"guffdat_file\" "
        "instead."
    ));
}

/**
 * @brief parse guff dat file of simulation and set it in settings
 *
 * @details default is guff.dat
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if file does not exist
 */
void InputFileParserFiles::parseGuffDatFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(
            std::format("Cannot open guff file - filename = {}", filename)
        );

    settings::FileSettings::setGuffDatFileName(filename);
}

/**
 * @brief parse mshake file of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throws customException::InputFileException if file does not exist
 */
void InputFileParserFiles::parseMShakeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!utilities::fileExists(filename))
        throw customException::InputFileException(
            std::format("Cannot open mshake file - filename = {}", filename)
        );

    settings::FileSettings::setMShakeFileName(filename);
}