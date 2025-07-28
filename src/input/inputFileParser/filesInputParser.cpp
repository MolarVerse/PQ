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

#include "filesInputParser.hpp"

#include <cstddef>       // for size_t
#include <functional>    // for _Bind_front_t, bind_front
#include <string_view>   // for string_view

#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException
#include "fileSettings.hpp"      // for FileSettings
#include "intraNonBonded.hpp"    // for IntraNonBonded
#include "stringUtilities.hpp"   // for fileExists

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser
 * Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) intra-nonBonded_file <string> 2)
 * topology_file <string> 3) parameter_file <string> 4) start_file <string>
 * (required) 5) rpmd_start_file <string> 6) moldescriptor_file <string>
 * 7) guff_path <string> (deprecated) 8) guff_file <string>
 * 9) mshake_file <string> 10) dftb_file <string>
 *
 * @param engine
 */
FilesInputParser::FilesInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("intra-nonBonded_file"),
        bind_front(&FilesInputParser::parseIntraNonBondedFile, this),
        false
    );

    addKeyword(
        std::string("topology_file"),
        bind_front(&FilesInputParser::parseTopologyFilename, this),
        false
    );

    addKeyword(
        std::string("parameter_file"),
        bind_front(&FilesInputParser::parseParameterFilename, this),
        false
    );

    addKeyword(
        std::string("start_file"),
        bind_front(&FilesInputParser::parseStartFilename, this),
        true
    );

    addKeyword(
        std::string("rpmd_start_file"),
        bind_front(&FilesInputParser::parseRingPolymerStartFilename, this),
        false
    );

    addKeyword(
        std::string("moldescriptor_file"),
        bind_front(&FilesInputParser::parseMoldescriptorFilename, this),
        false
    );

    addKeyword(
        std::string("guff_path"),
        bind_front(&FilesInputParser::parseGuffPath, this),
        false
    );

    addKeyword(
        std::string("guff_file"),
        bind_front(&FilesInputParser::parseGuffDatFilename, this),
        false
    );

    addKeyword(
        std::string("mshake_file"),
        bind_front(&FilesInputParser::parseMShakeFilename, this),
        false
    );

    addKeyword(
        std::string("dftb_file"),
        bind_front(&FilesInputParser::parseDFTBFilename, this),
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
 * @throws InputFileException if the file does not exist
 */
void FilesInputParser::parseIntraNonBondedFile(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &fileName = lineElements[2];

    if (!fileExists(fileName))
        throw InputFileException(
            std::format("Intra non bonded file \"{}\" File not found", fileName)
        );

    _engine.getIntraNonBonded().activate();

    FileSettings::setIntraNonBondedFileName(fileName);
    FileSettings::setIsIntraNonBondedFileNameSet();
}

/**
 * @brief parse topology file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if topology filename is empty or
 * file does not exist
 */
void FilesInputParser::parseTopologyFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open topology file - filename = {}", filename)
        );

    FileSettings::setTopologyFileName(filename);
    FileSettings::setIsTopologyFileNameSet();
}

/**
 * @brief parse parameter file name of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if parameter filename is empty or
 * file does not exist
 */
void FilesInputParser::parseParameterFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open parameter file - filename = {}", filename)
        );

    FileSettings::setParameterFileName(filename);
    FileSettings::setIsParameterFileNameSet();
}

/**
 * @brief parse start file of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void FilesInputParser::parseStartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open start file - filename = {}", filename)
        );

    FileSettings::setStartFileName(filename);
}

/**
 * @brief parse ring polymer start file of simulation and set it in settings
 *
 * @param lineElements
 * @param lineNumber
 */
void FilesInputParser::parseRingPolymerStartFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format(
                "Cannot open ring polymer start file - filename = {}",
                filename
            )
        );

    FileSettings::setRingPolymerStartFileName(filename);
    FileSettings::setIsRingPolymerStartFileNameSet();
}

/**
 * @brief parse moldescriptor file of simulation and set it in settings
 *
 * @details default is moldescriptor.dat
 *
 * @param lineElements
 *
 * @throws InputFileException if file does not exist
 */
void FilesInputParser::parseMoldescriptorFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format(
                "Cannot open moldescriptor file - filename = \"{}\" - file not "
                "found",
                filename
            )
        );

    FileSettings::setMolDescriptorFileName(filename);
}

/**
 * @brief parse guff path of simulation and set it in settings
 *
 * @throws InputFileException deprecated keyword
 */
void FilesInputParser::parseGuffPath(
    const std::vector<std::string> &,
    const size_t
)
{
    throw InputFileException(
        std::format(
            "The \"guff_path\" keyword id deprecated. Please use "
            "\"guffdat_file\" "
            "instead."
        )
    );
}

/**
 * @brief parse guff dat file of simulation and set it in settings
 *
 * @details default is guff.dat
 *
 * @param lineElements
 *
 * @throws InputFileException if file does not exist
 */
void FilesInputParser::parseGuffDatFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open guff file - filename = {}", filename)
        );

    FileSettings::setGuffDatFileName(filename);
}

/**
 * @brief parse mshake file of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throws InputFileException if file does not exist
 */
void FilesInputParser::parseMShakeFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open mshake file - filename = {}", filename)
        );

    FileSettings::setMShakeFileName(filename);
}

/**
 * @brief parse dftb file of simulation and set it in settings
 *
 * @param lineElements
 *
 * @throws InputFileException if file does not exist
 */
void FilesInputParser::parseDFTBFilename(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto &filename = lineElements[2];

    if (!fileExists(filename))
        throw InputFileException(
            std::format("Cannot open DFTB setup file - filename = {}", filename)
        );

    FileSettings::setDFTBFileName(filename);
}