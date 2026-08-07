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

#include "virialInputParser.hpp"

#include <format>       // for format
#include <functional>   // for _Bind_front_t, bind_front

#include "atomicVirial.hpp"      // for VirialAtomic
#include "engine.hpp"            // for Engine
#include "exceptions.hpp"        // for InputFileException, customException
#include "molecularVirial.hpp"   // for VirialMolecular
#include "physicalData.hpp"      // for PhysicalData
#include "stringUtilities.hpp"   // for toLowerCopy
#include "virial.hpp"            // for Virial

using namespace input;
using namespace virial;
using namespace engine;
using namespace customException;
using namespace utilities;

/**
 * @brief Construct a new Input File Parser Virial:: Input File Parser Virial
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) virial <molecular/atomic>
 *
 * @param engine
 */
VirialInputParser::VirialInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("virial"),
        bind_front(&VirialInputParser::parseVirial, this),
        false
    );
}

/**
 * @brief parses virial command
 *
 * @details possible options are:
 * 1) molecular - molecular virial (default)
 * 2) atomic    - atomic virial - sets file pointer to atomic virial file in
 * physical data
 *
 * @param lineElements
 *
 * @throws InputFileException if invalid virial keyword
 */
void VirialInputParser::parseVirial(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto virial = toLowerCopy(lineElements[2]);

    if (virial == "molecular")
        _engine.makeVirial(MolecularVirial());

    else if (virial == "atomic")
    {
        _engine.makeVirial(AtomicVirial());
        _engine.getPhysicalData().changeKineticVirialToAtomic();
    }
    else
        throw InputFileException(format(
            "Invalid virial setting \"{}\" at line {} in input file.\n"
            "Possible options are: molecular or atomic",
            lineElements[2],
            lineNumber
        ));
}