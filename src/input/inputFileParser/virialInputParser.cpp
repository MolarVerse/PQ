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

#include "inputKeyAdapter.hpp"
#include "keyRegistry.hpp"
#include "settings.hpp"

using namespace input;
using namespace exc;

/**
 * @brief Construct a new Input File Parser Virial:: Input File Parser Virial
 * object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) virial "<molecular/atomic>"
 */
VirialInputParser::VirialInputParser() { addVirialKey(); }

/**
 * @brief adds virial key metadata
 *
 * @details this function adds the metadata for the virial key to the parser
 */
void VirialInputParser::addVirialKey()
{
    const auto metaData = KeyMetadata{
        .name        = "virial",
        .title       = "Virial Type",
        .description = "Specifies the type of virial: molecular or atomic",
    };

    const auto onSet = [](settings::VirialType virial)
    { settings::Settings::setVirialType(virial); };

    auto& key = _getRegistry().registerKey(
        KeyRegistry<settings::VirialType>{
            .metadata     = metaData,
            .defaultValue = settings::VirialType::MOLECULAR,
            .onSet        = onSet,
        }
    );

    addKeyword(metaData.name, adapt(key), false);
}
