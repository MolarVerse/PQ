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

#include "ringPolymerInputParser.hpp"

#include "inputKeyAdapter.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"
#include "ringPolymerSettings.hpp"

using namespace input;
using namespace settings;

/**
 * @brief Construct a new RingPolymerInputParser::
 * RingPolymerInputParser object
 *
 * @details following keywords are registered: 1) rpmd_n_replica <size_t>,
 * must be at least 2
 *
 */
RingPolymerInputParser::RingPolymerInputParser() { addNumberOfBeadsKeyword(); }

void RingPolymerInputParser::addNumberOfBeadsKeyword()
{
    const auto metaData = KeyMetadata{
        .name  = "rpmd_n_replica",
        .title = "Number of Ring-Polymer Replicas",
        .description =
            "Number of beads (replicas) used in ring-polymer "
            "molecular dynamics"
    };

    const RangeValidator<size_t> rangeValidator{2, std::nullopt};

    const auto setValue = [](const size_t &nBeads)
    { RingPolymerSettings::setNumberOfBeads(nBeads); };

    auto &numberOfBeadsKey =
        _getRegistry().registerKey<size_t>(KeyRegistry<size_t>{
            .metadata = metaData,
            .onSet    = setValue,
            .validator =
                std::make_shared<RangeValidator<size_t>>(rangeValidator),
        });

    addKeyword("rpmd_n_replica", adapt(numberOfBeadsKey), false);
}
