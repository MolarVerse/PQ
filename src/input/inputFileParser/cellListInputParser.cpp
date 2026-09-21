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

#include "cellListInputParser.hpp"

#include <cstddef>   // for size_t
#include <format>    // for format
#include <optional>
#include <string>   // for allocator, operator==, string
#include <utility>

#include "celllist.hpp"
#include "inputKeyAdapter.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"
#include "settings.hpp"

using namespace input;
using namespace exc;

/**
 * @brief Construct a new Input File Parser Cell List:: Input File Parser Cell
 * List object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) cell-list "<on/off>"
 * 2) cell-number "<size_t>"
 *
 * @param cellListPtr pointer to the cell list object
 */
CellListInputParser::CellListInputParser(
    std::shared_ptr<molsys::CellList> cellListPtr
)
    : _cellListPtr(std::move(cellListPtr))
{
    addCellListActivated();
    addNumberOfCells();
}

/**
 * @brief Adds the keyword for activating the cell list
 *
 * @details default value is "off"
 */
void CellListInputParser::addCellListActivated()
{
    const auto defaultValue = false;
    const auto metaData     = KeyMetadata{
            .name        = "cell-list",
            .title       = "Activation of Cell List",
            .description = std::format(
            "Specifies whether the cell list is activated or the brute force "
                "method is used - default is {}",
            defaultValue ? "on" : "off"
        )
    };

    const auto setValue = [](bool isActivated)
    {
        if (isActivated)
            settings::Settings::activateCellList();
        else
            settings::Settings::deactivateCellList();
    };

    auto &key = _getRegistry().registerKey<bool>(KeyRegistry<bool>{
        .metadata     = metaData,
        .defaultValue = defaultValue,
        .onSet        = setValue
    });

    addKeyword(std::string("cell-list"), adapt(key), false);
}

/**
 * @brief Adds the keyword for specifying the number of cells
 *
 * @details default value is 7
 */
void CellListInputParser::addNumberOfCells()
{
    const auto defaultValue = 7UL;

    const auto metaData = KeyMetadata{
        .name        = "cell-number",
        .title       = "Number of Cells",
        .description = std::format(
            "Specifies the number of cells used for each dimension - default "
            "is {}",
            defaultValue
        )
    };

    const auto setValue = [this](size_t numberOfCells)
    { _cellListPtr->setNumberOfCells(numberOfCells); };

    const auto validator = RangeValidator<size_t>(1, std::nullopt);

    auto &key = _getRegistry().registerKey<size_t>(KeyRegistry<size_t>{
        .metadata     = metaData,
        .defaultValue = defaultValue,
        .onSet        = setValue,
        .validator    = std::make_shared<RangeValidator<size_t>>(validator)
    });

    addKeyword(std::string("cell-number"), adapt(key), false);
}
