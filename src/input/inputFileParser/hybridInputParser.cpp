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

#include "hybridInputParser.hpp"

#include <algorithm>     // for min, unique
#include <cstddef>       // for size_t
#include <format>        // for format
#include <functional>    // for _Bind_front_t, bind_front
#include <ranges>        // for sort
#include <string>        // for string
#include <string_view>   // for string_view
#include <vector>        // for vector

#include "exceptions.hpp"   // for InputFileException, customException
#include "fileSettings.hpp"
#include "hybridSettings.hpp"    // for HybridSettings
#include "inputFileParser.hpp"   // for InputFileParser
#include "stringUtilities.hpp"   // for toLowerCopy
#include "typeAliases.hpp"       // for pq::strings

#ifdef PYTHON_ENABLED
#include "selection.hpp"   // for parseSelection
#endif

using namespace input;
using namespace engine;
using namespace customException;
using namespace settings;
using namespace utilities;

/**
 * @brief Construct a new HybridInputParser:: HybridInputParser object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) qm_prog <string> 2) qm_script
 * <string>
 *
 * @param engine
 */
HybridInputParser::HybridInputParser(Engine &engine) : InputFileParser(engine)
{
    addKeyword(
        std::string("inner_region_center"),
        bind_front(&HybridInputParser::parseInnerRegionCenter, this),
        false
    );
    addKeyword(
        std::string("forced_inner_list"),
        bind_front(&HybridInputParser::parseForcedInnerList, this),
        false
    );
    addKeyword(
        std::string("forced_outer_list"),
        bind_front(&HybridInputParser::parseForcedOuterList, this),
        false
    );
    addKeyword(
        std::string("qm_charges"),
        bind_front(&HybridInputParser::parseUseQMCharges, this),
        false
    );
    addKeyword(
        std::string("core_radius"),
        bind_front(&HybridInputParser::parseCoreRadius, this),
        false
    );
    addKeyword(
        std::string("layer_radius"),
        bind_front(&HybridInputParser::parseLayerRadius, this),
        false
    );
    addKeyword(
        std::string("smoothing_region_thickness"),
        bind_front(&HybridInputParser::parseSmoothingRegionThickness, this),
        false
    );
    addKeyword(
        std::string("point_charge_thickness"),
        bind_front(&HybridInputParser::parsePointChargeThickness, this),
        false
    );
    addKeyword(
        std::string("smoothing_method"),
        bind_front(&HybridInputParser::parseSmoothingMethod, this),
        false
    );
}

/**
 * @brief parse atom index selection which defines the core region
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseInnerRegionCenter(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setInnerRegionCenter(
        parseSelection(lineElements[2], lineElements[0])
    );
}

/**
 * @brief parse list of molecules which are forced to the inner region in hybrid
 * calculations
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseForcedInnerList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setForcedInnerList(
        parseSelection(lineElements[2], lineElements[0])
    );
}

/**
 * @brief parse list of molecules which are forced to the outer region in hybrid
 * calculations
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseForcedOuterList(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    HybridSettings::setForcedOuterList(
        parseSelection(lineElements[2], lineElements[0])
    );
}

/**
 * @brief parse if QM charges should be used
 *
 * @param lineElements
 * @param lineNumber
 */
void HybridInputParser::parseUseQMCharges(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);
    auto use_qm_charges = toLowerAndReplaceDashesCopy(lineElements[2]);

    if ("qm" == use_qm_charges)
        HybridSettings::setUseQMCharges(true);

    else if ("mm" == use_qm_charges)
        HybridSettings::setUseQMCharges(false);

    else
        throw InputFileException(
            std::format(
                "Invalid qm_charges \"{}\" in input file\n"
                "Possible values are: qm, mm",
                lineElements[2]
            )
        );
}

/**
 * @brief parse core radius
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the radius is negative
 */
void HybridInputParser::parseCoreRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto coreRadius = std::stod(lineElements[2]);

    if (coreRadius < 0.0)
        throw InputFileException(
            std::format(
                "Invalid {} {} in input file - must be a positive number",
                lineElements[0],
                lineElements[2]
            )
        );

    HybridSettings::setCoreRadius(coreRadius);
}

/**
 * @brief parse layer radius
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the radius is negative
 */
void HybridInputParser::parseLayerRadius(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto layerRadius = std::stod(lineElements[2]);

    if (layerRadius < 0.0)
        throw InputFileException(
            std::format(
                "Invalid {} {} in input file - must be a positive number",
                lineElements[0],
                lineElements[2]
            )
        );

    HybridSettings::setLayerRadius(layerRadius);
}

/**
 * @brief parse smoothing region thickness
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the thickness is negative
 */
void HybridInputParser::parseSmoothingRegionThickness(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto thickness = std::stod(lineElements[2]);

    if (thickness < 0.0)
        throw InputFileException(
            std::format(
                "Invalid {} {} in input file - must be a positive number",
                lineElements[0],
                lineElements[2]
            )
        );

    HybridSettings::setSmoothingRegionThickness(thickness);
}

/**
 * @brief parse point charge thickness
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if the radius is negative
 */
void HybridInputParser::parsePointChargeThickness(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    checkCommand(lineElements, lineNumber);

    const auto radius = std::stod(lineElements[2]);

    if (radius < 0.0)
        throw InputFileException(
            std::format(
                "Invalid {} {} in input file - must be a positive number",
                lineElements[0],
                lineElements[2]
            )
        );

    HybridSettings::setPointChargeThickness(radius);
}

/**
 * @brief parse smoothing method
 *
 * @param lineElements
 * @param lineNumber
 *
 * @throws InputFileException if
 */
void HybridInputParser::parseSmoothingMethod(
    const std::vector<std::string> &lineElements,
    const size_t                    lineNumber
)
{
    const auto method = toLowerAndReplaceDashesCopy(lineElements[2]);

    using enum settings::SmoothingMethod;

    if (method == "hotspot")
        HybridSettings::setSmoothingMethod(HOTSPOT);

    else if (method == "exact")
        HybridSettings::setSmoothingMethod(EXACT);
    else
        throw InputFileException(
            std::format(
                "Invalid smoothing method \"{}\" in input file\n"
                "Possible values are: hotspot, exact",
                lineElements[2]
            )
        );
}

/**
 * @brief parse selection string
 *
 * @details This function parses a string that contains a selection of atoms.
 * The selection can be a list of atom indices or a selection string that is
 * understood by the PQAnalysis Python package. In order to use the full
 * selection parser power of the PQAnalysis Python package, the PQ build must be
 * compiled with Python bindings. If the PQ build is compiled without Python
 * bindings, the selection string must be a comma-separated list of integers or
 * a - separated range of indices, representing the atom indices in the restart
 * file that should be treated as the selection. If the selection is empty, the
 * function returns a vector with a single element, 0.
 *
 * @param selection The selection string
 * @param key The key of the selection string
 *
 * @return std::vector<int> The selection vector
 *
 * @throws customException::InputFileException if the selection string contains
 * characters that are not digits, "-" or commas and the PQ build is compiled
 * without Python bindings.
 */
std::vector<int> HybridInputParser::parseSelection(
    const std::string &selection,
    const std::string &key
)
{
    std::string restartFile = FileSettings::getStartFileName();
    std::string moldescFile = FileSettings::getMolDescriptorFileName();

    std::vector<int> selectionVec;

    if (selection.empty())
        return {0};

    auto needsPython = false;
    if (selection.find_first_not_of("0123456789,-") != std::string::npos)
        needsPython = true;

#ifdef PYTHON_ENABLED
    if (needsPython)
        selectionVec = pq_python::select(selection, restartFile, moldescFile);
#else

    // check if string contains any characters that are not digits or commas
    if (needsPython)
    {
        throw InputFileException(
            std::format(
                "The value of key {} - {} contains characters that are not "
                "digits, \"-\" or commas. The current build of PQ was compiled "
                "without Python bindings, so the {} string must be a "
                "comma-separated list of integers, representing the atom "
                "indices in the restart file that should be treated as the {}. "
                "In order to use the full selection parser power of the "
                "PQAnalysis Python package, the PQ build must be compiled with "
                "Python bindings.",
                key,
                selection,
                key,
                key
            )
        );
    }
#endif

    if (!needsPython)
        selectionVec = parseSelectionNoPython(selection, key);

    std::ranges::sort(selectionVec);
    auto ret = std::ranges::unique(selectionVec);
    selectionVec.erase(ret.begin(), ret.end());

    return selectionVec;
}

/**
 * @brief parse selection string without Python
 *
 * @param selection The selection string
 * @param key The key of the selection string
 *
 * @return std::vector<int> The selection vector
 *
 * @throws customException::InputFileException if the selection string is an
 * empty list
 */
std::vector<int> HybridInputParser::parseSelectionNoPython(
    const std::string &selection,
    const std::string &key
)
{
    std::vector<int> selectionVec;

    size_t pos = 0;
    while (pos < selection.size())
    {
        size_t nextPos = selection.find(',', pos);
        if (nextPos == std::string::npos)
            nextPos = selection.size();

        std::string_view atomIndexStr(selection.c_str() + pos, nextPos - pos);

        // remove all whitespaces from the atom index string
        atomIndexStr.remove_prefix(
            std::min(atomIndexStr.find_first_not_of(" "), atomIndexStr.size())
        );
        const auto min = std::min(
            atomIndexStr.find_last_not_of(" ") + 1,
            atomIndexStr.size()
        );
        atomIndexStr.remove_suffix(atomIndexStr.size() - min);

        // check if the atom index string is a range of indices
        size_t rangePos = atomIndexStr.find('-');
        if (rangePos != std::string::npos)
        {
            const auto startString = atomIndexStr.substr(0, rangePos);
            const auto endString   = atomIndexStr.substr(rangePos + 1);

            int start, end;
            try
            {
                start = std::stoi(std::string(startString));
            }
            catch (const std::invalid_argument &)
            {
                throw InputFileException(
                    std::format(
                        "Invalid start index \"{}\" in range \"{}\" for key "
                        "{}. Must be a valid integer.",
                        startString,
                        atomIndexStr,
                        key
                    )
                );
            }
            catch (const std::out_of_range &)
            {
                throw InputFileException(
                    std::format(
                        "Start index \"{}\" in range \"{}\" for key {} is out "
                        "of range.",
                        startString,
                        atomIndexStr,
                        key
                    )
                );
            }

            try
            {
                end = std::stoi(std::string(endString));
            }
            catch (const std::invalid_argument &)
            {
                throw InputFileException(
                    std::format(
                        "Invalid end index \"{}\" in range \"{}\" for key {}. "
                        "Must be a valid integer.",
                        endString,
                        atomIndexStr,
                        key
                    )
                );
            }
            catch (const std::out_of_range &)
            {
                throw InputFileException(
                    std::format(
                        "End index \"{}\" in range \"{}\" for key {} is out of "
                        "range.",
                        endString,
                        atomIndexStr,
                        key
                    )
                );
            }

            for (int i = start; i <= end; ++i) selectionVec.push_back(i);

            pos = nextPos + 1;
            continue;
        }

        try
        {
            selectionVec.push_back(std::stoi(std::string(atomIndexStr)));
        }
        catch (const std::invalid_argument &)
        {
            throw InputFileException(
                std::format(
                    "Invalid atom index \"{}\" for key {}. Must be a valid "
                    "integer.",
                    atomIndexStr,
                    key
                )
            );
        }
        catch (const std::out_of_range &)
        {
            throw InputFileException(
                std::format(
                    "Atom index \"{}\" for key {} is out of range.",
                    atomIndexStr,
                    key
                )
            );
        }
        pos = nextPos + 1;
    }

    // check if the selection vector is empty
    if (selectionVec.empty())
    {
        throw customException::InputFileException(
            std::format(
                "The value of key {} - {} is an empty list. The {} string must "
                "be a comma-separated list of integers or ranges, representing "
                "the atom indices in the restart file that should be treated "
                "as the {}.",
                key,
                selection,
                key,
                key
            )
        );
    }

    return selectionVec;
}