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

#include <algorithm>
#include <cstddef>
#include <format>
#include <ranges>
#include <string>
#include <string_view>
#include <vector>

#include "enums/hybrid.hpp"
#include "enums/qm.hpp"
#include "exceptions.hpp"
#include "hybridSettings.hpp"
#include "inputConverter.hpp"
#include "inputKeyAdapter.hpp"
#include "keyRegistry.hpp"
#include "rangeValidator.hpp"

#ifdef PYTHON_ENABLED
#include "fileSettings.hpp"
#include "selection.hpp"
#endif

namespace input
{

    /**
     * @brief Construct a new HybridInputParser:: HybridInputParser object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) qm_prog "<string>" 2)
     * qm_script
     * "<string>"
     */
    HybridInputParser::HybridInputParser()
    {
        addInnerRegionCenterKey();
        addForcedCoreListKey();
        addForcedLayerListKey();
        addForcedOuterListKey();
        addUseQMChargesKey();
        addCoreRadiusKey();
        addLayerRadiusKey();
        addSmoothingRegionThicknessKey();
        addPointChargeThicknessKey();
        addSmoothingMethodKey();
        addQMForceDistributionKey();
    }

    void HybridInputParser::addInnerRegionCenterKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "inner_region_center",
            .title = "Inner Region Center",
            .description =
                "Specifies the center of the inner region in hybrid "
                "calculations",
        };

        const auto setValue = [](const SelectionTag &selection)
        {
            std::vector<size_t> convertedIndices;

            for (const auto &index : selection.indices)
            {
                // check if indices are positive
                if (index <= 0)
                {
                    throw exc::InputFileException(
                        std::format(
                            "Invalid atom index \"{}\" in input file\n"
                            "Atom indices must be positive",
                            index
                        )
                    );
                }
                convertedIndices.push_back(static_cast<size_t>(index));
            }

            settings::HybridSettings::setInnerRegionCenter(convertedIndices);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addForcedCoreListKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "forced_core_list",
            .title = "Forced Core List",
            .description =
                "Specifies the list of molecules which are forced to the CORE "
                "region in hybrid calculations",
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedCoreList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addForcedLayerListKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "forced_layer_list",
            .title = "Forced Layer List",
            .description =
                "Specifies the list of molecules which are forced to the LAYER "
                "region in hybrid calculations",
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedLayerList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addForcedOuterListKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "forced_outer_list",
            .title = "Forced Outer List",
            .description =
                "Specifies the list of molecules which are forced to the OUTER "
                "region in hybrid calculations",
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedOuterList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addUseQMChargesKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "use_qm_charges",
            .title = "Use QM Charges",
            .description =
                "Specifies whether QM charges should be used in the hybrid "
                "calculations",
        };

        const auto setValue = [](QMCharges value)
        {
            switch (value)
            {
                case QMCharges::QM:
                    settings::HybridSettings::setUseQMCharges(true);
                    break;
                case QMCharges::MM:
                    settings::HybridSettings::setUseQMCharges(false);
                    break;
            }
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<QMCharges>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addCoreRadiusKey()
    {
        const auto metaData = input::KeyMetadata{
            .name        = "core_radius",
            .title       = "Core Radius",
            .description = "Specifies the core radius in hybrid calculations",
        };

        const auto setValue = [](double value)
        { settings::HybridSettings::setCoreRadius(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addLayerRadiusKey()
    {
        const auto metaData = input::KeyMetadata{
            .name        = "layer_radius",
            .title       = "Layer Radius",
            .description = "Specifies the layer radius in hybrid calculations",
        };

        const auto setValue = [](double value)
        { settings::HybridSettings::setLayerRadius(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addSmoothingRegionThicknessKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "smoothing_region_thickness",
            .title = "Smoothing Region Thickness",
            .description =
                "Specifies the smoothing region thickness in hybrid "
                "calculations",
        };

        const auto setValue = [](double value)
        { settings::HybridSettings::setSmoothingRegionThickness(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addPointChargeThicknessKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "point_charge_thickness",
            .title = "Point Charge Thickness",
            .description =
                "Specifies the point charge thickness in hybrid calculations",
        };

        const auto setValue = [](double value)
        { settings::HybridSettings::setPointChargeThickness(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(PositiveGTEZeroDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addSmoothingMethodKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "smoothing_method",
            .title = "Smoothing Method",
            .description =
                "Specifies the smoothing method in hybrid calculations",
        };

        const auto setValue = [](SmoothingMethod value)
        { settings::HybridSettings::setSmoothingMethod(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SmoothingMethod>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addQMForceDistributionKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "qm_force_distribution",
            .title = "QM Force Distribution",
            .description =
                "Specifies the QM force distribution method in hybrid "
                "calculations",
        };

        const auto setValue = [](QMForceDist value)
        { settings::HybridSettings::setQMForceDist(value); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<QMForceDist>{
                .metadata = metaData,
                .onSet    = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief parse selection string
     *
     * @details This function parses a string that contains a selection of
     * atoms. The selection can be a list of atom indices or a selection string
     * that is understood by the PQAnalysis Python package. In order to use the
     * full selection parser power of the PQAnalysis Python package, the PQ
     * build must be compiled with Python bindings. If the PQ build is compiled
     * without Python bindings, the selection string must be a comma-separated
     * list of integers or a - separated range of indices, representing the atom
     * indices in the restart file that should be treated as the selection. If
     * the selection is empty, the function returns a vector with a single
     * element, 0.
     *
     * @param selection The selection string
     * @param key The key of the selection string
     *
     * @return std::vector<int> The selection vector
     *
     * @throws exc::InputFileException if the selection string contains
     * characters that are not digits, "-" or commas and the PQ build is
     * compiled without Python bindings.
     */
    std::vector<int> HybridInputParser::parseSelection(
        const std::string &selection,
        const std::string &key
    )
    {
        std::vector<int> selectionVec;

        if (selection.empty())
            return {0};

        auto needsPython = false;
        if (selection.find_first_not_of("0123456789,-") != std::string::npos)
            needsPython = true;

#ifdef PYTHON_ENABLED
        std::string restartFile = FileSettings::getStartFileName();
        std::string moldescFile = FileSettings::getMolDescriptorFileName();

        if (needsPython)
            selectionVec =
                pq_python::select(selection, restartFile, moldescFile);
#else

        // check if string contains any characters that are not digits or commas
        if (needsPython)
        {
            throw exc::InputFileException(
                std::format(
                    "The value of key {} - {} contains characters that are not "
                    "digits, \"-\" or commas. The current build of PQ was "
                    "compiled "
                    "without Python bindings, so the {} string must be a "
                    "comma-separated list of integers, representing the atom "
                    "indices in the restart file that should be treated as the "
                    "{}. "
                    "In order to use the full selection parser power of the "
                    "PQAnalysis Python package, the PQ build must be compiled "
                    "with "
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
     * @throws exc::InputFileException if the selection string is an
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

            auto atomIndexStr =
                std::string_view(selection).substr(pos, nextPos - pos);

            // remove all whitespaces from the atom index string
            atomIndexStr.remove_prefix(
                std::min(
                    atomIndexStr.find_first_not_of(' '),
                    atomIndexStr.size()
                )
            );
            const auto min = std::min(
                atomIndexStr.find_last_not_of(' ') + 1,
                atomIndexStr.size()
            );
            atomIndexStr.remove_suffix(atomIndexStr.size() - min);

            // check if the atom index string is a range of indices
            size_t rangePos = atomIndexStr.find('-');
            if (rangePos != std::string::npos)
            {
                const auto startString = atomIndexStr.substr(0, rangePos);
                const auto endString   = atomIndexStr.substr(rangePos + 1);

                int start = -1;
                int end   = -1;

                try
                {
                    start = std::stoi(std::string(startString));
                }
                catch (const std::invalid_argument &)
                {
                    throw exc::InputFileException(
                        std::format(
                            "Invalid start index \"{}\" in range \"{}\" for "
                            "key "
                            "{}. Must be a valid integer.",
                            startString,
                            atomIndexStr,
                            key
                        )
                    );
                }
                catch (const std::out_of_range &)
                {
                    throw exc::InputFileException(
                        std::format(
                            "Start index \"{}\" in range \"{}\" for key {} is "
                            "out "
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
                    throw exc::InputFileException(
                        std::format(
                            "Invalid end index \"{}\" in range \"{}\" for key "
                            "{}. "
                            "Must be a valid integer.",
                            endString,
                            atomIndexStr,
                            key
                        )
                    );
                }
                catch (const std::out_of_range &)
                {
                    throw exc::InputFileException(
                        std::format(
                            "End index \"{}\" in range \"{}\" for key {} is "
                            "out of "
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
                throw exc::InputFileException(
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
                throw exc::InputFileException(
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
            throw exc::InputFileException(
                std::format(
                    "The value of key {} - {} is an empty list. The {} string "
                    "must be a comma-separated list of integers or ranges, "
                    "representing the atom indices in the restart file that "
                    "should be treated as the {}.",
                    key,
                    selection,
                    key,
                    key
                )
            );
        }

        return selectionVec;
    }

}   // namespace input
