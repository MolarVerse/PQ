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

#include <cstddef>
#include <format>
#include <optional>
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

namespace input
{
    namespace
    {
        /**
         * @brief parses a forced-list value: molecule indices and ranges only
         *
         * @details the selection grammar is rejected, because it returns atom
         * indices
         */
        std::optional<SelectionTag> parseMoleculeIndices(std::string_view raw)
        {
            if (raw.find_first_not_of("0123456789,-") != std::string_view::npos)
            {
                return std::nullopt;
            }

            auto converter = Converter<SelectionTag>("", std::string(raw));
            return converter.tryParse(raw);
        }

        /**
         * @brief error message for an invalid forced-list value
         */
        std::string moleculeIndexError(const std::string_view key)
        {
            return std::format(
                "{} takes molecule indices, e.g. 0,2-4 (the selection grammar "
                "is not supported)",
                key
            );
        }
    }   // namespace

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
            if (selection.expression.has_value())
            {
                settings::HybridSettings::setInnerRegionCenterExpression(
                    *selection.expression
                );
                return;
            }

            std::vector<size_t> convertedIndices;

            for (const auto &index : selection.indices)
            {
                if (index < 0)
                {
                    throw exc::InputFileException(
                        std::format(
                            "Invalid atom index \"{}\" in input file\n"
                            "Atom indices must be non-negative",
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
            .errorMessage = moleculeIndexError("forced_core_list"),
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedCoreList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata     = metaData,
                .customParser = parseMoleculeIndices,
                .onSet        = setValue,
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
            .errorMessage = moleculeIndexError("forced_layer_list"),
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedLayerList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata     = metaData,
                .customParser = parseMoleculeIndices,
                .onSet        = setValue,
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
            .errorMessage = moleculeIndexError("forced_outer_list"),
        };

        const auto setValue = [](const SelectionTag &selection)
        { settings::HybridSettings::setForcedOuterList(selection.indices); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<SelectionTag>{
                .metadata     = metaData,
                .customParser = parseMoleculeIndices,
                .onSet        = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    void HybridInputParser::addUseQMChargesKey()
    {
        const auto metaData = input::KeyMetadata{
            .name  = "qm_charges",
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

    /**
     * @brief not yet read from the input file as this requires a new smoothing
     * method, see issue #1012 https://github.com/MolarVerse/PQ/issues/1012
     *
     */
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

}   // namespace input
