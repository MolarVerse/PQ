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

#include "inputConverter.hpp"

#include <ranges>
#include <string>
#include <unordered_map>
#include <utility>

#include "stringUtilities.hpp"

#ifdef PYTHON_ENABLED
#include "fileSettings.hpp"   // for FileSettings
#include "selection.hpp"      // for parseSelection
#endif

namespace input
{
    /**
     * @brief attempts to parse a double from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed double if successful,
     *         std::nullopt otherwise
     */
    std::optional<double> Converter<double>::tryParse(std::string_view raw)
    {
        double     value{};
        const auto result =
            std::from_chars(raw.data(), raw.data() + raw.size(), value);

        if (result.ec != std::errc{} || result.ptr != raw.data() + raw.size())
            return std::nullopt;

        return value;
    }

    /**
     * @brief attempts to parse a bool from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed bool if successful,
     *         std::nullopt otherwise
     */
    std::optional<bool> Converter<bool>::tryParse(std::string_view raw)
    {
        const auto rawTransformed = utilities::toLowerCopy(raw);

        const auto& keys   = std::views::keys(boolKeywords);
        const auto& values = std::views::values(boolKeywords);

        if (std::ranges::find(keys, rawTransformed) != keys.end())
            return true;
        if (std::ranges::find(values, rawTransformed) != values.end())
            return false;

        return std::nullopt;
    }

    /**
     * @brief attempts to parse a File from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed File if successful,
     *         std::nullopt otherwise
     */
    std::optional<mstd::File> Converter<mstd::File>::tryParse(
        std::string_view raw
    )
    {
        mstd::File file((std::string(raw)));
        if (file.exists())
            return file;

        return std::nullopt;
    }

    /**
     * @brief attempts to parse a std::string from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed std::string if successful,
     *         std::nullopt otherwise
     */
    std::optional<std::string> Converter<std::string>::tryParse(
        std::string_view raw
    )
    {
        return std::string(raw);
    }

    /**
     * @brief attempts to parse a std::unordered_map<std::string, double> from a
     * raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed std::unordered_map<std::string,
     * double> if successful, std::nullopt otherwise
     */
    std::optional<std::unordered_map<std::string, double>> Converter<
        std::unordered_map<std::string, double>>::tryParse(std::string_view raw)
    {
        std::unordered_map<std::string, double> result;
        std::string                             input(raw);

        std::stringstream sstream(input);
        std::string       item;
        while (std::getline(sstream, item, ','))
        {
            const auto separator = item.find(':');
            if (separator == std::string::npos || 0 == separator ||
                separator + 1 == item.size() ||
                item.find(':', separator + 1) != std::string::npos)
            {
                return std::nullopt;
            }

            const auto key = item.substr(0, separator);
            try
            {
                result[key] =
                    utilities::stringToFiniteDouble(item.substr(separator + 1));
            }
            catch (const std::invalid_argument&)
            {
                return std::nullopt;
            }
            catch (const std::out_of_range&)
            {
                return std::nullopt;
            }
        }

        return result;
    }

    /**
     * @brief attempts to parse a SelectionTag from a raw input-file token
     *
     * @param raw the raw input-file token
     * @return an optional containing the parsed SelectionTag if successful,
     *         std::nullopt otherwise
     */
    std::optional<SelectionTag> Converter<SelectionTag>::tryParse(
        std::string_view raw
    )
    {
        std::vector<int> selectionVec;

        if (raw.empty())
            return SelectionTag{.indices = {0}};

        auto needsPython = false;
        if (raw.find_first_not_of("0123456789,-") != std::string::npos)
            needsPython = true;

#ifdef PYTHON_ENABLED
        std::string restartFile = settings::FileSettings::getStartFileName();
        std::string moldescFile =
            settings::FileSettings::getMolDescriptorFileName();

        if (needsPython)
            selectionVec =
                pq_python::select(std::string(raw), restartFile, moldescFile);
#else

        // check if string contains any characters that are not digits or commas
        if (needsPython)
        {
            _selectionError = SelectionError::NeedsPython;
            return std::nullopt;
        }
#endif

        if (!needsPython)
        {
            const auto result = _parseSelectionNoPython(std::string(raw));
            if (!result)
                return std::nullopt;
            selectionVec = result.value();
        }

        std::ranges::sort(selectionVec);
        auto ret = std::ranges::unique(selectionVec);
        selectionVec.erase(ret.begin(), ret.end());

        return SelectionTag{.indices = std::move(selectionVec)};
    }

    /**
     * @brief describes the domain of valid File inputs
     *
     * @return a string describing the domain
     */
    std::string Converter<mstd::File>::describeDomain(
        const std::vector<mstd::File>& notAllowed
    )
    {
        std::string message = "Value must be an existing file path.";

        for (const auto& value : notAllowed)
            message += ", not allowed: " + value.fileName();

        return message;
    }

    /**
     * @brief describes the domain of valid std::string inputs
     *
     * @return a string describing the domain
     */
    std::string Converter<bool>::describeDomain(
        const std::vector<bool>& /*notAllowed*/
    )
    {
        std::string options;
        for (const auto& [positive, negative] : boolKeywords)
        {
            if (!options.empty())
                options += "|";

            options += positive;
            options += "|";
            options += negative;
        }
        return "Allowed values: " + options;
    }

    /**
     * @brief describes the domain of valid std::unordered_map<std::string,
     * double> inputs
     *
     * @return a string describing the domain
     */
    std::string Converter<std::unordered_map<std::string, double>>::
        describeDomain(
            const std::vector<
                std::unordered_map<std::string, double>>& /*notAllowed*/
        )
    {
        return "Value must be a comma-separated list of key:value pairs, where "
               "the key is a string and the value is a double.";
    }

    std::string Converter<SelectionTag>::describeDomain(
        const std::vector<SelectionTag>& /*notAllowed*/
    )
    {
        switch (_selectionError)
        {
            case SelectionError::NeedsPython:
                return std::format(
                    "The value {} contains characters that are not "
                    "digits, \"-\" or commas. The current build of PQ was "
                    "compiled without Python bindings, so the string must be a "
                    "comma-separated list of integers, representing the atom "
                    "indices in the restart file that should be treated as the "
                    "desired key. In order to use the full selection parser "
                    "power of the PQAnalysis Python package, the PQ build must "
                    "be compiled with Python bindings.",
                    _getRaw()
                );
            case SelectionError::InvalidStartIndex:
                return "The start index of the selection is invalid. Must be a "
                       "valid integer.";
            case SelectionError::OutOfRangeStartIndex:
                return "The start index of the selection is out of range.";
            case SelectionError::InvalidEndIndex:
                return "The end index of the selection is invalid. Must be a "
                       "valid integer.";
            case SelectionError::OutOfRangeEndIndex:
                return "The end index of the selection is out of range.";
            case SelectionError::InvalidAtomIndex:
                return "An atom index in the selection is invalid. Must be a "
                       "valid integer.";
            case SelectionError::OutOfRangeAtomIndex:
                return "An atom index in the selection is out of range.";
            case SelectionError::EmptySelection:
                return "The selection is empty.";
            case SelectionError::None: return "Unknown selection error.";
        }

        std::unreachable();
    }

    /**
     * @brief parses a selection string without using Python
     *
     * @param selection the selection string
     * @return an optional vector of atom indices, or std::nullopt if parsing
     * fails
     */
    std::optional<std::vector<int>> Converter<
        SelectionTag>::_parseSelectionNoPython(const std::string& selection)
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
                catch (const std::invalid_argument&)
                {
                    _selectionError = SelectionError::InvalidStartIndex;
                    return std::nullopt;
                }
                catch (const std::out_of_range&)
                {
                    _selectionError = SelectionError::OutOfRangeStartIndex;
                    return std::nullopt;
                }
                catch (const std::exception&)
                {
                    // unknown exception occurred while parsing start index
                    return std::nullopt;
                }

                try
                {
                    end = std::stoi(std::string(endString));
                }
                catch (const std::invalid_argument&)
                {
                    _selectionError = SelectionError::InvalidEndIndex;
                    return std::nullopt;
                }
                catch (const std::out_of_range&)
                {
                    _selectionError = SelectionError::OutOfRangeEndIndex;
                    return std::nullopt;
                }
                catch (const std::exception&)
                {
                    // unknown exception occurred while parsing end index
                    return std::nullopt;
                }

                for (int i = start; i <= end; ++i) selectionVec.push_back(i);

                pos = nextPos + 1;
                continue;
            }

            try
            {
                selectionVec.push_back(std::stoi(std::string(atomIndexStr)));
            }
            catch (const std::invalid_argument&)
            {
                _selectionError = SelectionError::InvalidAtomIndex;
                return std::nullopt;
            }
            catch (const std::out_of_range&)
            {
                _selectionError = SelectionError::OutOfRangeAtomIndex;
                return std::nullopt;
            }
            pos = nextPos + 1;
        }

        // check if the selection vector is empty
        if (selectionVec.empty())
        {
            _selectionError = SelectionError::EmptySelection;
            return std::nullopt;
        }

        return selectionVec;
    }

}   // namespace input
