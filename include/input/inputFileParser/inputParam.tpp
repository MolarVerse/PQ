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

#ifndef _INPUT_PARAM_TPP_
#define _INPUT_PARAM_TPP_

#include <mstd/type_traits.hpp>

#include "exceptions.hpp"
#include "inputConverter.hpp"
#include "inputParam.hpp"
#include "stringUtilities.hpp"

namespace input
{
    /**
     * @brief Construct a new Input Key object
     *
     * @param registry The key registry containing metadata, default value,
     * allowed values, custom parser, on-set callback, and validator.
     */
    template <typename T>
    InputKey<T>::InputKey(const KeyRegistry<T> &registry)
        : _metadata(registry.metadata),
          _default(registry.defaultValue),
          _allowed(registry.allowed),
          _customParser(registry.customParser),
          _onSet(registry.onSet),
          _validator(registry.validator)
    {
    }

    /**
     * @brief Parses the input line elements and sets the value of the key.
     *
     * @param lineElements The elements of the input line.
     * @param lineNumber The line number in the input file.
     *
     * @throw exc::InputFileException if the value is invalid or violates any
     * constraints.
     */
    template <typename T>
    void InputKey<T>::parse(
        const std::vector<std::string> &lineElements,
        size_t                          lineNumber
    )
    {
        const auto &raw = lineElements.at(2);

        std::optional<T> parsed =
            _customParser ? _customParser(raw) : Converter<T>::tryParse(raw);

        if (!parsed)
        {
            throw exc::InputFileException(
                _metadata.errorMessage
                    ? std::format(
                          "{} at line {} in input file",
                          *_metadata.errorMessage,
                          lineNumber
                      )
                    : std::format(
                          "Invalid value \"{}\" for key \"{}\" at line "
                          "{} in input file. Possible options are: {}",
                          raw,
                          _metadata.name,
                          lineNumber,
                          describeDomain<T>()
                      )
            );
        }

        if (_allowed &&
            std::ranges::find(*_allowed, *parsed) == _allowed->end())
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid value \"{}\" for key \"{}\" at line {} in "
                    "input file: out of allowed range",
                    raw,
                    _metadata.name,
                    lineNumber
                )
            );
        }

        if (_validator && !_validator->validate(*parsed))
        {
            throw exc::InputFileException(
                std::format(
                    "Invalid value \"{}\" for key \"{}\" at line {} in "
                    "input file: failed validation with message {}",
                    raw,
                    _metadata.name,
                    lineNumber,
                    _validator->errorMessage()
                )
            );
        }

        if (_value)
        {
            throw exc::InputFileException(
                std::format(
                    "Multiple keywords \"{}\" in input file",
                    _metadata.name
                )
            );
        }

        _value = std::move(parsed);

        if (_onSet)
            _onSet(*_value);
    }

    /**
     * @brief Returns the name of the input key.
     *
     * @return The name of the input key.
     */
    template <typename T>
    const std::string &InputKey<T>::name() const
    {
        return _metadata.name;
    }

    /**
     * @brief Returns whether the input key has been explicitly set.
     *
     * @return true if the input key has been set, false otherwise.
     */
    template <typename T>
    bool InputKey<T>::isSet() const
    {
        return _value.has_value();
    }

    /**
     * @brief Returns the explicitly set value of the input key, if any.
     *
     * @return The explicitly set value, or std::nullopt if not set.
     */
    template <typename T>
    const std::optional<T> &InputKey<T>::explicitValue() const
    {
        return _value;
    }

    /**
     * @brief Returns the default value of the input key, if any.
     *
     * @return The default value, or std::nullopt if not set.
     */
    template <typename T>
    const std::optional<T> &InputKey<T>::defaultValue() const
    {
        return _default;
    }

    /**
     * @brief what the rest of the program should read: explicit value
     * if given, else default
     *
     * @throw std::logic_error if neither an explicit value nor a
     * default exists
     *
     * @return The value to be used by the rest of the program.
     */
    template <typename T>
    const T &InputKey<T>::value() const
    {
        if (_value)
            return *_value;
        if (_default)
            return *_default;

        throw std::logic_error(
            std::format(
                "Key \"{}\" has neither a set value nor a default",
                _metadata.name
            )
        );
    }

    /**
     * @brief non-throwing variant for genuinely optional keys --
     * caller decides what "absent" means
     *
     * @return The explicitly set value if available, otherwise the default
     * value if available, or std::nullopt if neither is set.
     */
    template <typename T>
    std::optional<T> InputKey<T>::tryValue() const
    {
        if (_value)
            return _value;
        if (_default)
            return _default;

        return std::nullopt;
    }

    /**
     * @brief returns a human-readable description of this key,
     * including its name, title, description, current value,
     * default value, unit, and allowed values
     *
     * @return a string describing this key
     */
    template <typename T>
    std::string InputKey<T>::describe() const
    {
        std::string result =
            std::format("{} ({})", _metadata.name, _metadata.title);

        if (!_metadata.description.empty())
            result += std::format(": {}", _metadata.description);

        if (isSet())
            result += std::format(" = {}", _valueToString(*_value));
        else if (_default)
            result += std::format(" = {} (default)", _valueToString(*_default));
        else
            result += " = <unset>";

        if (_metadata.unit && !_metadata.unit->empty())
            result += std::format(" [{}]", *_metadata.unit);

        if (_allowed)
        {
            std::string allowedStr;
            for (size_t i = 0; i < _allowed->size(); ++i)
            {
                if (i != 0)
                    allowedStr += ", ";
                allowedStr += _valueToString((*_allowed)[i]);
            }
            result += std::format(" [allowed: {}]", allowedStr);
        }

        return result;
    }

    /**
     * @brief clears the current value of the input key
     *
     * This function resets the current value, effectively marking the key
     * as unset.
     */
    template <typename T>
    void InputKey<T>::clearValue()
    {
        _value.reset();
    }

    /**
     * @brief converts a value of type T to a string representation
     *
     * @param value the value to convert
     * @return a string representation of the value
     */
    template <typename T>
    std::string InputKey<T>::_valueToString(const T &value)
    {
        if constexpr (mstd::has_enum_meta<T>)
            return mstd::enum_meta_t<T>::toString(value);
        else if constexpr (std::same_as<T, bool>)
            return value ? "true" : "false";
        else if constexpr (std::same_as<T, mstd::File>)
            return value.fileName();
        else
            return std::format("{}", value);
    }
}   // namespace input

#endif   // _INPUT_PARAM_TPP_
