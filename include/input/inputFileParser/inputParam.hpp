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

#ifndef _INPUT_PARAM_HPP_
#define _INPUT_PARAM_HPP_

#include <algorithm>
#include <concepts>
#include <format>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "exceptions.hpp"
#include "inputConverter.hpp"

namespace input
{
    /**
     * @struct KeyMetadata
     *
     * @brief everything needed to describe a key to a human -- the
     * input-file token itself plus documentation fields
     *
     * @details deliberately separate from parsing behavior (default value,
     * allowed values, customParser, onSet) -- this is what gets shown, not
     * how parsing happens
     *
     */
    struct KeyMetadata
    {
        std::string                name;
        std::string                title;
        std::string                description;
        std::string                unit;
        std::optional<std::string> errorMessage = std::nullopt;
    };

    /**
     * @class InputKeyBase
     *
     * @brief type-erased base for InputKey<T>, so keys of different T can
     * live in the same InputRegistry
     *
     */
    class InputKeyBase
    {
       public:
        virtual ~InputKeyBase() = default;

        virtual void parse(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        ) = 0;

        [[nodiscard]] virtual const std::string &name() const noexcept = 0;

        [[nodiscard]] virtual bool isSet() const noexcept = 0;

        /**
         * @brief human-readable summary: key, title, description, unit,
         * current value, default, and allowed set where applicable
         *
         * @details for docs, startup config dumps, or diagnostics -- not
         * used during parsing
         */
        [[nodiscard]] virtual std::string describe() const = 0;
    };

    /**
     * @class InputKey
     *
     * @brief owns key identity, default/actual value storage, and the
     * allowed-values restriction for a single input-file key
     *
     * @details delegates raw-token -> T conversion entirely to
     * Converter<T> (or a per-key customParser override) -- no parsing
     * logic lives here
     *
     * @tparam T
     */
    template <typename T>
    class InputKey : public InputKeyBase
    {
       public:
        using CustomParser = std::function<std::optional<T>(std::string_view)>;

       private:
        KeyMetadata                    _metadata;
        std::optional<T>               _default;
        std::optional<T>               _value;
        std::optional<std::vector<T>>  _allowed;
        CustomParser                   _customParser;
        std::function<void(const T &)> _onSet;

       public:
        // NOLINTBEGIN(fuchsia-default-arguments-declarations)
        // NOTE: here default arguments make really sense
        explicit InputKey(
            KeyMetadata                    metadata,
            std::optional<T>               defaultValue = std::nullopt,
            std::optional<std::vector<T>>  allowed      = std::nullopt,
            CustomParser                   customParser = nullptr,
            std::function<void(const T &)> onSet        = nullptr
        )
            : _metadata(std::move(metadata)),
              _default(std::move(defaultValue)),
              _allowed(std::move(allowed)),
              _customParser(std::move(customParser)),
              _onSet(std::move(onSet))
        {
        }
        // NOLINTEND(fuchsia-default-arguments-declarations)

        void parse(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        ) override
        {
            const auto &raw = lineElements.at(2);

            std::optional<T> parsed = _customParser
                                          ? _customParser(raw)
                                          : Converter<T>::tryParse(raw);

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

            // only parse() ever touches _value -- _default is untouched
            _value = std::move(parsed);

            if (_onSet)
                _onSet(*_value);
        }

        [[nodiscard]] const std::string &name() const noexcept override
        {
            return _metadata.name;
        }

        /**
         * @brief true iff a line in the input file actually set this key
         * -- a default never counts
         */
        [[nodiscard]] bool isSet() const noexcept override
        {
            return _value.has_value();
        }

        [[nodiscard]] const std::optional<T> &explicitValue() const noexcept
        {
            return _value;
        }

        [[nodiscard]] const std::optional<T> &defaultValue() const noexcept
        {
            return _default;
        }

        /**
         * @brief what the rest of the program should read: explicit value
         * if given, else default
         *
         * @throw std::logic_error if neither an explicit value nor a
         * default exists
         */
        [[nodiscard]] const T &value() const
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
         */
        [[nodiscard]] std::optional<T> tryValue() const noexcept
        {
            if (_value)
                return _value;
            if (_default)
                return _default;

            return std::nullopt;
        }

        [[nodiscard]] std::string describe() const override
        {
            std::string result =
                std::format("{} ({})", _metadata.name, _metadata.title);

            if (!_metadata.description.empty())
                result += std::format(": {}", _metadata.description);

            if (isSet())
                result += std::format(" = {}", valueToString(*_value));
            else if (_default)
                result +=
                    std::format(" = {} (default)", valueToString(*_default));
            else
                result += " = <unset>";

            if (!_metadata.unit.empty())
                result += std::format(" [{}]", _metadata.unit);

            if (_allowed)
            {
                std::string allowedStr;
                for (size_t i = 0; i < _allowed->size(); ++i)
                {
                    if (i != 0)
                        allowedStr += ", ";
                    allowedStr += valueToString((*_allowed)[i]);
                }
                result += std::format(" [allowed: {}]", allowedStr);
            }

            return result;
        }

       private:
        [[nodiscard]]
        static std::string valueToString(const T &value)
        {
            if constexpr (mstd::has_enum_meta<T>)
                return mstd::enum_meta_t<T>::toString(value);
            else if constexpr (std::same_as<T, bool>)
                return value ? "true" : "false";
            else
                return std::format("{}", value);
        }
    };

    /**
     * @class InputRegistry
     *
     * @brief owns a set of InputKey<T> instances for one InputFileParser
     * subclass: register once with defaults, parse lines, query later
     *
     */
    class InputRegistry
    {
       private:
        std::unordered_map<std::string, std::unique_ptr<InputKeyBase>> _keys;

       public:
        // NOLINTBEGIN(fuchsia-default-arguments-declarations)
        // NOTE: here default arguments make really sense
        template <typename T>
        InputKey<T> &registerKey(
            KeyMetadata                        metadata,
            std::optional<T>                   defaultValue = std::nullopt,
            std::optional<std::vector<T>>      allowed      = std::nullopt,
            typename InputKey<T>::CustomParser customParser = nullptr,
            std::function<void(const T &)>     onSet        = nullptr
        )
        {
            const std::string name = metadata.name;

            auto key = std::make_unique<InputKey<T>>(
                std::move(metadata),
                std::move(defaultValue),
                std::move(allowed),
                std::move(customParser),
                std::move(onSet)
            );

            auto &ref = *key;
            _keys.emplace(name, std::move(key));
            return ref;
        }
        // NOLINTEND(fuchsia-default-arguments-declarations)

        void parseLine(
            const std::vector<std::string> &lineElements,
            size_t                          lineNumber
        )
        {
            const auto &key = lineElements.at(0);
            auto        it  = _keys.find(key);

            if (it == _keys.end())
            {
                throw exc::InputFileException(
                    std::format(
                        "Unknown key \"{}\" at line {} in input file",
                        key,
                        lineNumber
                    )
                );
            }

            it->second->parse(lineElements, lineNumber);
        }

        template <typename T>
        [[nodiscard]] const InputKey<T> &get(const std::string &name) const
        {
            return dynamic_cast<const InputKey<T> &>(*_keys.at(name));
        }

        /**
         * @brief full inventory of every registered key in this registry,
         * e.g. for a startup configuration dump or generated documentation
         */
        [[nodiscard]] std::vector<std::string> describeAll() const
        {
            std::vector<std::string> result;
            result.reserve(_keys.size());

            for (const auto &[name, key] : _keys)
                result.push_back(key->describe());

            return result;
        }
    };
}   // namespace input

#endif   // _INPUT_PARAM_HPP_
