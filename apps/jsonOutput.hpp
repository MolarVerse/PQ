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

#ifndef _JSON_OUTPUT_HPP_

#define _JSON_OUTPUT_HPP_

#include <concepts>
#include <cstddef>
#include <ostream>
#include <string_view>
#include <type_traits>
#include <vector>

namespace cli
{
    void writeJsonString(std::ostream &output, std::string_view value);

    class JsonWriter
    {
       private:
        std::ostream     &_output;
        std::size_t       _depth = 0;
        std::vector<bool> _firstValues;

        void indent() const;
        void beforeValue();
        void beforeMember(std::string_view key);
        void beginContainer(char opening);
        void beginContainer(std::string_view key, char opening);
        void endContainer(char closing);

       public:
        explicit JsonWriter(std::ostream &output);

        void beginObject();
        void beginObject(std::string_view key);
        void endObject();

        void beginArray();
        void beginArray(std::string_view key);
        void endArray();

        void value(std::string_view value);
        void value(const char *value);
        void value(bool value);
        void value(double value);
        void value(std::nullptr_t);

        void value(std::string_view key, std::string_view value);
        void value(std::string_view key, const char *value);
        void value(std::string_view key, bool value);
        void value(std::string_view key, double value);
        void value(std::string_view key, std::nullptr_t);

        template <std::integral T>
        requires(!std::same_as<std::remove_cv_t<T>, bool>)
        void value(const T value)
        {
            beforeValue();
            _output << value;
        }

        template <std::integral T>
        requires(!std::same_as<std::remove_cv_t<T>, bool>)
        void value(const std::string_view key, const T value)
        {
            beforeMember(key);
            _output << value;
        }
    };
}   // namespace cli

#endif   // _JSON_OUTPUT_HPP_
