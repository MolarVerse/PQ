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

#include "jsonOutput.hpp"

#include <iomanip>
#include <ostream>
#include <string>

void cli::writeJsonString(std::ostream &output, const std::string_view value)
{
    output << '"';

    for (const auto character : value)
    {
        switch (character)
        {
            case '"': output << "\\\""; break;
            case '\\': output << "\\\\"; break;
            case '\b': output << "\\b"; break;
            case '\f': output << "\\f"; break;
            case '\n': output << "\\n"; break;
            case '\r': output << "\\r"; break;
            case '\t': output << "\\t"; break;
            default:
                if (static_cast<unsigned char>(character) < 0x20)
                {
                    const auto flags = output.flags();
                    const auto fill  = output.fill();
                    output << "\\u" << std::hex << std::setw(4)
                           << std::setfill('0')
                           << static_cast<unsigned int>(
                                  static_cast<unsigned char>(character)
                              );
                    output.flags(flags);
                    output.fill(fill);
                }
                else
                    output << character;
        }
    }

    output << '"';
}

cli::JsonWriter::JsonWriter(std::ostream &output) : _output(output) {}

void cli::JsonWriter::indent() const
{
    _output << std::string(_depth * 2, ' ');
}

void cli::JsonWriter::beforeValue()
{
    if (_firstValues.empty())
        return;
    if (!_firstValues.back())
        _output << ',';
    _output << '\n';
    indent();
    _firstValues.back() = false;
}

void cli::JsonWriter::beforeMember(const std::string_view key)
{
    beforeValue();
    writeJsonString(_output, key);
    _output << ": ";
}

void cli::JsonWriter::beginContainer(const char opening)
{
    beforeValue();
    _output << opening;
    ++_depth;
    _firstValues.push_back(true);
}

void cli::JsonWriter::beginContainer(
    const std::string_view key,
    const char             opening
)
{
    beforeMember(key);
    _output << opening;
    ++_depth;
    _firstValues.push_back(true);
}

void cli::JsonWriter::endContainer(const char closing)
{
    const auto empty = _firstValues.back();
    _firstValues.pop_back();
    --_depth;

    if (!empty)
    {
        _output << '\n';
        indent();
    }

    _output << closing;
}

void cli::JsonWriter::beginObject() { beginContainer('{'); }

void cli::JsonWriter::beginObject(const std::string_view key)
{
    beginContainer(key, '{');
}

void cli::JsonWriter::endObject() { endContainer('}'); }

void cli::JsonWriter::beginArray() { beginContainer('['); }

void cli::JsonWriter::beginArray(const std::string_view key)
{
    beginContainer(key, '[');
}

void cli::JsonWriter::endArray() { endContainer(']'); }

void cli::JsonWriter::value(const std::string_view value)
{
    beforeValue();
    writeJsonString(_output, value);
}

void cli::JsonWriter::value(const char *value)
{
    this->value(std::string_view(value));
}

void cli::JsonWriter::value(const bool value)
{
    beforeValue();
    _output << (value ? "true" : "false");
}

void cli::JsonWriter::value(const double value)
{
    beforeValue();
    _output << value;
}

void cli::JsonWriter::value(std::nullptr_t)
{
    beforeValue();
    _output << "null";
}

void cli::JsonWriter::value(
    const std::string_view key,
    const std::string_view value
)
{
    beforeMember(key);
    writeJsonString(_output, value);
}

void cli::JsonWriter::value(const std::string_view key, const char *value)
{
    this->value(key, std::string_view(value));
}

void cli::JsonWriter::value(const std::string_view key, const bool value)
{
    beforeMember(key);
    _output << (value ? "true" : "false");
}

void cli::JsonWriter::value(const std::string_view key, const double value)
{
    beforeMember(key);
    _output << value;
}

void cli::JsonWriter::value(const std::string_view key, std::nullptr_t)
{
    beforeMember(key);
    _output << "null";
}
