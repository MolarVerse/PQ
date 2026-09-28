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
            {
                constexpr auto maxHex = 0x20;
                if (static_cast<unsigned char>(character) < maxHex)
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
                {
                    output << character;
                }
            }
        }
    }

    output << '"';
}

cli::JsonWriter::JsonWriter(std::ostream &output) : _output(output) {}

void cli::JsonWriter::_indent() const
{
    _output << std::string(_depth * 2, ' ');
}

void cli::JsonWriter::_beforeValue()
{
    if (_firstValues.empty())
        return;
    if (!_firstValues.back())
        _output << ',';
    _output << '\n';
    _indent();
    _firstValues.back() = false;
}

void cli::JsonWriter::_beforeMember(const std::string_view key)
{
    _beforeValue();
    writeJsonString(_output, key);
    _output << ": ";
}

void cli::JsonWriter::_beginContainer(const char opening)
{
    _beforeValue();
    _output << opening;
    ++_depth;
    _firstValues.push_back(true);
}

void cli::JsonWriter::_beginContainer(
    const std::string_view key,
    const char             opening
)
{
    _beforeMember(key);
    _output << opening;
    ++_depth;
    _firstValues.push_back(true);
}

void cli::JsonWriter::_endContainer(const char closing)
{
    const auto empty = _firstValues.back();
    _firstValues.pop_back();
    --_depth;

    if (!empty)
    {
        _output << '\n';
        _indent();
    }

    _output << closing;
}

void cli::JsonWriter::beginObject() { _beginContainer('{'); }

void cli::JsonWriter::beginObject(const std::string_view key)
{
    _beginContainer(key, '{');
}

void cli::JsonWriter::endObject() { _endContainer('}'); }

void cli::JsonWriter::beginArray() { _beginContainer('['); }

void cli::JsonWriter::beginArray(const std::string_view key)
{
    _beginContainer(key, '[');
}

void cli::JsonWriter::endArray() { _endContainer(']'); }

void cli::JsonWriter::value(const std::string_view value)
{
    _beforeValue();
    writeJsonString(_output, value);
}

void cli::JsonWriter::value(const char *value)
{
    this->value(std::string_view(value));
}

void cli::JsonWriter::value(const bool value)
{
    _beforeValue();
    _output << (value ? "true" : "false");
}

void cli::JsonWriter::value(const double value)
{
    _beforeValue();
    _output << value;
}

void cli::JsonWriter::value(std::nullptr_t)
{
    _beforeValue();
    _output << "null";
}

// NOLINTBEGIN(bugprone-easily-swappable-parameters) -- here fine as it is the
// specialized overload for string_view as second type
void cli::JsonWriter::value(
    const std::string_view key,
    const std::string_view value
)
{
    _beforeMember(key);
    writeJsonString(_output, value);
}
// NOLINTEND(bugprone-easily-swappable-parameters)

void cli::JsonWriter::value(const std::string_view key, const char *value)
{
    this->value(key, std::string_view(value));
}

void cli::JsonWriter::value(const std::string_view key, const bool value)
{
    _beforeMember(key);
    _output << (value ? "true" : "false");
}

void cli::JsonWriter::value(const std::string_view key, const double value)
{
    _beforeMember(key);
    _output << value;
}

void cli::JsonWriter::value(const std::string_view key, std::nullptr_t)
{
    _beforeMember(key);
    _output << "null";
}
