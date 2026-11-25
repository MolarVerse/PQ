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

#include "ulmWriter.hpp"

#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <format>
#include <ios>
#include <ostream>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <vector>

static_assert(
    std::endian::native == std::endian::little,
    "the ULM writer writes little endian data only"
);

namespace out::ulm
{
    /**
     * @brief write the header of a file without items
     *
     * @param stream an empty output stream that supports seeking
     */
    void Writer::writeHeader(std::ostream &stream)
    {
        stream.write(
            FILE_TAG.data(),
            static_cast<std::streamsize>(FILE_TAG.size())
        );
        _writeInt(stream, FORMAT_VERSION);
        _writeInt(stream, static_cast<std::int64_t>(_nItems));
        _writeInt(stream, _offsetsPosition);
        _writeRaw(stream, std::span<const std::int64_t>{_offsets});
    }

    /**
     * @brief write a float64 array at the next 8-byte boundary
     *
     * @param stream
     * @param values in C order
     * @param shape
     * @return Array
     */
    Array Writer::writeArray(
        std::ostream                   &stream,
        std::span<const double>         values,
        const std::vector<std::size_t> &shape
    )
    {
        const auto offset = _align(stream);
        _writeRaw(stream, values);

        return {.shape = shape, .dtype = "float64", .offset = offset};
    }

    /**
     * @brief write an int64 array at the next 8-byte boundary
     *
     * @param stream
     * @param values in C order
     * @param shape
     * @return Array
     */
    Array Writer::writeArray(
        std::ostream                   &stream,
        std::span<const std::int64_t>   values,
        const std::vector<std::size_t> &shape
    )
    {
        const auto offset = _align(stream);
        _writeRaw(stream, values);

        return {.shape = shape, .dtype = "int64", .offset = offset};
    }

    /**
     * @brief append the JSON of one item, register it and flush
     *
     * @details a full offsets table is copied to the end of the file with
     * OFFSETS_GROWTH_FACTOR times its size
     *
     * @param stream
     * @param json
     */
    void Writer::writeItem(std::ostream &stream, std::string_view json)
    {
        const auto jsonPosition = _getPosition(stream);
        _writeInt(stream, static_cast<std::int64_t>(json.size()));
        stream.write(json.data(), static_cast<std::streamsize>(json.size()));

        if (_nItems >= _offsets.size())
        {
            _offsets.resize(_offsets.size() * OFFSETS_GROWTH_FACTOR, 0);
            _offsetsPosition = _align(stream);
            _writeRaw(stream, std::span<const std::int64_t>{_offsets});
            _writeIntAt(stream, _offsetsPosition, OFFSETS_POINTER_POSITION);
        }

        const auto index  = static_cast<std::int64_t>(_nItems);
        _offsets[_nItems] = jsonPosition;
        _writeIntAt(
            stream,
            jsonPosition,
            _offsetsPosition +
                (index * static_cast<std::int64_t>(sizeof(index)))
        );

        ++_nItems;
        _writeIntAt(
            stream,
            static_cast<std::int64_t>(_nItems),
            ITEM_COUNT_POSITION
        );

        stream.flush();
        stream.seekp(0, std::ios::end);
    }

    /**
     * @brief number of items written
     *
     * @return std::size_t
     */
    std::size_t Writer::getNItems() const { return _nItems; }

    /**
     * @brief current write position
     *
     * @param stream
     * @return std::int64_t
     */
    std::int64_t Writer::_getPosition(std::ostream &stream)
    {
        return static_cast<std::int64_t>(stream.tellp());
    }

    /**
     * @brief pad with '#' to the next multiple of ALIGNMENT bytes
     *
     * @param stream
     * @return std::int64_t the aligned position
     */
    std::int64_t Writer::_align(std::ostream &stream)
    {
        const auto position = _getPosition(stream);
        const auto rest     = position % ALIGNMENT;

        if (rest == 0)
        {
            return position;
        }

        const std::string padding(
            static_cast<std::size_t>(ALIGNMENT - rest),
            '#'
        );
        stream.write(
            padding.data(),
            static_cast<std::streamsize>(padding.size())
        );

        return position + ALIGNMENT - rest;
    }

    /**
     * @brief write an int64 at the current position
     *
     * @param stream
     * @param value
     */
    void Writer::_writeInt(std::ostream &stream, std::int64_t value)
    {
        _writeRaw(stream, std::span<const std::int64_t>{&value, 1});
    }

    /**
     * @brief overwrite an int64 at position
     *
     * @param stream
     * @param value
     * @param position
     */
    void Writer::_writeIntAt(
        std::ostream &stream,
        std::int64_t  value,
        std::int64_t  position
    )
    {
        stream.seekp(position);
        _writeInt(stream, value);
    }

    /**
     * @brief write values as raw bytes
     *
     * @tparam T
     * @param stream
     * @param values
     */
    template <typename T>
    void Writer::_writeRaw(std::ostream &stream, std::span<const T> values)
    {
        _bytes.resize(values.size_bytes());
        std::memcpy(_bytes.data(), values.data(), values.size_bytes());
        stream.write(
            _bytes.data(),
            static_cast<std::streamsize>(_bytes.size())
        );
    }

    /**
     * @brief "key": "text" with quotes and backslashes escaped
     *
     * @param key
     * @param text
     */
    void JsonObject::addString(std::string_view key, std::string_view text)
    {
        _value.clear();
        _value.put('"');
        for (const auto character : text)
        {
            if (character == '"' || character == '\\')
            {
                _value.put('\\');
            }

            _value.put(character);
        }
        _value.put('"');
        _add(key);
    }

    /**
     * @brief "key": value
     *
     * @param key
     * @param value
     */
    void JsonObject::addInteger(std::string_view key, std::size_t value)
    {
        _value.clear();
        _value.left(value, 0);
        _add(key);
    }

    /**
     * @brief "key": value, read back as the same double
     *
     * @param key
     * @param value
     */
    void JsonObject::addNumber(std::string_view key, double value)
    {
        _value.clear();
        _number(value);
        _add(key);
    }

    /**
     * @brief "key": [true, false, ...]
     *
     * @param key
     * @param values
     */
    void JsonObject::addBools(
        std::string_view      key,
        std::span<const bool> values
    )
    {
        _value.clear();
        _value.put('[');
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index > 0)
            {
                _value.append(", ");
            }

            _value.append(values[index] ? "true" : "false");
        }
        _value.put(']');
        _add(key);
    }

    /**
     * @brief "key": [[..], [..], [..]] of a 3x3 matrix in row order
     *
     * @param key
     * @param values
     */
    void JsonObject::addMatrix(
        std::string_view                     key,
        std::span<const double, MATRIX_SIZE> values
    )
    {
        _value.clear();
        _value.put('[');
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index % MATRIX_DIM == 0)
            {
                _value.append(index == 0 ? "[" : "], [");
            }
            else
            {
                _value.append(", ");
            }

            _number(values[index]);
        }
        _value.append("]]");
        _add(key);
    }

    /**
     * @brief "key.": {"ndarray": [shape, dtype, offset]} of a stored array
     *
     * @param key
     * @param array
     */
    void JsonObject::addArray(std::string_view key, const Array &array)
    {
        _value.clear();
        _value.append(R"({"ndarray": [[)");
        for (std::size_t dim = 0; dim < array.shape.size(); ++dim)
        {
            if (dim > 0)
            {
                _value.append(", ");
            }

            _value.left(array.shape[dim], 0);
        }
        _value.append(R"(], ")").append(array.dtype).append(R"(", )");
        _value.left(static_cast<std::size_t>(array.offset), 0);
        _value.append("]}");
        _add(std::string(key) + '.');
    }

    /**
     * @brief "key": {...} of a nested object
     *
     * @param key
     * @param object
     */
    void JsonObject::addObject(std::string_view key, const JsonObject &object)
    {
        _value.clear();
        _value.append(object.str());
        _add(key);
    }

    /**
     * @brief "key.": {...} of a ULM child object
     *
     * @param key
     * @param object
     */
    void JsonObject::addChild(std::string_view key, const JsonObject &object)
    {
        addObject(std::string(key) + '.', object);
    }

    /**
     * @brief the object with all members in the order they were added
     *
     * @return std::string
     */
    std::string JsonObject::str() const
    {
        std::string text{"{"};

        if (!_members.empty())
        {
            text += _members.front();
        }

        for (const auto &member : _members | std::views::drop(1))
        {
            text += ", ";
            text += member;
        }

        text += '}';
        return text;
    }

    /**
     * @brief store the value formatted in _value as member key
     *
     * @param key
     */
    void JsonObject::_add(std::string_view key)
    {
        _members.push_back(std::format(R"("{}": {})", key, _value.view()));
    }

    /**
     * @brief a double as Python's json module reads it, also NaN and inf
     *
     * @param value
     */
    void JsonObject::_number(double value)
    {
        if (std::isnan(value))
        {
            _value.append("NaN");
        }
        else if (std::isinf(value))
        {
            _value.append(value > 0.0 ? "Infinity" : "-Infinity");
        }
        else
        {
            _value.exactScientific(value, 0);
        }
    }

}   // namespace out::ulm
