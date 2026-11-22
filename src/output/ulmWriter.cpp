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
#include <ios>
#include <ostream>
#include <span>
#include <string>
#include <string_view>
#include <utility>

static_assert(
    std::endian::native == std::endian::little,
    "the ULM writer writes little endian data only"
);

namespace out::ulm
{
    /**
     * @brief writer of a ULM file on stream
     *
     * @param stream an empty output stream that supports seeking
     */
    Writer::Writer(std::ostream &stream) : _stream(stream) {}

    Writer::~Writer() = default;

    /**
     * @brief write the header of a file without items
     */
    void Writer::writeHeader()
    {
        _stream.write(
            FILE_TAG.data(),
            static_cast<std::streamsize>(FILE_TAG.size())
        );
        _writeInt(FORMAT_VERSION);
        _writeInt(static_cast<std::int64_t>(_nItems));
        _writeInt(_offsetsPosition);
        _writeRaw(std::span<const std::int64_t>{_offsets});
    }

    /**
     * @brief write a float64 array at the next 8-byte boundary
     *
     * @param values in C order
     * @param shape
     * @return Array
     */
    Array Writer::writeArray(std::span<const double> values, Shape shape)
    {
        const auto offset = _align();
        _writeRaw(values);

        return {
            .shape  = std::move(shape),
            .dtype  = "float64",
            .offset = offset
        };
    }

    /**
     * @brief write an int64 array at the next 8-byte boundary
     *
     * @param values in C order
     * @param shape
     * @return Array
     */
    Array Writer::writeArray(std::span<const std::int64_t> values, Shape shape)
    {
        const auto offset = _align();
        _writeRaw(values);

        return {.shape = std::move(shape), .dtype = "int64", .offset = offset};
    }

    /**
     * @brief append the JSON of one item, register it and flush
     *
     * @details a full offsets table is copied to the end of the file with
     * OFFSETS_GROWTH_FACTOR times its size
     *
     * @param json
     */
    void Writer::writeItem(std::string_view json)
    {
        const auto jsonPosition = _position();
        _writeInt(static_cast<std::int64_t>(json.size()));
        _stream.write(json.data(), static_cast<std::streamsize>(json.size()));

        if (_nItems >= _offsets.size())
        {
            _offsets.resize(_offsets.size() * OFFSETS_GROWTH_FACTOR, 0);
            _offsetsPosition = _align();
            _writeRaw(std::span<const std::int64_t>{_offsets});
            _writeIntAt(_offsetsPosition, OFFSETS_POINTER_POSITION);
        }

        const auto index  = static_cast<std::int64_t>(_nItems);
        _offsets[_nItems] = jsonPosition;
        _writeIntAt(
            jsonPosition,
            _offsetsPosition +
                (index * static_cast<std::int64_t>(sizeof(index)))
        );

        ++_nItems;
        _writeIntAt(static_cast<std::int64_t>(_nItems), ITEM_COUNT_POSITION);

        _stream.flush();
        _stream.seekp(0, std::ios::end);
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
     * @return std::int64_t
     */
    std::int64_t Writer::_position()
    {
        return static_cast<std::int64_t>(_stream.tellp());
    }

    /**
     * @brief pad with '#' to the next multiple of ALIGNMENT bytes
     *
     * @return std::int64_t the aligned position
     */
    std::int64_t Writer::_align()
    {
        const auto position = _position();
        const auto rest     = position % ALIGNMENT;

        if (rest == 0)
        {
            return position;
        }

        const std::string padding(
            static_cast<std::size_t>(ALIGNMENT - rest),
            '#'
        );
        _stream.write(
            padding.data(),
            static_cast<std::streamsize>(padding.size())
        );

        return position + ALIGNMENT - rest;
    }

    /**
     * @brief write an int64 at the current position
     *
     * @param value
     */
    void Writer::_writeInt(std::int64_t value)
    {
        _writeRaw(std::span<const std::int64_t>{&value, 1});
    }

    /**
     * @brief overwrite an int64 at position
     *
     * @param value
     * @param position
     */
    void Writer::_writeIntAt(std::int64_t value, std::int64_t position)
    {
        _stream.seekp(position);
        _writeInt(value);
    }

    /**
     * @brief write values as raw bytes
     *
     * @tparam T
     * @param values
     */
    template <typename T>
    void Writer::_writeRaw(std::span<const T> values)
    {
        _bytes.resize(values.size_bytes());
        std::memcpy(_bytes.data(), values.data(), values.size_bytes());
        _stream.write(
            _bytes.data(),
            static_cast<std::streamsize>(_bytes.size())
        );
    }

    /**
     * @brief start the root object
     */
    void JsonBuilder::begin()
    {
        _buffer.clear();
        _buffer.put('{');
        _needsComma = false;
    }

    /**
     * @brief close the root object
     */
    void JsonBuilder::end() { _buffer.put('}'); }

    /**
     * @brief start a nested object under key
     *
     * @param key
     */
    void JsonBuilder::beginObject(std::string_view key)
    {
        _key(key);
        _buffer.put('{');
        _needsComma = false;
    }

    /**
     * @brief start a ULM child object, stored under key plus "."
     *
     * @param key
     */
    void JsonBuilder::beginChild(std::string_view key)
    {
        _key(std::string(key) + '.');
        _buffer.put('{');
        _needsComma = false;
    }

    /**
     * @brief close a nested object
     */
    void JsonBuilder::endObject()
    {
        _buffer.put('}');
        _needsComma = true;
    }

    /**
     * @brief "key": "text" with quotes and backslashes escaped
     *
     * @param key
     * @param text
     */
    void JsonBuilder::addString(std::string_view key, std::string_view text)
    {
        _key(key);
        _buffer.put('"');
        for (const auto character : text)
        {
            if (character == '"' || character == '\\')
            {
                _buffer.put('\\');
            }

            _buffer.put(character);
        }
        _buffer.put('"');
    }

    /**
     * @brief "key": value
     *
     * @param key
     * @param value
     */
    void JsonBuilder::addInteger(std::string_view key, std::size_t value)
    {
        _key(key);
        _buffer.left(value, 0);
    }

    /**
     * @brief "key": value, read back as the same double
     *
     * @param key
     * @param value
     */
    void JsonBuilder::addNumber(std::string_view key, double value)
    {
        _key(key);
        _number(value);
    }

    /**
     * @brief "key": [true, false, ...]
     *
     * @param key
     * @param values
     */
    void JsonBuilder::addBools(
        std::string_view      key,
        std::span<const bool> values
    )
    {
        _key(key);
        _buffer.put('[');
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index > 0)
            {
                _buffer.append(", ");
            }

            _buffer.append(values[index] ? "true" : "false");
        }
        _buffer.put(']');
    }

    /**
     * @brief "key": [[..], [..], [..]] of a 3x3 matrix in row order
     *
     * @param key
     * @param values
     */
    void JsonBuilder::addMatrix(
        std::string_view                     key,
        std::span<const double, MATRIX_SIZE> values
    )
    {
        _key(key);
        _buffer.put('[');
        for (std::size_t index = 0; index < values.size(); ++index)
        {
            if (index % MATRIX_DIM == 0)
            {
                _buffer.append(index == 0 ? "[" : "], [");
            }
            else
            {
                _buffer.append(", ");
            }

            _number(values[index]);
        }
        _buffer.append("]]");
    }

    /**
     * @brief "key.": {"ndarray": [shape, dtype, offset]} of a stored array
     *
     * @param key
     * @param array
     */
    void JsonBuilder::addArray(std::string_view key, const Array &array)
    {
        _key(std::string(key) + '.');
        _buffer.append(R"({"ndarray": [[)");
        for (std::size_t dim = 0; dim < array.shape.size(); ++dim)
        {
            if (dim > 0)
            {
                _buffer.append(", ");
            }

            _buffer.left(array.shape[dim], 0);
        }
        _buffer.append(R"(], ")").append(array.dtype).append(R"(", )");
        _buffer.left(static_cast<std::size_t>(array.offset), 0);
        _buffer.append("]}");
    }

    /**
     * @brief the JSON text built so far
     *
     * @return std::string_view
     */
    std::string_view JsonBuilder::view() const { return _buffer.view(); }

    /**
     * @brief write the comma before every but the first key of an object
     *
     * @param key
     */
    void JsonBuilder::_key(std::string_view key)
    {
        if (_needsComma)
        {
            _buffer.append(", ");
        }

        _buffer.put('"').append(key).append(R"(": )");
        _needsComma = true;
    }

    /**
     * @brief a double as Python's json module reads it, also NaN and inf
     *
     * @param value
     */
    void JsonBuilder::_number(double value)
    {
        if (std::isnan(value))
        {
            _buffer.append("NaN");
        }
        else if (std::isinf(value))
        {
            _buffer.append(value > 0.0 ? "Infinity" : "-Infinity");
        }
        else
        {
            _buffer.exactScientific(value, 0);
        }
    }

}   // namespace out::ulm
