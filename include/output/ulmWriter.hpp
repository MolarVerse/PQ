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

#ifndef OUTPUT__ULM_WRITER_HPP
#define OUTPUT__ULM_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <span>
#include <string_view>
#include <vector>

#include "textBuffer.hpp"

/**
 * @brief typed writer of the ULM container used by ASE trajectories
 *
 * @details the layout follows version 3 of ase/io/ulm.py
 */
namespace out::ulm
{
    inline constexpr std::string_view FILE_TAG = "- of UlmASE-Trajectory  ";
    inline constexpr std::int64_t     FORMAT_VERSION           = 3;
    inline constexpr std::int64_t     ITEM_COUNT_POSITION      = 32;
    inline constexpr std::int64_t     OFFSETS_POINTER_POSITION = 40;
    inline constexpr std::int64_t     HEADER_SIZE              = 48;
    inline constexpr std::size_t      OFFSETS_GROWTH_FACTOR    = 42;
    inline constexpr std::int64_t     ALIGNMENT                = 8;

    inline constexpr std::size_t MATRIX_DIM  = 3;
    inline constexpr std::size_t MATRIX_SIZE = MATRIX_DIM * MATRIX_DIM;

    /**
     * @brief an array stored in the file, as referenced from the JSON
     */
    struct Array
    {
        std::vector<std::size_t> shape;
        std::string_view         dtype;
        std::int64_t             offset;
    };

    /**
     * @brief writes the header, aligned arrays and JSON items of one file
     * to the stream given to each call
     */
    class Writer
    {
       private:
        std::vector<std::int64_t> _offsets{-1};
        std::vector<char>         _bytes;
        std::int64_t              _offsetsPosition = HEADER_SIZE;
        std::size_t               _nItems          = 0;

       public:
        void writeHeader(std::ostream &stream);
        void writeItem(std::ostream &stream, std::string_view json);

        [[nodiscard]] Array writeArray(
            std::ostream                   &stream,
            std::span<const double>         values,
            const std::vector<std::size_t> &shape
        );
        [[nodiscard]] Array writeArray(
            std::ostream                   &stream,
            std::span<const std::int64_t>   values,
            const std::vector<std::size_t> &shape
        );

        [[nodiscard]] std::size_t getNItems() const;

       private:
        [[nodiscard]] static std::int64_t _getPosition(std::ostream &stream);
        [[nodiscard]] static std::int64_t _align(std::ostream &stream);

        void _writeInt(std::ostream &stream, std::int64_t value);
        void _writeIntAt(
            std::ostream &stream,
            std::int64_t  value,
            std::int64_t  position
        );

        template <typename T>
        void _writeRaw(std::ostream &stream, std::span<const T> values);
    };

    /**
     * @brief builds the JSON text of one item like Python's json.dumps
     */
    class JsonBuilder
    {
       private:
        TextBuffer _buffer;
        bool       _needsComma = false;

       public:
        void begin();
        void end();
        void beginObject(std::string_view key);
        void beginChild(std::string_view key);
        void endObject();

        void addString(std::string_view key, std::string_view text);
        void addInteger(std::string_view key, std::size_t value);
        void addNumber(std::string_view key, double value);
        void addBools(std::string_view key, std::span<const bool> values);
        void addMatrix(
            std::string_view                     key,
            std::span<const double, MATRIX_SIZE> values
        );
        void addArray(std::string_view key, const Array &array);

        [[nodiscard]] std::string_view view() const;

       private:
        void _key(std::string_view key);
        void _number(double value);
    };

}   // namespace out::ulm

#endif   // OUTPUT__ULM_WRITER_HPP
