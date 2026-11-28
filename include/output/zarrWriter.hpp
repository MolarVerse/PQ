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

#ifndef OUTPUT__ZARR_WRITER_HPP
#define OUTPUT__ZARR_WRITER_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "textBuffer.hpp"
#include "unit_types.hpp"

/**
 * @brief typed writer of Zarr v3 stores: zarr.json metadata and chunk files
 * with the bytes and, if PQ is built with zstd, the zstd codec
 */
namespace out::zarr
{
    enum class Codec : std::uint8_t
    {
        BYTES,
        ZSTD
    };

    [[nodiscard]] bool zstdAvailable();

    /**
     * @brief JSON object of a zarr.json file
     *
     * @details every add appends one complete "key": value member; str()
     * joins the members with ", "
     */
    class JsonObject
    {
       private:
        std::vector<std::string> _members;
        TextBuffer               _value;

       public:
        void addString(std::string_view key, std::string_view text);
        void addStrings(
            std::string_view                key,
            const std::vector<std::string> &texts
        );
        void addInteger(std::string_view key, std::size_t value);
        void addIntegers(
            std::string_view                key,
            const std::vector<std::size_t> &values
        );
        void addBool(std::string_view key, bool value);
        void addObject(std::string_view key, const JsonObject &object);
        void addObjects(
            std::string_view               key,
            const std::vector<JsonObject> &objects
        );

        [[nodiscard]] std::string str() const;

       private:
        void _add(std::string_view key);
        void _string(std::string_view text);
    };

    [[nodiscard]] JsonObject unitAttribute(std::string_view unit);

    void writeFile(const std::filesystem::path &path, std::string_view content);
    void writeGroup(
        const std::filesystem::path &path,
        const JsonObject            &attributes
    );

    [[nodiscard]] std::vector<JsonObject> codecs(Codec codec);
    [[nodiscard]] std::string encode(std::string_view bytes, Codec codec);

    template <typename T>
    [[nodiscard]] std::string_view dataType();

    template <typename T>
    [[nodiscard]] JsonObject arrayMetadata(
        const std::vector<std::size_t> &shape,
        const std::vector<std::size_t> &chunkShape,
        Codec                           codec,
        const JsonObject               &attributes
    );

    template <typename T>
    void writeArray(
        const std::filesystem::path &path,
        std::span<const T>           values,
        Codec                        codec
    );

    /**
     * @brief array that grows by one frame per append along its first axis
     *
     * @details frames are kept until a chunk of chunkFrames is full; then
     * the chunk file and the new shape in zarr.json are written
     */
    template <typename T>
    class Array
    {
       private:
        std::filesystem::path    _path;
        std::vector<std::size_t> _frameShape;
        std::size_t              _frameSize;
        std::size_t              _chunkFrames;
        Codec                    _codec;
        JsonObject               _attributes;
        std::vector<T>           _chunk;
        std::size_t              _nFrames = 0;

       public:
        Array(
            std::filesystem::path           path,
            const std::vector<std::size_t> &frameShape,
            std::size_t                     chunkFrames,
            Codec                           codec,
            JsonObject                      attributes
        );

        void append(std::span<const T> frame);
        void flush();

        [[nodiscard]] std::size_t getNumberOfFrames() const;

       private:
        void _writeChunk();
        void _writeMetadata() const;
    };

    /**
     * @brief H5MD time-dependent element: a group with step, time and value
     * arrays
     */
    template <typename T>
    class Element
    {
       private:
        Array<std::int64_t> _step;
        Array<double>       _time;
        Array<T>            _value;

       public:
        Element(
            const std::filesystem::path    &path,
            const std::vector<std::size_t> &frameShape,
            std::size_t                     chunkFrames,
            Codec                           codec,
            std::string_view                unit
        );

        void append(std::int64_t step, Time time, std::span<const T> value);
        void flush();
    };

}   // namespace out::zarr

#endif   // OUTPUT__ZARR_WRITER_HPP
