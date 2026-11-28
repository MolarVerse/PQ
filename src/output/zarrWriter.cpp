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

#include "zarrWriter.hpp"

#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <format>
#include <fstream>
#include <functional>
#include <numeric>
#include <ranges>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "textBuffer.hpp"
#include "unit_types.hpp"
#include "units.hpp"

#ifdef WITH_ZSTD
#include <zstd.h>
#endif

// the bytes codec says little endian and chunks are copied from memory
static_assert(std::endian::native == std::endian::little);

namespace out::zarr
{
    /**
     * @brief whether PQ was built with zstd
     *
     * @return bool
     */
    bool zstdAvailable()
    {
#ifdef WITH_ZSTD
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief "key": "text"
     *
     * @param key
     * @param text
     */
    void JsonObject::addString(std::string_view key, std::string_view text)
    {
        _value.clear();
        _string(text);
        _add(key);
    }

    /**
     * @brief "key": ["text", ...]
     *
     * @param key
     * @param texts
     */
    void JsonObject::addStrings(
        std::string_view                key,
        const std::vector<std::string> &texts
    )
    {
        _value.clear();
        _value.put('[');
        for (const auto &[index, text] : std::views::enumerate(texts))
        {
            if (index > 0)
            {
                _value.append(", ");
            }

            _string(text);
        }
        _value.put(']');
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
     * @brief "key": [value, ...]
     *
     * @param key
     * @param values
     */
    void JsonObject::addIntegers(
        std::string_view                key,
        const std::vector<std::size_t> &values
    )
    {
        _value.clear();
        _value.put('[');
        for (const auto &[index, value] : std::views::enumerate(values))
        {
            if (index > 0)
            {
                _value.append(", ");
            }

            _value.left(value, 0);
        }
        _value.put(']');
        _add(key);
    }

    /**
     * @brief "key": true or false
     *
     * @param key
     * @param value
     */
    void JsonObject::addBool(std::string_view key, bool value)
    {
        _value.clear();
        _value.append(value ? "true" : "false");
        _add(key);
    }

    /**
     * @brief "key": {...}
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
     * @brief "key": [{...}, ...]
     *
     * @param key
     * @param objects
     */
    void JsonObject::addObjects(
        std::string_view               key,
        const std::vector<JsonObject> &objects
    )
    {
        _value.clear();
        _value.put('[');
        for (const auto &[index, object] : std::views::enumerate(objects))
        {
            if (index > 0)
            {
                _value.append(", ");
            }

            _value.append(object.str());
        }
        _value.put(']');
        _add(key);
    }

    /**
     * @brief the object with all members in the order they were added
     *
     * @return std::string
     */
    std::string JsonObject::str() const
    {
        std::string text{"{"};

        for (const auto &[index, member] : std::views::enumerate(_members))
        {
            if (index > 0)
            {
                text += ", ";
            }

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
     * @brief text in quotes with quotes and backslashes escaped
     *
     * @param text
     */
    void JsonObject::_string(std::string_view text)
    {
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
    }

    /**
     * @brief attributes {"unit": unit} of an H5MD value or time array
     *
     * @param unit
     *
     * @return JsonObject
     */
    JsonObject unitAttribute(std::string_view unit)
    {
        JsonObject attributes;
        attributes.addString("unit", unit);
        return attributes;
    }

    /**
     * @brief replace the file at path by content
     *
     * @details written next to it and renamed, so a reader or a crash never
     * sees half a file
     *
     * @param path
     * @param content
     */
    void writeFile(const std::filesystem::path &path, std::string_view content)
    {
        auto partial = path;
        partial += ".partial";

        {
            std::ofstream file{partial, std::ios::binary | std::ios::trunc};
            if (!file.is_open())
            {
                throw exc::InputFileException(
                    "Could not write Zarr file - filename = " + path.string()
                );
            }

            file.write(
                content.data(),
                static_cast<std::streamsize>(content.size())
            );
        }

        std::filesystem::rename(partial, path);
    }

    /**
     * @brief create the group directory and its zarr.json
     *
     * @param path
     * @param attributes
     */
    void writeGroup(
        const std::filesystem::path &path,
        const JsonObject            &attributes
    )
    {
        JsonObject metadata;
        metadata.addInteger("zarr_format", 3);
        metadata.addString("node_type", "group");
        metadata.addObject("attributes", attributes);

        std::filesystem::create_directories(path);
        writeFile(path / "zarr.json", metadata.str());
    }

    /**
     * @brief the codecs entry of zarr.json
     *
     * @param codec
     *
     * @return std::vector<JsonObject>
     *
     * @throw CompileTimeException for zstd if PQ was built without it
     */
    std::vector<JsonObject> codecs(Codec codec)
    {
        JsonObject endian;
        endian.addString("endian", "little");

        JsonObject bytes;
        bytes.addString("name", "bytes");
        bytes.addObject("configuration", endian);

        if (codec == Codec::BYTES)
        {
            return {bytes};
        }

#ifdef WITH_ZSTD
        JsonObject level;
        level.addInteger("level", static_cast<std::size_t>(ZSTD_CLEVEL_DEFAULT));
        level.addBool("checksum", false);

        JsonObject zstd;
        zstd.addString("name", "zstd");
        zstd.addObject("configuration", level);

        return {bytes, zstd};
#else
        throw exc::CompileTimeException(
            "Zarr zstd codec needs PQ built with -DBUILD_WITH_ZSTD=ON"
        );
#endif
    }

    /**
     * @brief chunk file content of the raw little endian bytes
     *
     * @param bytes
     * @param codec
     *
     * @return std::string
     *
     * @throw CompileTimeException for zstd if PQ was built without it
     */
    std::string encode(std::string_view bytes, Codec codec)
    {
        if (codec == Codec::BYTES)
        {
            return std::string{bytes};
        }

#ifdef WITH_ZSTD
        std::string compressed(ZSTD_compressBound(bytes.size()), '\0');

        const auto size = ZSTD_compress(
            compressed.data(),
            compressed.size(),
            bytes.data(),
            bytes.size(),
            ZSTD_CLEVEL_DEFAULT
        );
        if (ZSTD_isError(size) != 0U)
        {
            throw exc::InputFileException(
                std::string{"zstd failed: "} + ZSTD_getErrorName(size)
            );
        }

        compressed.resize(size);
        return compressed;
#else
        throw exc::CompileTimeException(
            "Zarr zstd codec needs PQ built with -DBUILD_WITH_ZSTD=ON"
        );
#endif
    }

    template <>
    std::string_view dataType<std::int32_t>()
    {
        return "int32";
    }

    template <>
    std::string_view dataType<std::int64_t>()
    {
        return "int64";
    }

    template <>
    std::string_view dataType<float>()
    {
        return "float32";
    }

    template <>
    std::string_view dataType<double>()
    {
        return "float64";
    }

    /**
     * @brief zarr.json of a regular chunked array with fill value 0
     *
     * @param shape
     * @param chunkShape
     * @param codec
     * @param attributes
     *
     * @return JsonObject
     */
    template <typename T>
    JsonObject arrayMetadata(
        const std::vector<std::size_t> &shape,
        const std::vector<std::size_t> &chunkShape,
        Codec                           codec,
        const JsonObject               &attributes
    )
    {
        JsonObject chunkConfiguration;
        chunkConfiguration.addIntegers("chunk_shape", chunkShape);

        JsonObject chunkGrid;
        chunkGrid.addString("name", "regular");
        chunkGrid.addObject("configuration", chunkConfiguration);

        JsonObject separator;
        separator.addString("separator", "/");

        JsonObject chunkKeyEncoding;
        chunkKeyEncoding.addString("name", "default");
        chunkKeyEncoding.addObject("configuration", separator);

        JsonObject metadata;
        metadata.addInteger("zarr_format", 3);
        metadata.addString("node_type", "array");
        metadata.addIntegers("shape", shape);
        metadata.addString("data_type", dataType<T>());
        metadata.addObject("chunk_grid", chunkGrid);
        metadata.addObject("chunk_key_encoding", chunkKeyEncoding);
        metadata.addInteger("fill_value", 0);
        metadata.addObjects("codecs", codecs(codec));
        metadata.addObject("attributes", attributes);

        return metadata;
    }

    /**
     * @brief write a 1-D array of fixed size in one chunk
     *
     * @param path
     * @param values
     * @param codec
     */
    template <typename T>
    void writeArray(
        const std::filesystem::path &path,
        std::span<const T>           values,
        Codec                        codec
    )
    {
        const std::vector<std::size_t> shape{values.size()};
        const auto metadata = arrayMetadata<T>(shape, shape, codec, {});

        std::string bytes(values.size_bytes(), '\0');
        std::memcpy(bytes.data(), values.data(), values.size_bytes());

        std::filesystem::create_directories(path / "c");
        writeFile(path / "c" / "0", encode(bytes, codec));
        writeFile(path / "zarr.json", metadata.str());
    }

    /**
     * @brief create the array directory with zarr.json of zero frames
     *
     * @param path
     * @param frameShape shape of one frame, empty for a scalar
     * @param chunkFrames frames per chunk
     * @param codec
     * @param attributes
     *
     * @throw CompileTimeException for zstd if PQ was built without it
     */
    template <typename T>
    Array<T>::Array(
        std::filesystem::path           path,
        const std::vector<std::size_t> &frameShape,
        std::size_t                     chunkFrames,
        Codec                           codec,
        const JsonObject               &attributes
    )
        : _path(std::move(path)),
          _frameShape(frameShape),
          _frameSize(std::accumulate(
              frameShape.begin(),
              frameShape.end(),
              std::size_t{1},
              std::multiplies{}
          )),
          _chunkFrames(chunkFrames),
          _codec(codec),
          _attributes(attributes)
    {
        _chunk.reserve(_chunkFrames * _frameSize);

        std::filesystem::create_directories(_path);
        _writeMetadata();
    }

    /**
     * @brief append one frame; a full chunk is written to disk
     *
     * @param frame values of one frame in row-major order
     *
     * @throw InputFileException if frame does not match the frame shape
     */
    template <typename T>
    void Array<T>::append(std::span<const T> frame)
    {
        if (frame.size() != _frameSize)
        {
            throw exc::InputFileException(
                "Zarr frame has " + std::to_string(frame.size()) +
                " values, expected " + std::to_string(_frameSize)
            );
        }

        _chunk.insert(_chunk.end(), frame.begin(), frame.end());
        ++_nFrames;

        if (_chunk.size() == _chunkFrames * _frameSize)
        {
            _writeChunk();
            _writeMetadata();
            _chunk.clear();
        }
    }

    /**
     * @brief write the frames of the chunk that is not full yet
     */
    template <typename T>
    void Array<T>::flush()
    {
        if (_chunk.empty())
        {
            return;
        }

        _writeChunk();
        _writeMetadata();
    }

    /**
     * @brief number of appended frames
     *
     * @return std::size_t
     */
    template <typename T>
    std::size_t Array<T>::getNumberOfFrames() const
    {
        return _nFrames;
    }

    /**
     * @brief write the current chunk, padded with zeros to the full chunk
     */
    template <typename T>
    void Array<T>::_writeChunk()
    {
        const auto chunkIndex = (_nFrames - 1) / _chunkFrames;

        std::string bytes(_chunkFrames * _frameSize * sizeof(T), '\0');
        std::memcpy(bytes.data(), _chunk.data(), _chunk.size() * sizeof(T));

        auto chunkPath = _path / "c" / std::to_string(chunkIndex);
        for (std::size_t dim = 0; dim < _frameShape.size(); ++dim)
        {
            chunkPath /= "0";
        }

        std::filesystem::create_directories(chunkPath.parent_path());
        writeFile(chunkPath, encode(bytes, _codec));
    }

    /**
     * @brief write zarr.json with the number of frames as first dimension
     */
    template <typename T>
    void Array<T>::_writeMetadata() const
    {
        std::vector<std::size_t> shape{_nFrames};
        shape.insert(shape.end(), _frameShape.begin(), _frameShape.end());

        std::vector<std::size_t> chunkShape{_chunkFrames};
        chunkShape.insert(
            chunkShape.end(),
            _frameShape.begin(),
            _frameShape.end()
        );

        const auto metadata =
            arrayMetadata<T>(shape, chunkShape, _codec, _attributes);
        writeFile(_path / "zarr.json", metadata.str());
    }

    /**
     * @brief create the group with step, time and value arrays
     *
     * @param path
     * @param frameShape shape of one value, empty for a scalar
     * @param chunkFrames frames per chunk
     * @param codec
     * @param unit of the values
     */
    template <typename T>
    Element<T>::Element(
        const std::filesystem::path    &path,
        const std::vector<std::size_t> &frameShape,
        std::size_t                     chunkFrames,
        Codec                           codec,
        std::string_view                unit
    )
        : _step(path / "step", {}, chunkFrames, codec, {}),
          _time(path / "time", {}, chunkFrames, codec, unitAttribute("fs")),
          _value(
              path / "value",
              frameShape,
              chunkFrames,
              codec,
              unitAttribute(unit)
          )
    {
        writeGroup(path, {});
    }

    /**
     * @brief append the value of one step
     *
     * @param step
     * @param time
     * @param value
     */
    template <typename T>
    void Element<T>::append(
        std::int64_t       step,
        Time               time,
        std::span<const T> value
    )
    {
        const auto timeInFs = time.numerical_value_in(fs);

        _step.append(std::span{&step, 1});
        _time.append(std::span{&timeInFs, 1});
        _value.append(value);
    }

    /**
     * @brief write the frames of the chunks that are not full yet
     */
    template <typename T>
    void Element<T>::flush()
    {
        _step.flush();
        _time.flush();
        _value.flush();
    }

    template JsonObject arrayMetadata<std::int32_t>(
        const std::vector<std::size_t> &,
        const std::vector<std::size_t> &,
        Codec,
        const JsonObject &
    );

    template void writeArray<std::int32_t>(
        const std::filesystem::path &,
        std::span<const std::int32_t>,
        Codec
    );

    template class Array<std::int64_t>;
    template class Array<float>;
    template class Array<double>;
    template class Element<float>;
    template class Element<double>;

}   // namespace out::zarr
