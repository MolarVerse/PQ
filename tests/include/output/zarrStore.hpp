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

#ifndef TESTS__INCLUDE__OUTPUT__ZARR_STORE_HPP
#define TESTS__INCLUDE__OUTPUT__ZARR_STORE_HPP

#include <gtest/gtest.h>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "zarrWriter.hpp"

#ifdef WITH_ZSTD
#include <zstd.h>
#endif

namespace zarrTest
{
    /**
     * @brief whole content of a file
     */
    inline std::string readFile(const std::filesystem::path &path)
    {
        std::ifstream      file(path, std::ios::binary);
        std::ostringstream content;
        content << file.rdbuf();
        return content.str();
    }

    /**
     * @brief values of a chunk file written with codec
     */
    template <typename T>
    std::vector<T> readChunk(
        const std::filesystem::path &path,
        out::zarr::Codec             codec
    )
    {
        auto bytes = readFile(path);

        if (codec == out::zarr::Codec::ZSTD)
        {
#ifdef WITH_ZSTD
            std::string raw(
                ZSTD_getFrameContentSize(bytes.data(), bytes.size()),
                '\0'
            );
            raw.resize(ZSTD_decompress(
                raw.data(),
                raw.size(),
                bytes.data(),
                bytes.size()
            ));
            bytes = raw;
#else
            ADD_FAILURE() << "zstd chunk in a build without zstd";
#endif
        }

        std::vector<T> values(bytes.size() / sizeof(T));
        std::memcpy(values.data(), bytes.data(), values.size() * sizeof(T));
        return values;
    }

    /**
     * @brief the codec ZarrOutput uses in this build
     */
    inline out::zarr::Codec outputCodec()
    {
        return out::zarr::zstdAvailable() ? out::zarr::Codec::ZSTD
                                          : out::zarr::Codec::BYTES;
    }
}   // namespace zarrTest

#endif   // TESTS__INCLUDE__OUTPUT__ZARR_STORE_HPP
