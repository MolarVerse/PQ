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

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "throwWithMessage.hpp"
#include "units.hpp"
#include "zarrStore.hpp"
#include "zarrWriter.hpp"

namespace
{
    const std::string BYTES_CODEC =
        R"([{"name": "bytes", "configuration": {"endian": "little"}}])";
    const std::string ZSTD_CODEC =
        R"([{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd", "configuration": {"level": 3, "checksum": false}}])";
    const std::string NO_ZSTD =
        "Zarr zstd codec needs PQ built with -DBUILD_WITH_ZSTD=ON";

    std::vector<out::zarr::Codec> availableCodecs()
    {
        if (out::zarr::zstdAvailable())
        {
            return {out::zarr::Codec::BYTES, out::zarr::Codec::ZSTD};
        }

        return {out::zarr::Codec::BYTES};
    }

    // zarr.json of the test array after nFrames frames
    std::string arrayMetadata(std::size_t nFrames, out::zarr::Codec codec)
    {
        return out::zarr::arrayMetadata<double>(
                   {nFrames, 3},
                   {2, 3},
                   codec,
                   out::zarr::unitAttribute("Angstrom")
        )
            .str();
    }

    std::string codecJson(out::zarr::Codec codec)
    {
        return codec == out::zarr::Codec::ZSTD ? ZSTD_CODEC : BYTES_CODEC;
    }
}   // namespace

/**
 * @brief every member type in the order it was added
 *
 */
TEST(TestZarrWriter, jsonObjectWritesMembersInOrder)
{
    out::zarr::JsonObject inner;
    inner.addString("name", R"(a"b\c)");

    out::zarr::JsonObject object;
    object.addInteger("count", 3);
    object.addIntegers("shape", {1, 2});
    object.addIntegers("empty", {});
    object.addStrings("names", {"O", "H"});
    object.addBool("yes", true);
    object.addBool("no", false);
    object.addObject("inner", inner);
    object.addObjects("list", {inner, out::zarr::JsonObject{}});

    EXPECT_EQ(
        object.str(),
        R"({"count": 3, "shape": [1, 2], "empty": [], "names": ["O", "H"], "yes": true, "no": false, "inner": {"name": "a\"b\\c"}, "list": [{"name": "a\"b\\c"}, {}]})"
    );
    EXPECT_EQ(out::zarr::JsonObject{}.str(), "{}");
    EXPECT_EQ(out::zarr::unitAttribute("fs").str(), R"({"unit": "fs"})");
}

/**
 * @brief group metadata and codecs as the Zarr v3 spec names them
 *
 */
TEST(TestZarrWriter, writesGroupAndArrayMetadata)
{
    const std::filesystem::path path = "testZarrWriterGroup.zarr";
    std::filesystem::remove_all(path);

    out::zarr::writeGroup(path / "sub", out::zarr::unitAttribute("e"));
    EXPECT_EQ(
        zarrTest::readFile(path / "sub" / "zarr.json"),
        R"({"zarr_format": 3, "node_type": "group", "attributes": {"unit": "e"}})"
    );
    EXPECT_FALSE(std::filesystem::exists(path / "sub" / "zarr.json.partial"));

    EXPECT_EQ(
        out::zarr::arrayMetadata<std::int32_t>(
            {4},
            {2},
            out::zarr::Codec::BYTES,
            {}
        )
            .str(),
        R"({"zarr_format": 3, "node_type": "array", "shape": [4], "data_type": "int32", "chunk_grid": {"name": "regular", "configuration": {"chunk_shape": [2]}}, "chunk_key_encoding": {"name": "default", "configuration": {"separator": "/"}}, "fill_value": 0, "codecs": [{"name": "bytes", "configuration": {"endian": "little"}}], "attributes": {}})"
    );

    EXPECT_EQ(out::zarr::dataType<std::int64_t>(), "int64");
    EXPECT_EQ(out::zarr::dataType<float>(), "float32");
    EXPECT_EQ(out::zarr::dataType<double>(), "float64");

    std::filesystem::remove_all(path);
}

/**
 * @brief zstd needs PQ built with it; otherwise it is refused, not ignored
 *
 */
TEST(TestZarrWriter, zstdCodecOnlyWithZstd)
{
    if (out::zarr::zstdAvailable())
    {
        out::zarr::JsonObject codecs;
        codecs.addObjects("codecs", out::zarr::codecs(out::zarr::Codec::ZSTD));
        EXPECT_EQ(
            codecs.str(),
            R"({"codecs": [{"name": "bytes", "configuration": {"endian": "little"}}, {"name": "zstd", "configuration": {"level": 3, "checksum": false}}]})"
        );

        const std::string bytes(1000, 'a');
        EXPECT_LT(
            out::zarr::encode(bytes, out::zarr::Codec::ZSTD).size(),
            bytes.size()
        );
        return;
    }

    EXPECT_THROW_MSG(
        static_cast<void>(out::zarr::codecs(out::zarr::Codec::ZSTD)),
        exc::CompileTimeException,
        NO_ZSTD
    );
    EXPECT_THROW_MSG(
        static_cast<void>(out::zarr::encode("a", out::zarr::Codec::ZSTD)),
        exc::CompileTimeException,
        NO_ZSTD
    );
    EXPECT_THROW_MSG(
        out::zarr::Array<float>(
            "testZarrWriterNoZstd.zarr",
            {},
            2,
            out::zarr::Codec::ZSTD,
            {}
        ),
        exc::CompileTimeException,
        NO_ZSTD
    );
    std::filesystem::remove_all("testZarrWriterNoZstd.zarr");
}

/**
 * @brief a fixed array is one chunk
 *
 */
TEST(TestZarrWriter, writesFixedArray)
{
    for (const auto codec : availableCodecs())
    {
        const std::filesystem::path path = "testZarrWriterFixed.zarr";
        std::filesystem::remove_all(path);

        const std::vector<std::int32_t> values{8, 1, 0};
        out::zarr::writeArray(
            path,
            std::span<const std::int32_t>{values},
            codec
        );

        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            out::zarr::arrayMetadata<std::int32_t>({3}, {3}, codec, {}).str()
        );
        EXPECT_EQ(
            zarrTest::readChunk<std::int32_t>(path / "c" / "0", codec),
            values
        );

        std::filesystem::remove_all(path);
    }
}

/**
 * @brief frames are appended in chunks; the shape grows with each full
 * chunk and with flush
 *
 */
TEST(TestZarrWriter, appendsFramesInChunks)
{
    for (const auto codec : availableCodecs())
    {
        const std::filesystem::path path = "testZarrWriterArray.zarr";
        std::filesystem::remove_all(path);

        out::zarr::Array<double>
            array{path, {3}, 2, codec, out::zarr::unitAttribute("Angstrom")};
        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            arrayMetadata(0, codec)
        );
        EXPECT_NE(
            arrayMetadata(0, codec).find(codecJson(codec)),
            std::string::npos
        );

        const std::vector<double> first{1.0, 2.0, 3.0};
        const std::vector<double> second{4.0, 5.0, 6.0};
        const std::vector<double> third{7.0, 8.0, 9.0};

        array.append(first);
        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            arrayMetadata(0, codec)
        );
        EXPECT_FALSE(std::filesystem::exists(path / "c"));

        array.append(second);
        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            arrayMetadata(2, codec)
        );
        EXPECT_EQ(
            zarrTest::readChunk<double>(path / "c" / "0" / "0", codec),
            (std::vector<double>{1.0, 2.0, 3.0, 4.0, 5.0, 6.0})
        );

        array.flush();
        EXPECT_FALSE(std::filesystem::exists(path / "c" / "1"));

        array.append(third);
        array.flush();
        EXPECT_EQ(array.getNumberOfFrames(), 3U);
        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            arrayMetadata(3, codec)
        );
        EXPECT_EQ(
            zarrTest::readChunk<double>(path / "c" / "1" / "0", codec),
            (std::vector<double>{7.0, 8.0, 9.0, 0.0, 0.0, 0.0})
        );

        EXPECT_THROW_MSG(
            array.append(std::vector<double>{1.0}),
            exc::InputFileException,
            "Zarr frame has 1 values, expected 3"
        );

        std::filesystem::remove_all(path);
    }
}

/**
 * @brief an element is a group of step, time in fs and value
 *
 */
TEST(TestZarrWriter, elementWritesStepTimeAndValue)
{
    for (const auto codec : availableCodecs())
    {
        const std::filesystem::path path = "testZarrWriterElement.zarr";
        std::filesystem::remove_all(path);

        {
            out::zarr::Element<float> element{path, {}, 4, codec, "e"};
            element.append(10, 5.0 * fs, std::vector<float>{0.5F});
            element.append(20, 10.0 * fs, std::vector<float>{-0.5F});
            element.flush();
        }

        EXPECT_EQ(
            zarrTest::readFile(path / "zarr.json"),
            R"({"zarr_format": 3, "node_type": "group", "attributes": {}})"
        );
        EXPECT_NE(
            zarrTest::readFile(path / "time" / "zarr.json")
                .find(R"("attributes": {"unit": "fs"})"),
            std::string::npos
        );
        EXPECT_NE(
            zarrTest::readFile(path / "value" / "zarr.json")
                .find(R"("shape": [2], "data_type": "float32")"),
            std::string::npos
        );
        EXPECT_EQ(
            zarrTest::readChunk<std::int64_t>(path / "step" / "c" / "0", codec),
            (std::vector<std::int64_t>{10, 20, 0, 0})
        );
        EXPECT_EQ(
            zarrTest::readChunk<double>(path / "time" / "c" / "0", codec),
            (std::vector<double>{5.0, 10.0, 0.0, 0.0})
        );
        EXPECT_EQ(
            zarrTest::readChunk<float>(path / "value" / "c" / "0", codec),
            (std::vector<float>{0.5F, -0.5F, 0.0F, 0.0F})
        );

        std::filesystem::remove_all(path);
    }
}

/**
 * @brief a file that cannot be written is reported
 *
 */
TEST(TestZarrWriter, throwsIfFileCannotBeWritten)
{
    const std::filesystem::path path = "testZarrWriterMissing/zarr.json";
    std::filesystem::remove_all("testZarrWriterMissing");

    EXPECT_THROW_MSG(
        out::zarr::writeFile(path, "{}"),
        exc::InputFileException,
        "Could not write Zarr file - filename = " + path.string()
    );
}
