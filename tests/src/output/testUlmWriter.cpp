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

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "ulmWriter.hpp"

namespace
{
    std::int64_t readInt(const std::string &data, std::int64_t position)
    {
        std::int64_t value = 0;
        const auto   bytes =
            data.substr(static_cast<std::size_t>(position), sizeof(value));
        std::memcpy(&value, bytes.data(), sizeof(value));
        return value;
    }

    double readDouble(const std::string &data, std::int64_t position)
    {
        double     value = 0.0;
        const auto bytes =
            data.substr(static_cast<std::size_t>(position), sizeof(value));
        std::memcpy(&value, bytes.data(), sizeof(value));
        return value;
    }

    // JSON text of item index
    std::string itemJson(const std::string &data, std::int64_t index)
    {
        const auto offsets  = readInt(data, out::ulm::OFFSETS_POINTER_POSITION);
        const auto position = readInt(data, offsets + (8 * index));
        const auto length   = readInt(data, position);

        return data.substr(
            static_cast<std::size_t>(position) + sizeof(std::int64_t),
            static_cast<std::size_t>(length)
        );
    }
}   // namespace

/**
 * @brief header, aligned arrays and items read back from the bytes
 *
 */
TEST(TestUlmWriter, writesHeaderArraysAndItems)
{
    std::stringstream stream;
    out::ulm::Writer  writer;

    writer.writeHeader(stream);
    auto data = stream.str();
    EXPECT_EQ(data.size(), static_cast<std::size_t>(out::ulm::HEADER_SIZE + 8));
    EXPECT_EQ(data.substr(0, 24), out::ulm::FILE_TAG);
    EXPECT_EQ(readInt(data, 24), out::ulm::FORMAT_VERSION);
    EXPECT_EQ(readInt(data, out::ulm::ITEM_COUNT_POSITION), 0);
    EXPECT_EQ(
        readInt(data, out::ulm::OFFSETS_POINTER_POSITION),
        out::ulm::HEADER_SIZE
    );
    EXPECT_EQ(readInt(data, out::ulm::HEADER_SIZE), -1);

    const auto numbers = std::vector<std::int64_t>{8, 1};
    const auto values  = std::vector<double>{1.5, -2.25, 3.0};

    const auto intArray    = writer.writeArray(stream, numbers, {2});
    const auto doubleArray = writer.writeArray(stream, values, {1, 3});
    stream.write("x", 1);
    const auto aligned = writer.writeArray(stream, values, {3});

    EXPECT_EQ(intArray.offset, 56);
    EXPECT_EQ(intArray.dtype, "int64");
    EXPECT_EQ(doubleArray.offset, 72);
    EXPECT_EQ(doubleArray.dtype, "float64");
    EXPECT_EQ(doubleArray.shape, (std::vector<std::size_t>{1, 3}));
    EXPECT_EQ(aligned.offset, 104);

    writer.writeItem(stream, R"({"a": 1})");
    writer.writeItem(stream, R"({"b": 2})");
    EXPECT_EQ(writer.getNItems(), 2);

    data = stream.str();
    EXPECT_EQ(readInt(data, intArray.offset + 8), 1);
    EXPECT_EQ(readDouble(data, doubleArray.offset + 8), -2.25);
    EXPECT_EQ(data.substr(97, 7), "#######");
    EXPECT_EQ(readDouble(data, aligned.offset + 16), 3.0);

    EXPECT_EQ(readInt(data, out::ulm::ITEM_COUNT_POSITION), 2);
    EXPECT_EQ(itemJson(data, 0), R"({"a": 1})");
    EXPECT_EQ(itemJson(data, 1), R"({"b": 2})");
    EXPECT_EQ(readInt(data, out::ulm::HEADER_SIZE), 128);
}

/**
 * @brief a full offsets table moves to the end of the file, 42 times larger
 *
 */
TEST(TestUlmWriter, growsTheOffsetsTable)
{
    std::stringstream stream;
    out::ulm::Writer  writer;
    writer.writeHeader(stream);

    constexpr std::int64_t nItems = 44;
    for (std::int64_t index = 0; index < nItems; ++index)
    {
        writer.writeItem(stream, "{}");
    }

    const auto data    = stream.str();
    const auto offsets = readInt(data, out::ulm::OFFSETS_POINTER_POSITION);
    EXPECT_GT(offsets, out::ulm::HEADER_SIZE);
    EXPECT_EQ(offsets % out::ulm::ALIGNMENT, 0);
    EXPECT_EQ(readInt(data, out::ulm::ITEM_COUNT_POSITION), nItems);

    std::int64_t previous = 0;
    for (std::int64_t index = 0; index < nItems; ++index)
    {
        const auto position = readInt(data, offsets + (8 * index));
        EXPECT_GT(position, previous);
        EXPECT_EQ(itemJson(data, index), "{}");
        previous = position;
    }

    EXPECT_EQ(readInt(data, offsets + (8 * nItems)), 0);
    EXPECT_EQ(static_cast<std::int64_t>(data.size()), previous + 8 + 2);
}

/**
 * @brief JSON as Python's json.dumps writes it, numbers read back exactly
 *
 */
TEST(TestUlmWriter, buildsJson)
{
    out::ulm::JsonBuilder json;
    const auto            flags = std::array{true, false};
    const auto matrix = std::array{1.0, 0.0, 0.0, 0.0, 2.5, 0.0, 0.0, 0.0, 0.1};
    const auto array =
        out::ulm::Array{.shape = {3, 3}, .dtype = "float64", .offset = 80};

    json.begin();
    json.addInteger("version", 1);
    json.addString("name", R"(p"q\)");
    json.addBools("pbc", flags);
    json.addMatrix("cell", matrix);
    json.addArray("positions", array);
    json.beginChild("calculator");
    json.beginObject("parameters");
    json.endObject();
    json.addNumber("energy", -1.5);
    json.endObject();
    json.beginObject("info");
    json.addNumber("nan", std::numeric_limits<double>::quiet_NaN());
    json.addNumber("inf", std::numeric_limits<double>::infinity());
    json.addNumber("minf", -std::numeric_limits<double>::infinity());
    json.endObject();
    json.end();

    EXPECT_EQ(
        json.view(),
        R"({"version": 1, "name": "p\"q\\", "pbc": [true, false], )"
        R"("cell": [[1e+00, 0e+00, 0e+00], [0e+00, 2.5e+00, 0e+00], [0e+00, 0e+00, 1e-01]], )"
        R"("positions.": {"ndarray": [[3, 3], "float64", 80]}, )"
        R"("calculator.": {"parameters": {}, "energy": -1.5e+00}, )"
        R"("info": {"nan": NaN, "inf": Infinity, "minf": -Infinity}})"
    );

    json.begin();
    json.end();
    EXPECT_EQ(json.view(), "{}");
}
