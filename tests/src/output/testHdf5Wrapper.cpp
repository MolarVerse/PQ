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

// only built with BUILD_WITH_HDF5; empty for tools without the HDF5 headers
#if __has_include(<hdf5.h>)

#include <gtest/gtest.h>
#include <hdf5.h>

#include <array>
#include <filesystem>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "exceptions.hpp"
#include "hdf5Wrapper.hpp"
#include "throwWithMessage.hpp"

namespace
{
    const std::string fileName = "testHdf5Wrapper.h5";

    out::hdf5::Handle createFile()
    {
        return {
            H5Fcreate(
                fileName.c_str(),
                H5F_ACC_TRUNC,
                H5P_DEFAULT,
                H5P_DEFAULT
            ),
            H5Fclose
        };
    }

    std::vector<double> readDoubles(hid_t dataset)
    {
        const out::hdf5::Handle space{H5Dget_space(dataset), H5Sclose};
        const auto size = H5Sget_simple_extent_npoints(space.get());

        std::vector<double> values(static_cast<std::size_t>(size));
        out::hdf5::check(H5Dread(
            dataset,
            H5T_NATIVE_DOUBLE,
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()
        ));
        return values;
    }

}   // namespace

TEST(TestHdf5Wrapper, handleRejectsFailedCall)
{
    EXPECT_THROW_MSG(
        out::hdf5::Handle(H5I_INVALID_HID, H5Gclose),
        exc::InputFileException,
        "HDF5 call failed"
    );
    EXPECT_THROW_MSG(
        out::hdf5::check(-1),
        exc::InputFileException,
        "HDF5 call failed"
    );
}

TEST(TestHdf5Wrapper, handleMovesOwnership)
{
    {
        auto       file    = createFile();
        auto       group   = out::hdf5::createGroup(file.get(), "first");
        const auto firstId = group.get();

        out::hdf5::Handle moved{std::move(group)};
        EXPECT_EQ(moved.get(), firstId);
        EXPECT_EQ(
            group.get(),
            H5I_INVALID_HID
        );   // NOLINT(bugprone-use-after-move)

        auto other = out::hdf5::createGroup(file.get(), "second");
        moved      = std::move(other);
        EXPECT_GT(H5Iis_valid(moved.get()), 0);
        EXPECT_EQ(H5Iis_valid(firstId), 0);
    }
    std::filesystem::remove(fileName);
}

TEST(TestHdf5Wrapper, elementWritesStepTimeAndValue)
{
    {
        auto file = createFile();

        out::hdf5::Element<double> element{
            file.get(),
            "energy",
            H5T_IEEE_F64LE,
            out::hdf5::Shape{2},
            "kcal mol-1"
        };
        const std::array<double, 2> first{1.0, 2.0};
        const std::array<double, 2> second{3.0, 4.0};
        element.append(10, 5.0, std::span<const double>{first});
        element.append(20, 10.0, std::span<const double>{second});

        const out::hdf5::Handle value{
            H5Dopen2(file.get(), "energy/value", H5P_DEFAULT),
            H5Dclose
        };
        const out::hdf5::Handle time{
            H5Dopen2(file.get(), "energy/time", H5P_DEFAULT),
            H5Dclose
        };
        EXPECT_EQ(readDoubles(value.get()), (std::vector{1.0, 2.0, 3.0, 4.0}));
        EXPECT_EQ(readDoubles(time.get()), (std::vector{5.0, 10.0}));
        EXPECT_GT(H5Aexists(value.get(), "unit"), 0);
    }
    std::filesystem::remove(fileName);
}

TEST(TestHdf5Wrapper, seriesRejectsWrongFrameSize)
{
    {
        auto file = createFile();

        out::hdf5::Series<float> series{
            file.get(),
            "positions",
            H5T_IEEE_F32LE,
            out::hdf5::Shape{2, 3}
        };
        const std::vector<float> frame(6, 1.0F);
        series.append(std::span<const float>{frame});
        EXPECT_EQ(series.getNumberOfFrames(), 1);

        EXPECT_THROW_MSG(
            series.append(std::span<const float>{frame.data(), 5}),
            exc::InputFileException,
            "HDF5 frame has 5 values, expected 6"
        );
        EXPECT_EQ(series.getNumberOfFrames(), 1);
    }
    std::filesystem::remove(fileName);
}

TEST(TestHdf5Wrapper, writesAttributesAndDatasets)
{
    {
        auto file  = createFile();
        auto group = out::hdf5::createGroup(file.get(), "group");

        out::hdf5::writeAttribute(group.get(), "name", std::string{"PQ"});
        out::hdf5::writeAttribute(group.get(), "version", std::vector{1, 1});
        out::hdf5::writeAttribute(
            group.get(),
            "boundary",
            std::vector<std::string>{"periodic", "none"}
        );
        out::hdf5::writeDataset(group.get(), "species", std::vector{8, 1, 1});
        out::hdf5::writeDataset(
            group.get(),
            "names",
            std::vector<std::string>{"O", "H", "H"}
        );

        for (const auto *name : {"name", "version", "boundary"})
        {
            EXPECT_GT(H5Aexists(group.get(), name), 0) << name;
        }

        for (const auto *name : {"species", "names"})
        {
            EXPECT_GT(H5Lexists(group.get(), name, H5P_DEFAULT), 0) << name;
        }
    }
    std::filesystem::remove(fileName);
}

#endif   // __has_include(<hdf5.h>)
