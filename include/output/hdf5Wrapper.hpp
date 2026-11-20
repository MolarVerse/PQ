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

#ifndef OUTPUT__HDF5_WRAPPER_HPP
#define OUTPUT__HDF5_WRAPPER_HPP

#include <hdf5.h>

#include <cstddef>
#include <cstdint>
#include <mstd/utils.hpp>
#include <span>
#include <string>
#include <vector>

/**
 * @brief RAII and typed C++ wrappers around the HDF5 C API
 *
 * @details only built with BUILD_WITH_HDF5; used by the H5MD output
 */
namespace out::hdf5
{
    using CloseFunction = herr_t (*)(hid_t);
    using Shape         = std::vector<hsize_t>;

    inline constexpr std::size_t CHUNK_BYTES   = std::size_t{1} << 16U;
    inline constexpr std::size_t MAX_CHUNK     = 128;
    inline constexpr std::size_t CACHE_SLOTS   = 521;
    inline constexpr int         DEFLATE_LEVEL = 1;

    /**
     * @brief owns an HDF5 identifier and closes it
     */
    class Handle
    {
       private:
        hid_t         _id;
        CloseFunction _close;

       public:
        Handle(hid_t id, CloseFunction close);
        ~Handle();

        MSTD_NO_COPY(Handle)
        Handle(Handle &&other) noexcept;
        Handle &operator=(Handle &&other) noexcept;

        [[nodiscard]] hid_t get() const;
    };

    /**
     * @brief HDF5 memory type of a C++ value type
     */
    template <typename T>
    [[nodiscard]] hid_t memoryType();

    void check(herr_t status);

    [[nodiscard]] Handle createGroup(hid_t parent, const std::string &name);
    [[nodiscard]] Handle createStringType();

    void writeAttribute(
        hid_t              object,
        const std::string &name,
        const std::string &value
    );
    void writeAttribute(
        hid_t                   object,
        const std::string      &name,
        const std::vector<int> &values
    );
    void writeAttribute(
        hid_t                           object,
        const std::string              &name,
        const std::vector<std::string> &values
    );

    void writeDataset(
        hid_t                   parent,
        const std::string      &name,
        const std::vector<int> &values
    );
    void writeDataset(
        hid_t                           parent,
        const std::string              &name,
        const std::vector<std::string> &values
    );

    /**
     * @brief dataset that grows by one frame per append, chunked and
     * compressed along the frames
     */
    template <typename T>
    class Series
    {
       private:
        Handle      _dataset;
        Shape       _frameShape;
        std::size_t _frameSize;
        hsize_t     _nFrames = 0;

       public:
        Series(
            hid_t              parent,
            const std::string &name,
            hid_t              fileType,
            Shape              frameShape
        );

        void append(std::span<const T> frame);

        [[nodiscard]] hid_t   get() const;
        [[nodiscard]] hsize_t getNumberOfFrames() const;

       private:
        [[nodiscard]] static Handle _create(
            hid_t              parent,
            const std::string &name,
            hid_t              fileType,
            const Shape       &frameShape
        );
    };

    /**
     * @brief H5MD time-dependent element: step, time in fs and value
     */
    template <typename T>
    class Element
    {
       private:
        Handle               _group;
        Series<std::int64_t> _step;
        Series<double>       _time;
        Series<T>            _value;

       public:
        Element(
            hid_t              parent,
            const std::string &name,
            hid_t              fileType,
            Shape              frameShape,
            const std::string &unit
        );

        void append(std::int64_t step, double time, std::span<const T> value);
    };

}   // namespace out::hdf5

#endif   // OUTPUT__HDF5_WRAPPER_HPP
