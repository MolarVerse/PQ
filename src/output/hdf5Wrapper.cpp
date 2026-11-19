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

#include "hdf5Wrapper.hpp"

#include <algorithm>
#include <functional>
#include <numeric>
#include <utility>

#include "exceptions.hpp"

namespace out::hdf5
{
    /**
     * @brief take ownership of an HDF5 identifier
     *
     * @param id result of an HDF5 create or open call
     * @param close the matching HDF5 close function
     *
     * @throw InputFileException if the HDF5 call failed (negative id)
     */
    Handle::Handle(hid_t id, CloseFunction close) : _id(id), _close(close)
    {
        if (_id < 0)
        {
            throw exc::InputFileException("HDF5 call failed");
        }
    }

    /**
     * @brief close the identifier
     *
     * @details the constructor rejects negative ids, so only a moved-from
     * handle is invalid here; a failing close cannot throw from a destructor
     */
    Handle::~Handle()
    {
        if (_id != H5I_INVALID_HID)
        {
            static_cast<void>(_close(_id));
        }
    }

    Handle::Handle(Handle &&other) noexcept
        : _id(std::exchange(other._id, H5I_INVALID_HID)), _close(other._close)
    {
    }

    Handle &Handle::operator=(Handle &&other) noexcept
    {
        if (this != &other)
        {
            if (_id != H5I_INVALID_HID)
            {
                static_cast<void>(_close(_id));
            }

            _id    = std::exchange(other._id, H5I_INVALID_HID);
            _close = other._close;
        }

        return *this;
    }

    /**
     * @brief the HDF5 identifier
     *
     * @return hid_t
     */
    hid_t Handle::get() const { return _id; }

    template <>
    hid_t memoryType<int>()
    {
        return H5T_NATIVE_INT;
    }

    template <>
    hid_t memoryType<std::int64_t>()
    {
        return H5T_NATIVE_INT64;
    }

    template <>
    hid_t memoryType<float>()
    {
        return H5T_NATIVE_FLOAT;
    }

    template <>
    hid_t memoryType<double>()
    {
        return H5T_NATIVE_DOUBLE;
    }

    /**
     * @brief throw if an HDF5 call returned an error status
     *
     * @param status
     *
     * @throw InputFileException if status is negative
     */
    void check(herr_t status)
    {
        if (status < 0)
        {
            throw exc::InputFileException("HDF5 call failed");
        }
    }

    /**
     * @brief create a group
     *
     * @param parent file or group
     * @param name
     * @return Handle
     */
    Handle createGroup(hid_t parent, const std::string &name)
    {
        return {
            H5Gcreate2(
                parent,
                name.c_str(),
                H5P_DEFAULT,
                H5P_DEFAULT,
                H5P_DEFAULT
            ),
            H5Gclose
        };
    }

    /**
     * @brief variable-length UTF-8 string type, read as str by h5py
     *
     * @return Handle
     */
    Handle createStringType()
    {
        Handle type{H5Tcopy(H5T_C_S1), H5Tclose};
        check(H5Tset_size(type.get(), H5T_VARIABLE));
        check(H5Tset_cset(type.get(), H5T_CSET_UTF8));
        return type;
    }

    namespace
    {
        Handle createAttribute(
            hid_t              object,
            const std::string &name,
            hid_t              fileType,
            hsize_t            size
        )
        {
            const Handle space{H5Screate_simple(1, &size, nullptr), H5Sclose};
            return {
                H5Acreate2(
                    object,
                    name.c_str(),
                    fileType,
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Aclose
            };
        }

        Handle createDataset(
            hid_t              parent,
            const std::string &name,
            hid_t              fileType,
            hsize_t            size
        )
        {
            const Handle space{H5Screate_simple(1, &size, nullptr), H5Sclose};
            return {
                H5Dcreate2(
                    parent,
                    name.c_str(),
                    fileType,
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Dclose
            };
        }

        std::vector<const char *> cStrings(
            const std::vector<std::string> &values
        )
        {
            std::vector<const char *> pointers;
            pointers.reserve(values.size());
            for (const auto &value : values)
            {
                pointers.push_back(value.c_str());
            }
            return pointers;
        }

    }   // namespace

    /**
     * @brief write a scalar string attribute
     *
     * @param object group or dataset
     * @param name
     * @param value
     */
    void writeAttribute(
        hid_t              object,
        const std::string &name,
        const std::string &value
    )
    {
        const auto   type = createStringType();
        const Handle space{H5Screate(H5S_SCALAR), H5Sclose};
        const Handle attribute{
            H5Acreate2(
                object,
                name.c_str(),
                type.get(),
                space.get(),
                H5P_DEFAULT,
                H5P_DEFAULT
            ),
            H5Aclose
        };

        const char *text = value.c_str();
        check(H5Awrite(attribute.get(), type.get(), static_cast<void *>(&text))
        );
    }

    /**
     * @brief write an integer array attribute
     *
     * @param object group or dataset
     * @param name
     * @param values
     */
    void writeAttribute(
        hid_t                   object,
        const std::string      &name,
        const std::vector<int> &values
    )
    {
        const auto attribute =
            createAttribute(object, name, H5T_STD_I32LE, values.size());
        check(H5Awrite(attribute.get(), memoryType<int>(), values.data()));
    }

    /**
     * @brief write a string array attribute
     *
     * @param object group or dataset
     * @param name
     * @param values
     */
    void writeAttribute(
        hid_t                           object,
        const std::string              &name,
        const std::vector<std::string> &values
    )
    {
        const auto type = createStringType();
        const auto attribute =
            createAttribute(object, name, type.get(), values.size());
        const auto pointers = cStrings(values);
        check(H5Awrite(
            attribute.get(),
            type.get(),
            static_cast<const void *>(pointers.data())
        ));
    }

    /**
     * @brief write a fixed integer array dataset
     *
     * @param parent file or group
     * @param name
     * @param values
     */
    void writeDataset(
        hid_t                   parent,
        const std::string      &name,
        const std::vector<int> &values
    )
    {
        const auto dataset =
            createDataset(parent, name, H5T_STD_I32LE, values.size());
        check(H5Dwrite(
            dataset.get(),
            memoryType<int>(),
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            values.data()
        ));
    }

    /**
     * @brief write a fixed string array dataset
     *
     * @param parent file or group
     * @param name
     * @param values
     */
    void writeDataset(
        hid_t                           parent,
        const std::string              &name,
        const std::vector<std::string> &values
    )
    {
        const auto type = createStringType();
        const auto dataset =
            createDataset(parent, name, type.get(), values.size());
        const auto pointers = cStrings(values);
        check(H5Dwrite(
            dataset.get(),
            type.get(),
            H5S_ALL,
            H5S_ALL,
            H5P_DEFAULT,
            static_cast<const void *>(pointers.data())
        ));
    }

    /**
     * @brief create an empty dataset with an unlimited frame axis
     *
     * @details one chunk holds about CHUNK_BYTES; a partly written chunk
     * stays in the cache until it is full
     */
    template <typename T>
    Handle Series<T>::_create(
        hid_t              parent,
        const std::string &name,
        hid_t              fileType,
        const Shape       &frameShape
    )
    {
        Shape dims{0};
        Shape maxDims{H5S_UNLIMITED};
        dims.insert(dims.end(), frameShape.begin(), frameShape.end());
        maxDims.insert(maxDims.end(), frameShape.begin(), frameShape.end());

        std::size_t frameBytes = H5Tget_size(fileType);
        for (const auto extent : frameShape)
        {
            frameBytes *= extent;
        }

        const auto framesPerChunk = std::clamp<std::size_t>(
            CHUNK_BYTES / frameBytes,
            1,
            frameShape.empty() ? CHUNK_BYTES : MAX_CHUNK
        );

        auto chunk = dims;
        chunk[0]   = framesPerChunk;

        const auto   rank = static_cast<int>(dims.size());
        const Handle space{
            H5Screate_simple(rank, dims.data(), maxDims.data()),
            H5Sclose
        };

        const Handle createList{H5Pcreate(H5P_DATASET_CREATE), H5Pclose};
        check(H5Pset_chunk(createList.get(), rank, chunk.data()));
        check(H5Pset_shuffle(createList.get()));
        check(H5Pset_deflate(createList.get(), DEFLATE_LEVEL));

        const Handle accessList{H5Pcreate(H5P_DATASET_ACCESS), H5Pclose};
        check(H5Pset_chunk_cache(
            accessList.get(),
            CACHE_SLOTS,
            2 * framesPerChunk * frameBytes,
            1.0
        ));

        return {
            H5Dcreate2(
                parent,
                name.c_str(),
                fileType,
                space.get(),
                H5P_DEFAULT,
                createList.get(),
                accessList.get()
            ),
            H5Dclose
        };
    }

    /**
     * @brief create the dataset
     *
     * @param parent file or group
     * @param name
     * @param fileType HDF5 type stored in the file
     * @param frameShape shape of one frame, empty for a scalar
     */
    template <typename T>
    Series<T>::Series(
        hid_t              parent,
        const std::string &name,
        hid_t              fileType,
        Shape              frameShape
    )
        : _dataset(_create(parent, name, fileType, frameShape)),
          _frameShape(std::move(frameShape)),
          _frameSize(
              std::accumulate(
                  _frameShape.begin(),
                  _frameShape.end(),
                  std::size_t{1},
                  std::multiplies<>()
              )
          )
    {
    }

    /**
     * @brief append one frame
     *
     * @param frame values of one frame in row-major order
     *
     * @throw InputFileException if frame does not match the frame shape
     */
    template <typename T>
    void Series<T>::append(std::span<const T> frame)
    {
        if (frame.size() != _frameSize)
        {
            throw exc::InputFileException(
                "HDF5 frame has " + std::to_string(frame.size()) +
                " values, expected " + std::to_string(_frameSize)
            );
        }

        Shape dims{_nFrames + 1};
        dims.insert(dims.end(), _frameShape.begin(), _frameShape.end());
        check(H5Dset_extent(_dataset.get(), dims.data()));

        Shape start(dims.size(), 0);
        start[0]   = _nFrames;
        auto count = dims;
        count[0]   = 1;

        const Handle fileSpace{H5Dget_space(_dataset.get()), H5Sclose};
        check(H5Sselect_hyperslab(
            fileSpace.get(),
            H5S_SELECT_SET,
            start.data(),
            nullptr,
            count.data(),
            nullptr
        ));

        const Handle memorySpace{
            H5Screate_simple(
                static_cast<int>(count.size()),
                count.data(),
                nullptr
            ),
            H5Sclose
        };

        check(H5Dwrite(
            _dataset.get(),
            memoryType<T>(),
            memorySpace.get(),
            fileSpace.get(),
            H5P_DEFAULT,
            frame.data()
        ));
        ++_nFrames;
    }

    /**
     * @brief the dataset identifier, e.g. for attributes
     *
     * @return hid_t
     */
    template <typename T>
    hid_t Series<T>::get() const
    {
        return _dataset.get();
    }

    /**
     * @brief number of frames written so far
     *
     * @return hsize_t
     */
    template <typename T>
    hsize_t Series<T>::getNumberOfFrames() const
    {
        return _nFrames;
    }

    /**
     * @brief create the element group with its step, time and value series
     *
     * @param parent file or group
     * @param name
     * @param fileType HDF5 type of the values in the file
     * @param frameShape shape of one value, empty for a scalar
     * @param unit unit attribute of the values
     */
    template <typename T>
    Element<T>::Element(
        hid_t              parent,
        const std::string &name,
        hid_t              fileType,
        Shape              frameShape,
        const std::string &unit
    )
        : _group(createGroup(parent, name)),
          _step(_group.get(), "step", H5T_STD_I64LE, {}),
          _time(_group.get(), "time", H5T_IEEE_F64LE, {}),
          _value(_group.get(), "value", fileType, std::move(frameShape))
    {
        writeAttribute(_time.get(), "unit", std::string{"fs"});
        writeAttribute(_value.get(), "unit", unit);
    }

    /**
     * @brief append the value of one step
     *
     * @param step
     * @param time in fs
     * @param value
     */
    template <typename T>
    void Element<T>::append(
        std::int64_t       step,
        double             time,
        std::span<const T> value
    )
    {
        _step.append(std::span{&step, 1});
        _time.append(std::span{&time, 1});
        _value.append(value);
    }

    template class Series<std::int64_t>;
    template class Series<float>;
    template class Series<double>;
    template class Element<float>;
    template class Element<double>;

}   // namespace out::hdf5
