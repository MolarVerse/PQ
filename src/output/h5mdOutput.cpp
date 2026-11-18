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

#include "h5mdOutput.hpp"

#include <cstddef>
#include <memory>
#include <string>

#include "frame.hpp"

#ifdef WITH_HDF5

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <filesystem>
#include <vector>

#include "atomNumberMap.hpp"
#include "box.hpp"
#include "exceptions.hpp"
#include "stringUtilities.hpp"
#include "systemInfo.hpp"

namespace out
{
    namespace
    {
        constexpr std::size_t CHUNK_BYTES   = std::size_t{1} << 16U;
        constexpr std::size_t MAX_CHUNK     = 128;
        constexpr std::size_t CACHE_SLOTS   = 521;
        constexpr int         DEFLATE_LEVEL = 1;

        // H5MD readers know Angstrom fs-1, not the Angstrom s-1 of PQ
        constexpr double PER_S_TO_PER_FS = 1.0e-15;

        // species of atoms without an element (dummy and virtual atoms)
        constexpr int NO_ELEMENT  = 0;
        constexpr int MAX_ELEMENT = 118;

        /**
         * @brief owns an HDF5 identifier and closes it
         */
        class Handle
        {
           private:
            hid_t _id;
            herr_t (*_close)(hid_t);

           public:
            Handle(const hid_t id, herr_t (*close)(hid_t))
                : _id(id), _close(close)
            {
                if (_id < 0)
                    throw exc::InputFileException("HDF5 call failed");
            }

            ~Handle()
            {
                if (_id >= 0)
                    static_cast<void>(_close(_id));
            }

            Handle(Handle &&other) noexcept
                : _id(other._id), _close(other._close)
            {
                other._id = -1;
            }

            Handle(const Handle &)            = delete;
            Handle &operator=(const Handle &) = delete;
            Handle &operator=(Handle &&)      = delete;

            [[nodiscard]] hid_t get() const { return _id; }
        };

        void check(const herr_t status)
        {
            if (status < 0)
                throw exc::InputFileException("HDF5 call failed");
        }

        Handle group(const hid_t parent, const char *name)
        {
            return {
                H5Gcreate2(parent, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT),
                H5Gclose
            };
        }

        /**
         * @brief variable-length UTF-8 string type, read as str by h5py
         */
        Handle stringType()
        {
            Handle type{H5Tcopy(H5T_C_S1), H5Tclose};
            check(H5Tset_size(type.get(), H5T_VARIABLE));
            check(H5Tset_cset(type.get(), H5T_CSET_UTF8));
            return type;
        }

        /**
         * @brief write a string attribute
         */
        void attribute(
            const hid_t        object,
            const char        *name,
            const std::string &value
        )
        {
            const auto   type = stringType();
            const Handle space{H5Screate(H5S_SCALAR), H5Sclose};
            const Handle attr{
                H5Acreate2(
                    object,
                    name,
                    type.get(),
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Aclose
            };

            const char *data = value.c_str();
            check(H5Awrite(attr.get(), type.get(), static_cast<void *>(&data)));
        }

        /**
         * @brief write an integer array attribute
         */
        void attribute(
            const hid_t             object,
            const char             *name,
            const std::vector<int> &values
        )
        {
            const auto   size = static_cast<hsize_t>(values.size());
            const Handle space{H5Screate_simple(1, &size, nullptr), H5Sclose};
            const Handle attr{
                H5Acreate2(
                    object,
                    name,
                    H5T_STD_I32LE,
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Aclose
            };

            check(H5Awrite(attr.get(), H5T_NATIVE_INT, values.data()));
        }

        /**
         * @brief write a string array, as attribute or dataset
         */
        void strings(
            const hid_t                     object,
            const char                     *name,
            const std::vector<std::string> &values,
            const bool                      asAttribute
        )
        {
            std::vector<const char *> data;
            data.reserve(values.size());
            for (const auto &value : values) data.push_back(value.c_str());

            const auto   type = stringType();
            const auto   size = static_cast<hsize_t>(values.size());
            const Handle space{H5Screate_simple(1, &size, nullptr), H5Sclose};

            if (asAttribute)
            {
                const Handle attr{
                    H5Acreate2(
                        object,
                        name,
                        type.get(),
                        space.get(),
                        H5P_DEFAULT,
                        H5P_DEFAULT
                    ),
                    H5Aclose
                };
                check(H5Awrite(attr.get(), type.get(), data.data()));
                return;
            }

            const Handle set{
                H5Dcreate2(
                    object,
                    name,
                    type.get(),
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Dclose
            };
            check(H5Dwrite(
                set.get(),
                type.get(),
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                data.data()
            ));
        }

        /**
         * @brief dataset that grows by one frame per write, chunked and
         * compressed along the frames
         */
        class Series
        {
           private:
            Handle               _set;
            hid_t                _memType;
            std::vector<hsize_t> _frameShape;
            hsize_t              _frames = 0;

            static Handle create(
                const hid_t                 parent,
                const char                 *name,
                const hid_t                 fileType,
                const std::vector<hsize_t> &frameShape
            )
            {
                std::vector<hsize_t> dims{0};
                std::vector<hsize_t> maxDims{H5S_UNLIMITED};
                dims.insert(dims.end(), frameShape.begin(), frameShape.end());
                maxDims.insert(
                    maxDims.end(),
                    frameShape.begin(),
                    frameShape.end()
                );

                std::size_t frameBytes = H5Tget_size(fileType);
                for (const auto extent : frameShape) frameBytes *= extent;

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

                const Handle create{H5Pcreate(H5P_DATASET_CREATE), H5Pclose};
                check(H5Pset_chunk(create.get(), rank, chunk.data()));
                check(H5Pset_shuffle(create.get()));
                check(H5Pset_deflate(create.get(), DEFLATE_LEVEL));

                // a partly written chunk stays in the cache until it is full
                const Handle access{H5Pcreate(H5P_DATASET_ACCESS), H5Pclose};
                check(H5Pset_chunk_cache(
                    access.get(),
                    CACHE_SLOTS,
                    2 * framesPerChunk * frameBytes,
                    1.0
                ));

                return {
                    H5Dcreate2(
                        parent,
                        name,
                        fileType,
                        space.get(),
                        H5P_DEFAULT,
                        create.get(),
                        access.get()
                    ),
                    H5Dclose
                };
            }

           public:
            Series(
                const hid_t          parent,
                const char          *name,
                const hid_t          fileType,
                const hid_t          memType,
                std::vector<hsize_t> frameShape
            )
                : _set(create(parent, name, fileType, frameShape)),
                  _memType(memType),
                  _frameShape(std::move(frameShape))
            {
            }

            [[nodiscard]] hid_t get() const { return _set.get(); }

            void append(const void *data)
            {
                std::vector<hsize_t> dims{_frames + 1};
                dims.insert(dims.end(), _frameShape.begin(), _frameShape.end());
                check(H5Dset_extent(_set.get(), dims.data()));

                std::vector<hsize_t> start(dims.size(), 0);
                start[0]   = _frames;
                auto count = dims;
                count[0]   = 1;

                const Handle fileSpace{H5Dget_space(_set.get()), H5Sclose};
                check(H5Sselect_hyperslab(
                    fileSpace.get(),
                    H5S_SELECT_SET,
                    start.data(),
                    nullptr,
                    count.data(),
                    nullptr
                ));

                const Handle memSpace{
                    H5Screate_simple(
                        static_cast<int>(count.size()),
                        count.data(),
                        nullptr
                    ),
                    H5Sclose
                };

                check(H5Dwrite(
                    _set.get(),
                    _memType,
                    memSpace.get(),
                    fileSpace.get(),
                    H5P_DEFAULT,
                    data
                ));
                ++_frames;
            }
        };

        /**
         * @brief H5MD time-dependent element: step, time and value
         */
        class Element
        {
           private:
            Handle _group;
            Series _step;
            Series _time;
            Series _value;

           public:
            Element(
                const hid_t                 parent,
                const char                 *name,
                const hid_t                 fileType,
                const hid_t                 memType,
                const std::vector<hsize_t> &frameShape,
                const std::string          &unit
            )
                : _group(group(parent, name)),
                  _step(
                      _group.get(),
                      "step",
                      H5T_STD_I64LE,
                      H5T_NATIVE_INT64,
                      {}
                  ),
                  _time(
                      _group.get(),
                      "time",
                      H5T_IEEE_F64LE,
                      H5T_NATIVE_DOUBLE,
                      {}
                  ),
                  _value(_group.get(), "value", fileType, memType, frameShape)
            {
                attribute(_time.get(), "unit", "fs");
                attribute(_value.get(), "unit", unit);
            }

            void append(
                const std::int64_t step,
                const double       time,
                const void        *value
            )
            {
                _step.append(&step);
                _time.append(&time);
                _value.append(value);
            }
        };

        /**
         * @brief H5MD species: atomic number, 0 for atoms without element
         */
        int species(const std::string &name)
        {
            const auto key = utilities::toLowerCopy(name);

            if (!atomNumberMap.contains(key) || key == "dum")
                return NO_ELEMENT;

            const auto number = atomNumberMap.at(key).get();
            return number <= MAX_ELEMENT ? static_cast<int>(number)
                                         : NO_ELEMENT;
        }

    }   // namespace

    /**
     * @brief open HDF5 file and the H5MD elements written each frame
     */
    struct H5mdOutput::Impl
    {
        Handle      file;
        double      timeStep;
        std::size_t nAtoms = 0;

        std::unique_ptr<Element> box;
        std::unique_ptr<Element> position;
        std::unique_ptr<Element> velocity;
        std::unique_ptr<Element> force;
        std::unique_ptr<Element> charge;
        std::unique_ptr<Element> potentialEnergy;

        std::vector<float> buffer;

        Impl(const std::string &filename, const bool overwrite, const double dt)
            : file(
                  H5Fcreate(
                      filename.c_str(),
                      overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL,
                      H5P_DEFAULT,
                      H5P_DEFAULT
                  ),
                  H5Fclose
              ),
              timeStep(dt)
        {
            const auto h5md = group(file.get(), "h5md");
            attribute(h5md.get(), "version", std::vector<int>{1, 1});

            const auto author = group(h5md.get(), "author");
            attribute(author.get(), "name", "PQ");

            const auto creator = group(h5md.get(), "creator");
            attribute(creator.get(), "name", "PQ");
            attribute(creator.get(), "version", COMPILE_VERSION_);
        }

        void init(const Frame &frame)
        {
            nAtoms = frame.atoms.size();

            std::vector<int>         speciesOfAtoms;
            std::vector<std::string> names;
            for (const auto &atom : frame.atoms)
            {
                speciesOfAtoms.push_back(species(atom.name));
                names.push_back(atom.name);
            }

            const auto particles = group(file.get(), "particles");
            const auto all       = group(particles.get(), "all");

            const auto   n = static_cast<hsize_t>(nAtoms);
            const Handle space{H5Screate_simple(1, &n, nullptr), H5Sclose};
            const Handle set{
                H5Dcreate2(
                    all.get(),
                    "species",
                    H5T_STD_I32LE,
                    space.get(),
                    H5P_DEFAULT,
                    H5P_DEFAULT,
                    H5P_DEFAULT
                ),
                H5Dclose
            };
            check(H5Dwrite(
                set.get(),
                H5T_NATIVE_INT,
                H5S_ALL,
                H5S_ALL,
                H5P_DEFAULT,
                speciesOfAtoms.data()
            ));

            const auto boxGroup = group(all.get(), "box");
            attribute(boxGroup.get(), "dimension", std::vector<int>{3});
            strings(
                boxGroup.get(),
                "boundary",
                {"periodic", "periodic", "periodic"},
                true
            );

            box = std::make_unique<Element>(
                boxGroup.get(),
                "edges",
                H5T_IEEE_F64LE,
                H5T_NATIVE_DOUBLE,
                std::vector<hsize_t>{3, 3},
                "Angstrom"
            );
            position = std::make_unique<Element>(
                all.get(),
                "position",
                H5T_IEEE_F32LE,
                H5T_NATIVE_FLOAT,
                std::vector<hsize_t>{n, 3},
                "Angstrom"
            );
            velocity = std::make_unique<Element>(
                all.get(),
                "velocity",
                H5T_IEEE_F32LE,
                H5T_NATIVE_FLOAT,
                std::vector<hsize_t>{n, 3},
                "Angstrom fs-1"
            );
            force = std::make_unique<Element>(
                all.get(),
                "force",
                H5T_IEEE_F32LE,
                H5T_NATIVE_FLOAT,
                std::vector<hsize_t>{n, 3},
                "kcal mol-1 Angstrom-1"
            );
            charge = std::make_unique<Element>(
                all.get(),
                "charge",
                H5T_IEEE_F32LE,
                H5T_NATIVE_FLOAT,
                std::vector<hsize_t>{n},
                "e"
            );

            const auto observables = group(file.get(), "observables");
            potentialEnergy        = std::make_unique<Element>(
                observables.get(),
                "potential_energy",
                H5T_IEEE_F64LE,
                H5T_NATIVE_DOUBLE,
                std::vector<hsize_t>{},
                "kcal mol-1"
            );

            const auto parameters = group(file.get(), "parameters");
            strings(parameters.get(), "atom_names", names, false);

            buffer.resize(3 * nAtoms);
        }

        template <typename Get>
        void appendVectors(
            Element           &element,
            const std::int64_t step,
            const double       time,
            Get                get
        )
        {
            for (std::size_t i = 0; i < nAtoms; ++i)
            {
                const auto value = get(i);
                for (std::size_t d = 0; d < 3; ++d)
                    buffer[(3 * i) + d] = static_cast<float>(value[d]);
            }

            element.append(step, time, buffer.data());
        }

        void write(const Frame &frame, const std::size_t step)
        {
            if (nAtoms == 0)
                init(frame);

            if (frame.atoms.size() != nAtoms)
                throw exc::InputFileException(
                    "H5MD output needs the same number of atoms in every frame"
                );

            const auto stepValue = static_cast<std::int64_t>(step);
            const auto time      = static_cast<double>(step) * timeStep;

            const auto            matrix = frame.box->getBoxMatrix();
            std::array<double, 9> edges{};
            for (std::size_t vector = 0; vector < 3; ++vector)
                for (std::size_t i = 0; i < 3; ++i)
                    edges[(3 * vector) + i] = matrix[i][vector];

            box->append(stepValue, time, edges.data());

            const auto &atoms = frame.atoms;
            appendVectors(
                *position,
                stepValue,
                time,
                [&](const std::size_t i)
                { return frame.box->wrapPositionIntoBox(atoms[i].position); }
            );
            appendVectors(
                *velocity,
                stepValue,
                time,
                [&](const std::size_t i)
                { return atoms[i].velocity * PER_S_TO_PER_FS; }
            );
            appendVectors(
                *force,
                stepValue,
                time,
                [&](const std::size_t i) { return atoms[i].force; }
            );

            for (std::size_t i = 0; i < nAtoms; ++i)
                buffer[i] = static_cast<float>(atoms[i].charge);
            charge->append(stepValue, time, buffer.data());

            potentialEnergy->append(stepValue, time, &frame.potentialEnergy);

            check(H5Fflush(file.get(), H5F_SCOPE_LOCAL));
        }
    };

    /**
     * @brief create the H5MD file
     *
     * @param filename
     * @param overwrite whether an existing file may be replaced
     * @param timeStep in fs, for the time of each frame
     *
     * @throw InputFileException if the file exists and overwrite is false
     */
    H5mdOutput::H5mdOutput(
        const std::string &filename,
        const bool         overwrite,
        const double       timeStep
    )
    {
        if (!overwrite && std::filesystem::exists(filename))
            throw exc::InputFileException(
                "File already exists - filename = " + filename
            );

        _impl = std::make_unique<Impl>(filename, overwrite, timeStep);
    }

    /**
     * @brief append one frame
     *
     * @param frame
     * @param step
     */
    void H5mdOutput::write(const Frame &frame, const size_t step)
    {
        _impl->write(frame, step);
    }

}   // namespace out

#else   // WITH_HDF5

namespace out
{
    struct H5mdOutput::Impl
    {
    };

    /**
     * @brief without HDF5 nothing is written; EngineOutput::openH5mdOutput
     * refuses to open it
     */
    H5mdOutput::H5mdOutput(
        [[maybe_unused]] const std::string &filename,
        [[maybe_unused]] const bool         overwrite,
        [[maybe_unused]] const double       timeStep
    )
    {
    }

    void H5mdOutput::write(
        [[maybe_unused]] const Frame &frame,
        [[maybe_unused]] const size_t step
    )
    {
    }

}   // namespace out

#endif   // WITH_HDF5

namespace out
{
    H5mdOutput::~H5mdOutput() = default;

    /**
     * @brief whether PQ was built with HDF5
     *
     * @return bool
     */
    bool H5mdOutput::available()
    {
#ifdef WITH_HDF5
        return true;
#else
        return false;
#endif
    }

    /**
     * @brief write everything and close the file
     */
    void H5mdOutput::close() { _impl.reset(); }

}   // namespace out
