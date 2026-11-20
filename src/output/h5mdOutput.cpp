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
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "atomNumberMap.hpp"
#include "box.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "staticMatrix.hpp"
#include "stringUtilities.hpp"
#include "systemInfo.hpp"

#ifdef WITH_HDF5
#include "hdf5Wrapper.hpp"
#endif

namespace out
{
    // species of atoms without an element (dummy and virtual atoms)
    constexpr int NO_ELEMENT  = 0;
    constexpr int MAX_ELEMENT = 118;

    /**
     * @brief H5MD species of an atom: its atomic number, 0 for atoms without
     * an element (dummy and virtual atoms)
     *
     * @param name PQ atom name
     * @return int
     */
    int H5mdOutput::species(const std::string &name)
    {
        const auto key = utilities::toLowerCopy(name);

        if (!atomNumberMap.contains(key) || key == "dum")
        {
            return NO_ELEMENT;
        }

        const auto number = atomNumberMap.at(key).get();
        if (number > MAX_ELEMENT)
        {
            return NO_ELEMENT;
        }

        return static_cast<int>(number);
    }

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

}   // namespace out

#ifdef WITH_HDF5

namespace out
{
    // H5MD readers know Angstrom fs-1, not the Angstrom s-1 of PQ
    constexpr double PER_S_TO_PER_FS = 1.0e-15;

    /**
     * @brief open HDF5 file and the H5MD elements written each frame
     */
    struct H5mdOutput::Impl
    {
        hdf5::Handle file;
        double       timeStep;
        std::size_t  nAtoms = 0;

        std::unique_ptr<hdf5::Element<double>> box;
        std::unique_ptr<hdf5::Element<float>>  position;
        std::unique_ptr<hdf5::Element<float>>  velocity;
        std::unique_ptr<hdf5::Element<float>>  force;
        std::unique_ptr<hdf5::Element<float>>  charge;
        std::unique_ptr<hdf5::Element<double>> potentialEnergy;

        std::vector<float> buffer;

        Impl(const std::string &filename, bool overwrite, double timeStepInFs);

        void init(const Frame &frame);
        void write(const Frame &frame, std::size_t step);

        template <typename Get>
        void appendVectors(
            hdf5::Element<float> &element,
            std::int64_t          step,
            double                time,
            Get                   get
        );
    };

    /**
     * @brief create the file and write the H5MD header
     *
     * @param filename
     * @param overwrite whether an existing file may be replaced
     * @param timeStepInFs
     */
    H5mdOutput::Impl::Impl(
        const std::string &filename,
        bool               overwrite,
        double             timeStepInFs
    )
        : file(
              H5Fcreate(
                  filename.c_str(),
                  overwrite ? H5F_ACC_TRUNC : H5F_ACC_EXCL,
                  H5P_DEFAULT,
                  H5P_DEFAULT
              ),
              H5Fclose
          ),
          timeStep(timeStepInFs)
    {
        const auto h5md = hdf5::createGroup(file.get(), "h5md");
        hdf5::writeAttribute(h5md.get(), "version", std::vector<int>{1, 1});

        const auto author = hdf5::createGroup(h5md.get(), "author");
        hdf5::writeAttribute(author.get(), "name", std::string{"PQ"});

        const auto creator = hdf5::createGroup(h5md.get(), "creator");
        hdf5::writeAttribute(creator.get(), "name", std::string{"PQ"});
        hdf5::writeAttribute(
            creator.get(),
            "version",
            std::string{COMPILE_VERSION_}
        );
    }

    /**
     * @brief create the particle, observable and parameter groups from the
     * first frame
     *
     * @param frame
     */
    void H5mdOutput::Impl::init(const Frame &frame)
    {
        nAtoms = frame.atoms.size();

        std::vector<int>         speciesOfAtoms;
        std::vector<std::string> names;
        for (const auto &atom : frame.atoms)
        {
            speciesOfAtoms.push_back(H5mdOutput::species(atom.name));
            names.push_back(atom.name);
        }

        const auto particles = hdf5::createGroup(file.get(), "particles");
        const auto all       = hdf5::createGroup(particles.get(), "all");
        hdf5::writeDataset(all.get(), "species", speciesOfAtoms);

        const auto boxGroup = hdf5::createGroup(all.get(), "box");
        hdf5::writeAttribute(boxGroup.get(), "dimension", std::vector<int>{3});
        hdf5::writeAttribute(
            boxGroup.get(),
            "boundary",
            std::vector<std::string>{"periodic", "periodic", "periodic"}
        );

        const auto nAtomsShape = static_cast<hsize_t>(nAtoms);

        box = std::make_unique<hdf5::Element<double>>(
            boxGroup.get(),
            "edges",
            H5T_IEEE_F64LE,
            hdf5::Shape{3, 3},
            "Angstrom"
        );
        position = std::make_unique<hdf5::Element<float>>(
            all.get(),
            "position",
            H5T_IEEE_F32LE,
            hdf5::Shape{nAtomsShape, 3},
            "Angstrom"
        );
        velocity = std::make_unique<hdf5::Element<float>>(
            all.get(),
            "velocity",
            H5T_IEEE_F32LE,
            hdf5::Shape{nAtomsShape, 3},
            "Angstrom fs-1"
        );
        force = std::make_unique<hdf5::Element<float>>(
            all.get(),
            "force",
            H5T_IEEE_F32LE,
            hdf5::Shape{nAtomsShape, 3},
            "kcal mol-1 Angstrom-1"
        );
        charge = std::make_unique<hdf5::Element<float>>(
            all.get(),
            "charge",
            H5T_IEEE_F32LE,
            hdf5::Shape{nAtomsShape},
            "e"
        );

        const auto observables = hdf5::createGroup(file.get(), "observables");
        potentialEnergy        = std::make_unique<hdf5::Element<double>>(
            observables.get(),
            "potential_energy",
            H5T_IEEE_F64LE,
            hdf5::Shape{},
            "kcal mol-1"
        );

        const auto parameters = hdf5::createGroup(file.get(), "parameters");
        hdf5::writeDataset(parameters.get(), "atom_names", names);

        buffer.resize(3 * nAtoms);
    }

    /**
     * @brief append a 3-vector per atom as float
     *
     * @param element
     * @param step
     * @param time in fs
     * @param get vector of atom i
     */
    template <typename Get>
    void H5mdOutput::Impl::appendVectors(
        hdf5::Element<float> &element,
        std::int64_t          step,
        double                time,
        Get                   get
    )
    {
        for (std::size_t atomIndex = 0; atomIndex < nAtoms; ++atomIndex)
        {
            const auto value = get(atomIndex);
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                buffer[(3 * atomIndex) + dim] = static_cast<float>(value[dim]);
            }
        }

        element.append(step, time, std::span<const float>{buffer});
    }

    /**
     * @brief append one frame to every element and flush the file
     *
     * @param frame
     * @param step
     *
     * @throw InputFileException if the number of atoms changed
     */
    void H5mdOutput::Impl::write(const Frame &frame, std::size_t step)
    {
        if (nAtoms == 0)
        {
            init(frame);
        }

        if (frame.atoms.size() != nAtoms)
        {
            throw exc::InputFileException(
                "H5MD output needs the same number of atoms in every frame"
            );
        }

        const auto stepValue = static_cast<std::int64_t>(step);
        const auto time      = static_cast<double>(step) * timeStep;

        // H5MD edges are the box vectors as rows
        const auto edges =
            linalg::transpose(frame.box->getBoxMatrix()).toStdVector();

        box->append(stepValue, time, std::span<const double>{edges});

        const auto &atoms = frame.atoms;
        appendVectors(
            *position,
            stepValue,
            time,
            [&](std::size_t atomIndex)
            {
                return frame.box->wrapPositionIntoBox(
                    atoms[atomIndex].position
                );
            }
        );
        appendVectors(
            *velocity,
            stepValue,
            time,
            [&](std::size_t atomIndex)
            { return atoms[atomIndex].velocity * PER_S_TO_PER_FS; }
        );
        appendVectors(
            *force,
            stepValue,
            time,
            [&](std::size_t atomIndex) { return atoms[atomIndex].force; }
        );

        for (std::size_t atomIndex = 0; atomIndex < nAtoms; ++atomIndex)
        {
            buffer[atomIndex] = static_cast<float>(atoms[atomIndex].charge);
        }
        charge->append(
            stepValue,
            time,
            std::span<const float>{buffer.data(), nAtoms}
        );

        potentialEnergy->append(
            stepValue,
            time,
            std::span<const double>{&frame.potentialEnergy, 1}
        );

        hdf5::check(H5Fflush(file.get(), H5F_SCOPE_LOCAL));
    }

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
        bool               overwrite,
        double             timeStep
    )
    {
        if (!overwrite && std::filesystem::exists(filename))
        {
            throw exc::InputFileException(
                "File already exists - filename = " + filename
            );
        }

        _impl = std::make_unique<Impl>(filename, overwrite, timeStep);
    }

    /**
     * @brief append one frame
     *
     * @param frame
     * @param step
     */
    void H5mdOutput::write(const Frame &frame, size_t step)
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
     * @brief without HDF5 there is no file; EngineOutput::openH5mdOutput
     * refuses traj_format = h5md before
     */
    H5mdOutput::H5mdOutput(
        const std::string & /*filename*/,
        bool /*overwrite*/,
        double /*timeStep*/
    )
    {
    }

    /**
     * @brief without HDF5 there is no file to write to
     */
    void H5mdOutput::write(const Frame & /*frame*/, size_t /*step*/) {}

}   // namespace out

#endif   // WITH_HDF5

namespace out
{
    // after Impl is complete in both builds
    H5mdOutput::~H5mdOutput() = default;

    /**
     * @brief write everything and close the file
     */
    void H5mdOutput::close() { _impl.reset(); }

}   // namespace out
