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

#include "zarrOutput.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "atomicNumber.hpp"
#include "box.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "staticMatrix.hpp"
#include "unit_types.hpp"
#include "units.hpp"
#include "version.hpp"
#include "zarrWriter.hpp"

namespace out
{
    /**
     * @brief create the store directory and its root group
     *
     * @details the chunks are zstd compressed if PQ was built with zstd
     *
     * @param filename directory of the store
     * @param overwrite whether an existing store may be replaced
     * @param timeStep for the time of each frame
     *
     * @throw InputFileException if filename exists and overwrite is false,
     * or if it is a directory without zarr.json
     */
    ZarrOutput::ZarrOutput(
        const std::string &filename,
        bool               overwrite,
        Time               timeStep
    )
        : _path(filename),
          _timeStep(timeStep),
          _codec(zarr::zstdAvailable() ? zarr::Codec::ZSTD : zarr::Codec::BYTES)
    {
        if (std::filesystem::exists(_path))
        {
            if (!overwrite)
            {
                throw exc::InputFileException(
                    "File already exists - filename = " + filename
                );
            }

            if (std::filesystem::is_directory(_path) &&
                !std::filesystem::exists(_path / "zarr.json"))
            {
                throw exc::InputFileException(
                    "Directory is not a Zarr store and is not replaced - "
                    "filename = " +
                    filename
                );
            }

            std::filesystem::remove_all(_path);
        }

        zarr::JsonObject author;
        author.addString("name", "PQ");

        zarr::JsonObject creator;
        creator.addString("name", "PQ");
        creator.addString("version", CURRENT_VERSION.to_string());

        zarr::JsonObject h5md;
        h5md.addIntegers("version", {1, 1});
        h5md.addObject("author", author);
        h5md.addObject("creator", creator);

        zarr::JsonObject attributes;
        attributes.addObject("h5md", h5md);

        zarr::writeGroup(_path, attributes);
    }

    ZarrOutput::~ZarrOutput() = default;

    /**
     * @brief append one frame
     *
     * @param frame
     * @param step
     *
     * @throw InputFileException if the number of atoms changed
     */
    void ZarrOutput::write(const Frame &frame, size_t step)
    {
        if (_nAtoms == 0)
        {
            _init(frame);
        }

        if (frame.atoms.size() != _nAtoms)
        {
            throw exc::InputFileException(
                "Zarr output needs the same number of atoms in every frame"
            );
        }

        const auto stepValue = static_cast<std::int64_t>(step);
        const auto time      = static_cast<double>(step) * _timeStep;

        // H5MD edges are the box vectors as rows
        const auto edges = linalg::transpose(frame.box->getBoxMatrix())
                               .numerical_value_in(angstrom)
                               .toStdVector();
        _box->append(stepValue, time, std::span<const double>{edges});

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto position = frame.box->wrapPositionIntoBox(atom.position)
                                      .numerical_value_in(angstrom);
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                _buffer.push_back(static_cast<float>(position[dim]));
            }
        }
        _position->append(stepValue, time, _buffer);

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto velocity = atom.velocity.numerical_value_in(angstrom / fs);
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                _buffer.push_back(static_cast<float>(velocity[dim]));
            }
        }
        _velocity->append(stepValue, time, _buffer);

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto force =
                atom.force.numerical_value_in(kcal_per_mol / angstrom);
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                _buffer.push_back(static_cast<float>(force[dim]));
            }
        }
        _force->append(stepValue, time, _buffer);

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            _buffer.push_back(static_cast<float>(
                atom.charge.numerical_value_in(elementary_charge)
            ));
        }
        _charge->append(stepValue, time, _buffer);

        const auto energy =
            frame.potentialEnergy.numerical_value_in(kcal_per_mol);
        _potentialEnergy
            ->append(stepValue, time, std::span<const double>{&energy, 1});
    }

    /**
     * @brief write the frames of the last chunk
     */
    void ZarrOutput::close()
    {
        if (_nAtoms == 0)
        {
            return;
        }

        _box->flush();
        _position->flush();
        _velocity->flush();
        _force->flush();
        _charge->flush();
        _potentialEnergy->flush();
    }

    /**
     * @brief create the particle, observable and parameter groups from the
     * first frame
     *
     * @param frame
     */
    void ZarrOutput::_init(const Frame &frame)
    {
        _nAtoms = frame.atoms.size();

        std::vector<std::int32_t> species;
        std::vector<std::string>  names;
        for (const auto &atom : frame.atoms)
        {
            species.push_back(atomicNumber(atom.name));
            names.push_back(atom.name);
        }

        const auto all = _path / "particles" / "all";
        zarr::writeGroup(_path / "particles", {});
        zarr::writeGroup(all, {});
        zarr::writeArray(
            all / "species",
            std::span<const std::int32_t>{species},
            _codec
        );

        zarr::JsonObject boxAttributes;
        boxAttributes.addInteger("dimension", 3);
        boxAttributes.addStrings(
            "boundary",
            {"periodic", "periodic", "periodic"}
        );
        zarr::writeGroup(all / "box", boxAttributes);

        _box = std::make_unique<zarr::Element<double>>(
            all / "box" / "edges",
            std::vector<std::size_t>{3, 3},
            ZARR_CHUNK_FRAMES,
            _codec,
            "Angstrom"
        );
        _position = std::make_unique<zarr::Element<float>>(
            all / "position",
            std::vector<std::size_t>{_nAtoms, 3},
            ZARR_CHUNK_FRAMES,
            _codec,
            "Angstrom"
        );
        _velocity = std::make_unique<zarr::Element<float>>(
            all / "velocity",
            std::vector<std::size_t>{_nAtoms, 3},
            ZARR_CHUNK_FRAMES,
            _codec,
            "Angstrom fs-1"
        );
        _force = std::make_unique<zarr::Element<float>>(
            all / "force",
            std::vector<std::size_t>{_nAtoms, 3},
            ZARR_CHUNK_FRAMES,
            _codec,
            "kcal mol-1 Angstrom-1"
        );
        _charge = std::make_unique<zarr::Element<float>>(
            all / "charge",
            std::vector<std::size_t>{_nAtoms},
            ZARR_CHUNK_FRAMES,
            _codec,
            "e"
        );

        zarr::writeGroup(_path / "observables", {});
        _potentialEnergy = std::make_unique<zarr::Element<double>>(
            _path / "observables" / "potential_energy",
            std::vector<std::size_t>{},
            ZARR_CHUNK_FRAMES,
            _codec,
            "kcal mol-1"
        );

        zarr::JsonObject parameters;
        parameters.addStrings("atom_names", names);
        zarr::writeGroup(_path / "parameters", parameters);
    }

}   // namespace out
