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

#include "aseTrajOutput.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <ios>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "atomicNumber.hpp"
#include "box.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "staticMatrix.hpp"
#include "ulmJson.hpp"
#include "units.hpp"

static_assert(
    std::endian::native == std::endian::little,
    "the ASE trajectory output writes little endian data only"
);

namespace out
{
    // ASE units: eV, Angstrom, amu, time unit Angstrom * sqrt(amu / eV)
    const double KCAL_PER_MOL_IN_EV =
        (1.0 * kcal_per_mol / avogadro_constant).numerical_value_in(eV);
    const double ASE_TIME_IN_S =
        std::sqrt((1.0 * amu / eV).numerical_value_in((s * s) / angstrom2));

    // ULM layout of ase/io/ulm.py (version 3)
    constexpr std::string_view ULM_MAGIC       = "- of UlmASE-Trajectory  ";
    constexpr std::int64_t     ULM_VERSION     = 3;
    constexpr std::int64_t     NITEMS_POSITION = 32;
    constexpr std::int64_t     OFFSETS_POINTER = 40;
    constexpr std::int64_t     HEADER_SIZE     = 48;
    constexpr std::size_t      OFFSETS_GROWTH  = 42;
    constexpr std::int64_t     ALIGNMENT       = 8;
    constexpr std::int64_t     INT_SIZE        = 8;
    constexpr std::string_view ASE_VERSION     = "3.29.0";

    /**
     * @brief write values as raw bytes
     *
     * @tparam T
     * @param values
     */
    template <typename T>
    void AseTrajOutput::_writeRaw(std::span<const T> values)
    {
        _bytes.resize(values.size_bytes());
        std::memcpy(_bytes.data(), values.data(), values.size_bytes());
        _file.write(_bytes.data(), static_cast<std::streamsize>(_bytes.size()));
    }

    /**
     * @brief write an 8-byte aligned array
     *
     * @tparam T double or std::int64_t
     * @param values in C order
     * @param shape
     * @return std::string its JSON entry
     */
    template <typename T>
    std::string AseTrajOutput::_writeArray(
        std::span<const T>           values,
        std::span<const std::size_t> shape
    )
    {
        static_assert(
            std::is_same_v<T, double> || std::is_same_v<T, std::int64_t>
        );
        constexpr std::string_view dtype =
            std::is_same_v<T, double> ? "float64" : "int64";

        const auto offset = _align();
        _writeRaw(values);

        return ulm::ndarray(shape, dtype, offset);
    }

    /**
     * @brief write a 3-vector per atom as float64 array
     *
     * @tparam Get
     * @param frame
     * @param get vector of an atom as plain numbers in ASE units
     * @return std::string its JSON entry
     */
    template <typename Get>
    std::string AseTrajOutput::_writeVectors(const Frame &frame, Get get)
    {
        _values.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto vector = get(atom);
            _values.insert(_values.end(), {vector[0], vector[1], vector[2]});
        }

        const auto shape = std::array{_nAtoms, std::size_t{3}};
        return _writeArray(std::span<const double>{_values}, shape);
    }

    /**
     * @brief create the file and write the ULM header of an empty
     * trajectory
     *
     * @param filename
     * @param overwrite whether an existing file may be replaced
     *
     * @throw InputFileException if the file exists and overwrite is false,
     * cannot be opened or written
     */
    AseTrajOutput::AseTrajOutput(const std::string &filename, bool overwrite)
        : _filename(filename), _offsets(1, -1), _offsetsPosition(HEADER_SIZE)
    {
        if (!overwrite && std::filesystem::exists(filename))
        {
            throw exc::InputFileException(
                "File already exists - filename = " + filename
            );
        }

        _file.open(filename, std::ios::binary | std::ios::trunc);
        if (!_file.is_open())
        {
            throw exc::InputFileException(
                "Could not open file - filename = " + filename
            );
        }

        _file.write(
            ULM_MAGIC.data(),
            static_cast<std::streamsize>(ULM_MAGIC.size())
        );
        _writeInt(ULM_VERSION);
        _writeInt(_nItems);
        _writeInt(_offsetsPosition);
        _writeRaw(std::span<const std::int64_t>{_offsets});
        _file.flush();
        _checkFile();
    }

    AseTrajOutput::~AseTrajOutput() = default;

    /**
     * @brief append one frame like ase.io.Trajectory.write and flush
     *
     * @details the first frame also holds the atomic numbers, masses and
     * pbc, which ASE reads for all frames
     *
     * @param frame
     * @param step stored as info step
     *
     * @throw InputFileException if the file is closed, the number of atoms
     * changed or writing failed
     */
    void AseTrajOutput::write(const Frame &frame, size_t step)
    {
        if (!_file.is_open())
        {
            throw exc::InputFileException(
                "ASE trajectory is closed - filename = " + _filename
            );
        }

        const auto &atoms = frame.atoms;
        const bool  first = _nItems == 0;

        if (first)
        {
            _nAtoms = atoms.size();
        }

        if (atoms.size() != _nAtoms)
        {
            throw exc::InputFileException(
                "ASE trajectory output needs the same number of atoms in "
                "every frame"
            );
        }

        const auto scalarShape = std::array{_nAtoms};

        std::string json = "{";

        if (first)
        {
            std::vector<std::int64_t> numbers;
            _values.clear();
            for (const auto &atom : atoms)
            {
                numbers.push_back(atomicNumber(atom.name));
                _values.push_back(atom.mass.numerical_value_in(g_per_mol));
            }

            json += R"("version": 1, "ase_version": ")";
            json += ASE_VERSION;
            json += R"(", "pbc": [true, true, true], "numbers.": )";
            json += _writeArray(
                std::span<const std::int64_t>{numbers},
                scalarShape
            );
            json += R"(, "masses.": )";
            json += _writeArray(std::span<const double>{_values}, scalarShape);
            json += ", ";
        }

        json += R"("positions.": )";
        json += _writeVectors(
            frame,
            [&frame](const Frame::Atom &atom)
            {
                return frame.box->wrapPositionIntoBox(atom.position)
                    .numerical_value_in(angstrom);
            }
        );

        const auto cell = linalg::transpose(frame.box->getBoxMatrix())
                              .numerical_value_in(angstrom)
                              .toStdVector();

        json += R"(, "cell": [)";
        for (std::size_t index = 0; index < cell.size(); ++index)
        {
            if (index % 3 == 0)
            {
                json += index == 0 ? "[" : "], [";
            }
            else
            {
                json += ", ";
            }

            json += ulm::number(cell[index]);
        }
        json += "]]";

        json += R"(, "momenta.": )";
        json += _writeVectors(
            frame,
            [](const Frame::Atom &atom)
            {
                return atom.velocity.numerical_value_in(angstrom / s) *
                       atom.mass.numerical_value_in(g_per_mol) * ASE_TIME_IN_S;
            }
        );

        _values.clear();
        for (const auto &atom : atoms)
        {
            _values.push_back(atom.charge.numerical_value_in(elementary_charge)
            );
        }
        json += R"(, "charges.": )";
        json += _writeArray(std::span<const double>{_values}, scalarShape);

        json += R"(, "calculator.": {"name": "pq", "parameters": {}, )";
        json += R"("energy": )";
        json += ulm::number(frame.potentialEnergy * KCAL_PER_MOL_IN_EV);
        json += R"(, "forces.": )";
        json += _writeVectors(
            frame,
            [](const Frame::Atom &atom)
            {
                return atom.force.numerical_value_in(kcal_per_mol / angstrom) *
                       KCAL_PER_MOL_IN_EV;
            }
        );
        json += R"(}, "info": {"step": )";
        json += std::to_string(step);
        json += "}}";

        const auto jsonPosition = _position();
        _writeInt(static_cast<std::int64_t>(json.size()));
        _file.write(json.data(), static_cast<std::streamsize>(json.size()));

        _addItem(jsonPosition);
    }

    /**
     * @brief close the file; every frame is already complete on disk
     */
    void AseTrajOutput::close()
    {
        if (_file.is_open())
        {
            _file.close();
        }
    }

    /**
     * @brief current write position
     *
     * @return std::int64_t
     */
    std::int64_t AseTrajOutput::_position()
    {
        return static_cast<std::int64_t>(_file.tellp());
    }

    /**
     * @brief pad with '#' to the next multiple of 8 bytes like ASE
     *
     * @return std::int64_t the aligned position
     */
    std::int64_t AseTrajOutput::_align()
    {
        const auto position = _position();
        const auto rest     = position % ALIGNMENT;

        if (rest == 0)
        {
            return position;
        }

        const std::string padding(
            static_cast<std::size_t>(ALIGNMENT - rest),
            '#'
        );
        _file.write(
            padding.data(),
            static_cast<std::streamsize>(padding.size())
        );

        return position + ALIGNMENT - rest;
    }

    /**
     * @brief write a little endian int64 at the current position
     *
     * @param value
     */
    void AseTrajOutput::_writeInt(std::int64_t value)
    {
        _writeRaw(std::span<const std::int64_t>{&value, 1});
    }

    /**
     * @brief overwrite a little endian int64 at position
     *
     * @param value
     * @param position
     */
    void AseTrajOutput::_writeIntAt(std::int64_t value, std::int64_t position)
    {
        _file.seekp(position);
        _writeInt(value);
    }

    /**
     * @brief register the frame whose JSON starts at jsonPosition
     *
     * @details like Writer.sync in ase/io/ulm.py: a full offsets table is
     * copied to the end of the file with 42 times the size
     *
     * @param jsonPosition
     *
     * @throw InputFileException if writing failed
     */
    void AseTrajOutput::_addItem(std::int64_t jsonPosition)
    {
        const auto index = static_cast<std::size_t>(_nItems);

        if (index >= _offsets.size())
        {
            _offsets.resize(_offsets.size() * OFFSETS_GROWTH, 0);
            _offsetsPosition = _align();
            _writeRaw(std::span<const std::int64_t>{_offsets});
            _writeIntAt(_offsetsPosition, OFFSETS_POINTER);
        }

        _offsets[index] = jsonPosition;
        _writeIntAt(jsonPosition, _offsetsPosition + (_nItems * INT_SIZE));

        ++_nItems;
        _writeIntAt(_nItems, NITEMS_POSITION);

        _file.flush();
        _file.seekp(0, std::ios::end);
        _checkFile();
    }

    /**
     * @brief fail loudly if a write did not reach the file
     *
     * @throw InputFileException if the stream is in an error state
     */
    void AseTrajOutput::_checkFile() const
    {
        if (!_file.good())
        {
            throw exc::InputFileException(
                "Could not write ASE trajectory - filename = " + _filename
            );
        }
    }

}   // namespace out
