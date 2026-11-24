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
#include <cstddef>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>

#include "atomicNumber.hpp"
#include "box.hpp"
#include "exceptions.hpp"
#include "frame.hpp"
#include "staticMatrix.hpp"
#include "ulmWriter.hpp"
#include "units.hpp"

namespace out
{
    /**
     * @brief ASE trajectory written to filename once setFilename opens it
     *
     * @param filename
     */
    AseTrajOutput::AseTrajOutput(std::string filename)
        : Output(std::move(filename))
    {
    }

    AseTrajOutput::~AseTrajOutput() = default;

    /**
     * @brief append one frame like ase.io.Trajectory.write and flush
     *
     * @details the first frame also holds the file header, atomic numbers,
     * masses and pbc, which ASE reads for all frames
     *
     * @param frame
     * @param step stored as info step
     *
     * @throw InputFileException if the file is closed or the number of
     * atoms changed
     */
    void AseTrajOutput::write(const Frame &frame, size_t step)
    {
        _checkFrame(frame);

        ulm::JsonObject item;

        if (_writer.getNItems() == 0)
        {
            _writer.writeHeader(_getFilePointer());
            _addSystem(item, frame);
        }

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto position = frame.box->wrapPositionIntoBox(atom.position)
                                      .numerical_value_in(angstrom);
            _buffer.insert(
                _buffer.end(),
                {position[0], position[1], position[2]}
            );
        }
        _addBuffer(item, "positions", {_nAtoms, 3});

        const auto cell = linalg::transpose(frame.box->getBoxMatrix())
                              .numerical_value_in(angstrom)
                              .toStdVector();
        item.addMatrix(
            "cell",
            std::span<const double, ulm::MATRIX_SIZE>{cell}
        );

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            const auto mass = atom.mass.numerical_value_in(g_per_mol);
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                _buffer.push_back(
                    mass * atom.velocity[dim].numerical_value_in(sqrt(eV / amu))
                );
            }
        }
        _addBuffer(item, "momenta", {_nAtoms, 3});

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            _buffer.push_back(atom.charge.numerical_value_in(elementary_charge)
            );
        }
        _addBuffer(item, "charges", {_nAtoms});

        const auto energy =
            frame.potentialEnergy / avogadro_constant;

        ulm::JsonObject calculator;
        calculator.addString("name", "pq");
        calculator.addObject("parameters", ulm::JsonObject{});
        calculator.addNumber("energy", energy.numerical_value_in(eV));

        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            for (std::size_t dim = 0; dim < 3; ++dim)
            {
                const auto force = atom.force[dim] / avogadro_constant;
                _buffer.push_back(force.numerical_value_in(eV / angstrom));
            }
        }
        _addBuffer(calculator, "forces", {_nAtoms, 3});
        item.addChild("calculator", calculator);

        ulm::JsonObject info;
        info.addInteger("step", step);
        item.addObject("info", info);

        _writer.writeItem(_getFilePointer(), item.str());
    }

    /**
     * @brief check that the file is open and the number of atoms did not
     * change since the first frame
     *
     * @param frame
     *
     * @throw InputFileException if not
     */
    void AseTrajOutput::_checkFrame(const Frame &frame)
    {
        if (!_getFilePointer().is_open())
        {
            throw exc::InputFileException(
                "ASE trajectory is closed - filename = " + getFilename()
            );
        }

        if (_writer.getNItems() == 0)
        {
            _nAtoms = frame.atoms.size();
        }

        if (frame.atoms.size() != _nAtoms)
        {
            throw exc::InputFileException(
                "ASE trajectory output needs the same number of atoms in "
                "every frame"
            );
        }
    }

    /**
     * @brief atomic numbers, masses and pbc of the first frame
     *
     * @param item
     * @param frame
     */
    void AseTrajOutput::_addSystem(ulm::JsonObject &item, const Frame &frame)
    {
        _numbers.clear();
        _buffer.clear();
        for (const auto &atom : frame.atoms)
        {
            _numbers.push_back(atomicNumber(atom.name));
            _buffer.push_back(atom.mass.numerical_value_in(g_per_mol));
        }

        item.addInteger("version", 1);
        item.addString("ase_version", "3.29.0");
        item.addBools("pbc", std::array{true, true, true});
        item.addArray(
            "numbers",
            _writer.writeArray(_getFilePointer(), _numbers, {_nAtoms})
        );
        _addBuffer(item, "masses", {_nAtoms});
    }

    /**
     * @brief store the buffer as float64 array under key of object
     *
     * @param object
     * @param key
     * @param shape
     */
    void AseTrajOutput::_addBuffer(
        ulm::JsonObject                &object,
        std::string_view                key,
        const std::vector<std::size_t> &shape
    )
    {
        object.addArray(
            key,
            _writer.writeArray(_getFilePointer(), _buffer, shape)
        );
    }

}   // namespace out
