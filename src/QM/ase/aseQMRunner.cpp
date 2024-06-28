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

#include "aseQMRunner.hpp"

#include <thread>

#include "physicalData.hpp"
#include "pybind11/embed.h"
#include "simulationBox.hpp"

using QM::ASEQMRunner;
using namespace simulationBox;
using namespace physicalData;
using namespace constants;

namespace
{
    const py::scoped_interpreter guard{};
}

using array_d = py::array_t<double>;
using array_i = py::array_t<int>;

ASEQMRunner::ASEQMRunner()
{
    try
    {
        _atomsModule = py::module_::import("ase.atoms");
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

void ASEQMRunner::run(SimulationBox &simBox, PhysicalData &physicalData)
{
    std::jthread timeoutThread{[this](const std::stop_token stopToken)
                               { throwAfterTimeout(stopToken); }};

    execute(simBox);
    collectData(simBox, physicalData);

    timeoutThread.request_stop();
}

void ASEQMRunner::execute(const SimulationBox &simBox)
{
    buildAseAtoms(simBox);
    _atoms.attr("set_calculator")(_calculator);

    const auto forces = _atoms.attr("get_forces")().cast<array_d>();
    const auto energy = _atoms.attr("get_potential_energy")();
    const auto stress = _atoms.attr("get_stress")(py::arg("voigt") = false);

    _forces = forces.cast<array_d>();
    _energy = energy.cast<double>();
    _stress = stress.cast<array_d>();
}

void ASEQMRunner::collectData(SimulationBox &simBox, PhysicalData &physicalData)
    const
{
    collectForces(simBox);
    collectEnergy(physicalData);
    collectStress(physicalData);
}

void ASEQMRunner::collectForces(SimulationBox &simBox) const
{
    const auto forces = _forces.unchecked<2>();
    const auto nAtoms = simBox.getNumberOfAtoms();

    for (size_t i = 0; i < nAtoms; ++i)
        simBox.getAtoms()[i]->setForce(
            {forces(i, 0) * _EV_TO_KCAL_PER_MOL_,
             forces(i, 1) * _EV_TO_KCAL_PER_MOL_,
             forces(i, 2) * _EV_TO_KCAL_PER_MOL_}
        );
}

void ASEQMRunner::collectEnergy(PhysicalData &physicalData) const
{
    physicalData.setQMEnergy(_energy * _EV_TO_KCAL_PER_MOL_);
}

void ASEQMRunner::collectStress(PhysicalData &physicalData) const
{
    const auto              stress = _stress.unchecked<1>();
    linearAlgebra::tensor3D stress_;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            stress_[j][i] = stress[ssize_t(i * 3 + j)];

    const auto virial = stress_ * physicalData.getVolume();

    physicalData.setStressTensor(stress_ * _EV_TO_KCAL_PER_MOL_);
    physicalData.addVirial(virial * _EV_TO_KCAL_PER_MOL_);
}

void ASEQMRunner::buildAseAtoms(const SimulationBox &simBox)
{
    try
    {
        const auto positions     = asePositions(simBox);
        const auto cell          = aseCell(simBox);
        const auto pbc           = asePBC(simBox);
        const auto atomicNumbers = aseAtomicNumbers(simBox);

        _atoms = _atomsModule.attr("Atoms")(
            py::arg("positions") = positions,
            py::arg("numbers")   = atomicNumbers,
            py::arg("cell")      = cell,
            py::arg("pbc")       = pbc
        );
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

py::array ASEQMRunner::asePositions(const SimulationBox &simBox) const
{
    try
    {
        const auto nAtoms = simBox.getNumberOfAtoms();

        const auto pos             = simBox.flattenPositions();
        auto       positions_array = array_d(ssize_t(nAtoms), &pos[0]);

        const auto shape      = std::vector<size_t>{nAtoms, 3};
        const auto sizeDouble = sizeof(double);
        const auto strides    = std::vector<size_t>{sizeDouble * 3, sizeDouble};

        auto positions_array_reshaped = pybind11::array(py::buffer_info(
            positions_array.mutable_data(),            // Pointer to data
            sizeDouble,                                // Size of one scalar
            py::format_descriptor<double>::format(),   // Data type
            2,                                         // Number of dimensions
            shape,                                     // Shape (N, 3)
            strides                                    // Strides
        ));

        return positions_array_reshaped;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

py::array_t<double> ASEQMRunner::aseCell(const SimulationBox &simBox) const
{
    try
    {
        const auto            boxDimension = simBox.getBoxDimensions();
        const auto            boxAngles    = simBox.getBoxAngles();
        std::array<double, 6> box_array    = {
            boxDimension[0],
            boxDimension[1],
            boxDimension[2],
            boxAngles[0],
            boxAngles[1],
            boxAngles[2]
        };

        auto box_array_ = array_d(6, &box_array[0]);

        return box_array_;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

py::array_t<bool> ASEQMRunner::asePBC(const SimulationBox &) const
{
    try
    {
        const auto          pbc       = std::vector<bool>{true, true, true};
        std::array<bool, 3> pbc_array = {pbc[0], pbc[1], pbc[2]};

        auto pbc_array_ = py::array_t<bool>(3, &pbc_array[0]);

        return pbc_array_;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

py::array_t<int> ASEQMRunner::aseAtomicNumbers(const SimulationBox &simBox
) const
{
    try
    {
        const auto atomicNumbers = simBox.getAtomicNumbers();
        const auto nAtoms        = simBox.getNumberOfAtoms();

        auto atomicNumbers_ = array_i(ssize_t(nAtoms), &atomicNumbers[0]);

        return atomicNumbers_;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}