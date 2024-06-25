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

#include "maceRunner.hpp"

#include "physicalData.hpp"
#include "pybind11/embed.h"
#include "simulationBox.hpp"

using QM::MaceRunner;
using std::vector;
namespace py = pybind11;

using array_d = py::array_t<double>;
using array_i = py::array_t<int>;

namespace
{
    const py::scoped_interpreter guard{};
}

MaceRunner::MaceRunner(const std::string &model, const std::string &fpType)
{
    try
    {
        const py::module_ mace        = py::module_::import("mace");
        const py::module_ calculators = py::module_::import("mace.calculators");
        _atoms_module                 = py::module_::import("ase.atoms");

        py::dict calculatorArgs;
        calculatorArgs["model"]         = model.c_str();
        calculatorArgs["dispersion"]    = pybind11::bool_(false);
        calculatorArgs["default_dtype"] = fpType.c_str();
        calculatorArgs["device"]        = pybind11::str("cuda");

        _calculator = calculators.attr("mace_mp")(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

void MaceRunner::execute(simulationBox::SimulationBox &simBox)
{
    startTimingsSection("MACE prepare");

    const auto nAtoms = simBox.getNumberOfAtoms();

    const auto pos             = simBox.flattenPositions();
    array_d    positions_array = array_d(ssize_t(nAtoms), &pos[0]);

    const auto shape = std::vector<size_t>{nAtoms, 3};
    const auto strides =
        std::vector<size_t>{sizeof(double) * 3, sizeof(double)};

    auto positions_array_reshaped = pybind11::array(py::buffer_info(
        positions_array.mutable_data(),            // Pointer to data
        sizeof(double),                            // Size of one scalar
        py::format_descriptor<double>::format(),   // Data type
        2,                                         // Number of dimensions
        shape,                                     // Shape (N, 3)
        strides                                    // Strides
    ));

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

    std::array<bool, 3> pbc_array  = {true, true, true};
    auto                pbc_array_ = py::array_t<bool>(3, &pbc_array[0]);

    std::vector<int> atomic_numbers = simBox.getAtomicNumbers();
    auto atomic_numbers_array = array_i(ssize_t(nAtoms), &atomic_numbers[0]);

    py::dict kwargs;
    kwargs["positions"] = positions_array_reshaped;
    kwargs["cell"]      = box_array_;
    kwargs["pbc"]       = pbc_array_;
    kwargs["numbers"]   = atomic_numbers_array;

    py::object atoms = _atoms_module.attr("Atoms")(**kwargs);
    atoms.attr("set_calculator")(_calculator);

    stopTimingsSection("MACE prepare");
    startTimingsSection("MACE get forces");
    _forces = atoms.attr("get_forces")().cast<array_d>();
    stopTimingsSection("MACE get forces");

    startTimingsSection("MACE get potential energy");
    _energy = atoms.attr("get_potential_energy")().cast<double>();
    stopTimingsSection("MACE get potential energy");

    py::dict stress_dict;
    stress_dict["voigt"] = pybind11::bool_(false);

    startTimingsSection("MACE get stress tensor");
    _stress_tensor = atoms.attr("get_stress")(stress_dict).cast<array_d>();
    stopTimingsSection("MACE get stress tensor");
}

void MaceRunner::collectData(
    simulationBox::SimulationBox &simBox,
    physicalData::PhysicalData   &physicalData
)
{
    startTimingsSection("MACE collect data");
    const auto forces = _forces.unchecked<2>();
    for (size_t i = 0; i < forces.shape(0); ++i)
        simBox.getAtoms()[i]->setForce(
            {forces(i, 0) * constants::_EV_TO_KCAL_PER_MOL_,
             forces(i, 1) * constants::_EV_TO_KCAL_PER_MOL_,
             forces(i, 2) * constants::_EV_TO_KCAL_PER_MOL_}
        );

    physicalData.setQMEnergy(_energy * constants::_EV_TO_KCAL_PER_MOL_);

    const auto              stress_tensor = _stress_tensor.unchecked<1>();
    linearAlgebra::tensor3D stress_tensor_;

    for (size_t i = 0; i < 3; ++i)
        for (size_t j = 0; j < 3; ++j)
            stress_tensor_[j][i] = stress_tensor[i * 3 + j];

    physicalData.setStressTensor(
        stress_tensor_ * constants::_EV_TO_KCAL_PER_MOL_
    );

    const auto virial = stress_tensor_ * simBox.getVolume();

    physicalData.addVirial(virial);
    stopTimingsSection("MACE collect data");
}
