#include "maceRunner.hpp"

#include "physicalData.hpp"
#include "pybind11/embed.h"
#include "simulationBox.hpp"

using QM::MaceRunner;
using std::vector;
namespace py = pybind11;

namespace
{
    const py::scoped_interpreter guard{};
}

MaceRunner::MaceRunner(const std::string &model)
{
    try
    {
        py::module_ mace        = py::module_::import("mace");
        py::module_ calculators = py::module_::import("mace.calculators");
        _atoms_module           = py::module_::import("ase.atoms");

        py::dict kwargs;
        kwargs["model"]         = "medium";
        kwargs["dispersion"]    = pybind11::bool_(false);
        kwargs["default_dtype"] = pybind11::str("float32");
        kwargs["device"]        = pybind11::str("cuda");

        _calculator = calculators.attr("mace_mp")(**kwargs);
    }
    catch (const py::error_already_set &e)
    {
        PyErr_Print();
        throw;
    }
}

void MaceRunner::execute(simulationBox::SimulationBox &simBox)
{
    const auto          positions = simBox.getPositions();
    std::vector<double> pos;
    for (const auto &p : positions)
    {
        pos.push_back(p[0]);
        pos.push_back(p[1]);
        pos.push_back(p[2]);
    }

    py::array_t<double> positions_array =
        py::array_t<double>(pos.size(), &pos[0]);

    auto shape =
        std::vector<size_t>{simBox.getNumberOfAtoms(), 3};   // Shape (N, 3)

    auto strides = std::vector<size_t>{
        sizeof(double) * 3,
        sizeof(double)
    };   // Strides (row-major)

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
    std::array<double, 9> box_array;

    box_array[0] = boxDimension[0];
    box_array[1] = boxDimension[1];
    box_array[2] = boxDimension[2];
    box_array[3] = boxAngles[0];
    box_array[4] = boxAngles[1];
    box_array[5] = boxAngles[2];

    py::array_t<double> box_array_ = py::array_t<double>(6, &box_array[0]);
    std::array<bool, 3> pbc_array  = {true, true, true};
    py::array_t<bool>   pbc_array_ = py::array_t<bool>(3, &pbc_array[0]);

    std::vector<int> atomic_numbers;
    for (const auto &atom : simBox.getAtoms())
        atomic_numbers.push_back(atom->getAtomicNumber());

    py::array_t<int> atomic_numbers_array =
        py::array_t<int>(atomic_numbers.size(), &atomic_numbers[0]);

    py::dict kwargs;
    kwargs["positions"] = positions_array_reshaped;
    kwargs["cell"]      = box_array_;
    kwargs["pbc"]       = pbc_array_;
    kwargs["numbers"]   = atomic_numbers_array;

    py::object atoms = _atoms_module.attr("Atoms")(**kwargs);
    atoms.attr("set_calculator")(_calculator);

    _forces = atoms.attr("get_forces")().cast<py::array_t<double>>();
    _energy = atoms.attr("get_potential_energy")().cast<double>();

    py::dict stress_dict;
    stress_dict["voigt"] = pybind11::bool_(false);

    _stress_tensor =
        atoms.attr("get_stress")(stress_dict).cast<py::array_t<double>>();
}

void MaceRunner::collectData(
    simulationBox::SimulationBox &simBox,
    physicalData::PhysicalData   &physicalData
)
{
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
}
