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
    startTimingsSection("MACE prepare");
    const auto pos             = simBox.flattenPositions();
    array_d    positions_array = array_d(pos.size(), &pos[0]);

    const auto nAtoms = simBox.getNumberOfAtoms();

    auto shape   = std::vector<size_t>{nAtoms, 3};
    auto strides = std::vector<size_t>{sizeof(double) * 3, sizeof(double)};

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

    array_d             box_array_ = array_d(6, &box_array[0]);
    std::array<bool, 3> pbc_array  = {true, true, true};
    py::array_t<bool>   pbc_array_ = py::array_t<bool>(3, &pbc_array[0]);

    std::vector<int> atomic_numbers;
    for (const auto &atom : simBox.getAtoms())
        atomic_numbers.push_back(atom->getAtomicNumber());

    array_i atomic_numbers_array = array_i(nAtoms, &atomic_numbers[0]);

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
