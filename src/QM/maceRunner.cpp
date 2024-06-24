#include "maceRunner.hpp"

#include "physicalData.hpp"
#include "pybind11/embed.h"
#include "simulationBox.hpp"

using QM::MaceRunner;
using namespace pybind11;

namespace
{
    const scoped_interpreter guard{};
}

MaceRunner::MaceRunner(const std::string &model)
{
    module_ mace        = module_::import("mace");
    module_ calculators = module_::import("mace.calculators");
    _atoms_module       = module_::import("ase.atoms");

    dict kwargs;
    kwargs["model"]        = model.c_str();
    kwargs["dispersion"]   = false;
    kwargs["default_type"] = "float32";
    kwargs["device"]       = "cuda";

    _calculator = calculators.attr("mace_mp")(**kwargs);
}

void MaceRunner::prepareAtoms(simulationBox::SimulationBox &simBox)
{
    const auto                         positions = simBox.getPositions();
    std::vector<std::array<double, 3>> pos;
    for (const auto &p : positions)
    {
        pos.push_back({p[0], p[1], p[2]});
    }

    array_t<double> positions_array =
        array_t<double>(pos.size() * 3, &pos[0][0]);

    const auto            boxDimension = simBox.getBoxDimensions();
    const auto            boxAngles    = simBox.getBoxAngles();
    std::array<double, 9> box_array;

    box_array[0] = boxDimension[0];
    box_array[1] = boxDimension[1];
    box_array[2] = boxDimension[2];
    box_array[3] = boxAngles[0];
    box_array[4] = boxAngles[1];
    box_array[5] = boxAngles[2];

    array_t<double>     box_array_ = array_t<double>(9, &box_array[0]);
    std::array<bool, 3> pbc_array  = {true, true, true};
    array_t<bool>       pbc_array_ = array_t<bool>(3, &pbc_array[0]);

    std::vector<int> atomic_numbers;
    for (const auto &atom : simBox.getAtoms())
    {
        atomic_numbers.push_back(atom->getAtomicNumber());
    }

    array_t<int> atomic_numbers_array =
        array_t<int>(atomic_numbers.size(), &atomic_numbers[0]);

    dict kwargs;
    kwargs["positions"] = positions_array;
    kwargs["cell"]      = box_array_;
    kwargs["pbc"]       = pbc_array_;
    kwargs["numbers"]   = atomic_numbers_array;

    object atoms = _atoms_module.attr("Atoms")(**kwargs);
    atoms.attr("set_calculator")(_calculator);

    _forces        = atoms.attr("get_forces")().cast<array_t<double>>();
    _energy        = atoms.attr("get_potential_energy")().cast<double>();
    _stress_tensor = atoms.attr("get_stress")().cast<array_t<double>>();
}

void MaceRunner::execute()
{
    // nothing to do here
}

void MaceRunner::collectData(
    simulationBox::SimulationBox &simBox,
    physicalData::PhysicalData   &physicalData
)
{
    const auto forces = _forces.unchecked<2>();
    for (size_t i = 0; i < forces.shape(0); ++i)
    {
        simBox.getAtoms()[i]->setForce(
            {forces(i, 0), forces(i, 1), forces(i, 2)}
        );
    }

    physicalData.setQMEnergy(_energy);

    const auto              stress_tensor = _stress_tensor.unchecked<2>();
    linearAlgebra::tensor3D stress_tensor_;

    for (size_t i = 0; i < stress_tensor.shape(0); ++i)
    {
        for (size_t j = 0; j < stress_tensor.shape(1); ++j)
        {
            stress_tensor_[j][i] = stress_tensor(i, j);
        }
    }

    physicalData.setStressTensor(stress_tensor_);
}
