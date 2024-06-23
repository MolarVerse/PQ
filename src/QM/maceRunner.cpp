#include "maceRunner.hpp"

#include "physicalData.hpp"
#include "simulationBox.hpp"

using QM::MaceRunner;
using namespace pybind11;

MaceRunner::MaceRunner()
{
    initialize_interpreter();

    module_ mace        = module_::import("mace");
    module_ calculators = module_::import("mace.calculators");
    _atoms_module       = module_::import("ase.atoms");

    dict kwargs;
    kwargs["mode"]         = "large";
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

    const auto            box = simBox.getBox();
    std::array<double, 9> box_array;

    box_array[0] = box.getBoxDimensions()[0];
    box_array[1] = box.getBoxDimensions()[1];
    box_array[2] = box.getBoxDimensions()[2];
    box_array[3] = box.getBoxAngles()[0];
    box_array[4] = box.getBoxAngles()[1];
    box_array[5] = box.getBoxAngles()[2];

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
