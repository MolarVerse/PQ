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

#include "box.hpp"   // for simulationBox::Periodicity
#include "physicalData.hpp"
#include "pybind11/embed.h"
#include "simulationBox.hpp"

using QM::ASEQMRunner;
using enum simulationBox::Periodicity;

using namespace simulationBox;
using namespace physicalData;
using namespace constants;

using array_d = py::array_t<double>;
using array_i = py::array_t<int>;

/**
 * @brief Construct a new ASEQMRunner::ASEQMRunner object
 *
 * @throw py::error_already_set if the import of the ase.atoms module fails
 */
ASEQMRunner::ASEQMRunner()
{
    try
    {
        const auto warningsModule = py::module_::import("warnings");
        warningsModule.attr("filterwarnings")("ignore");

        const auto ioModule      = py::module_::import("io");
        const auto sysModule     = py::module_::import("sys");
        auto       old_stdout    = sysModule.attr("stdout");
        const auto mystdout      = ioModule.attr("StringIO")();
        sysModule.attr("stdout") = mystdout;

        _atomsModule = py::module_::import("ase.atoms");

        sysModule.attr("stdout") = old_stdout;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief run the ASE QM calculation
 *
 * @param simBox
 * @param physicalData
 *
 * @throw QMRunnerException if the calculation takes too long
 */
void ASEQMRunner::run(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    Periodicity    per
)
{
    _periodicity = per;

    std::jthread timeoutThread{[this](const std::stop_token stopToken)
                               { throwAfterTimeout(stopToken); }};

    startTimingsSection("Build ASE Atoms");
    buildAseAtoms(simBox);
    stopTimingsSection("Build ASE Atoms");

    startTimingsSection("Execute ASE QM");
    execute();
    stopTimingsSection("Execute ASE QM");

    startTimingsSection("Collect ASE Data");
    collectData(simBox, physicalData);
    stopTimingsSection("Collect ASE Data");

    timeoutThread.request_stop();
}

/**
 * @brief execute the ASE QM calculation
 *
 * @param simBox
 *
 * @throw py::error_already_set if the execution of the ASE QM calculation fails
 */
void ASEQMRunner::execute()
{
    try
    {
        _atoms.attr("set_calculator")(_calculator);

        const auto forces = _atoms.attr("get_forces")();
        const auto energy = _atoms.attr("get_potential_energy")();
        const auto stress = _atoms.attr("get_stress")(py::arg("voigt") = false);

        _forces = forces.cast<array_d>();
        _energy = energy.cast<double>();
        _stress = stress.cast<array_d>();
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief collect the data from the ASE QM calculation
 *
 * @param simBox
 * @param physicalData
 */
void ASEQMRunner::collectData(
    SimulationBox &simBox,
    PhysicalData  &physicalData
) const
{
    collectForces(simBox);
    collectEnergy(physicalData);
    collectStress(simBox, physicalData);
}

/**
 * @brief collect the forces from the ASE QM calculation
 *
 * @param simBox
 *
 * @throw py::error_already_set if the collection of the forces fails
 */
void ASEQMRunner::collectForces(SimulationBox &simBox) const
{
    try
    {
        const auto forces = _forces.unchecked<2>();

        size_t i = 0;
        for (const auto &atom : simBox.getQMAtoms())
        {
            atom->setForce(
                {forces(i, 0) * _EV_TO_KCAL_PER_MOL_,
                 forces(i, 1) * _EV_TO_KCAL_PER_MOL_,
                 forces(i, 2) * _EV_TO_KCAL_PER_MOL_}
            );
            ++i;
        }
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief collect the energy from the ASE QM calculation
 *
 * @param physicalData
 */
void ASEQMRunner::collectEnergy(PhysicalData &physicalData) const
{
    physicalData.setQMEnergy(_energy * _EV_TO_KCAL_PER_MOL_);
}

/**
 * @brief collect the stress from the ASE QM calculation
 *
 * @param simBox
 * @param physicalData
 *
 * @throw py::error_already_set if the collection of the stress fails
 */
void ASEQMRunner::collectStress(
    const SimulationBox &simBox,
    PhysicalData        &data
) const
{
    linearAlgebra::tensor3D stress_;

    try
    {
        const auto stress = _stress.unchecked<2>();

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j) stress_[i][j] = -stress(i, j);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }

    stress_ = stress_ * _EV_TO_KCAL_PER_MOL_;

    const auto virial = stress_ * simBox.getVolume();

    data.setStressTensor(stress_);
    data.addVirial(virial);
}

/**
 * @brief build the ASE Atoms object
 *
 * @param simBox
 *
 * @throw py::error_already_set if the construction of the Atoms object fails
 */
void ASEQMRunner::buildAseAtoms(const SimulationBox &simBox)
{
    try
    {
        const auto positions     = asePositions(simBox);
        const auto cell          = aseCell(simBox);
        const auto pbc           = asePBC();
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

/**
 * @brief get the positions of the atoms in the ASE Atoms object
 *
 * @param simBox
 *
 * @return py::array
 *
 * @throw py::error_already_set if the construction of the array fails
 */
py::array ASEQMRunner::asePositions(const SimulationBox &simBox) const
{
    const auto nAtoms = simBox.getNumberOfQMAtoms();
    const auto pos    = simBox.getFlattenedQMPositions();

    const auto shape      = std::vector<size_t>{nAtoms, 3};
    const auto sizeDouble = sizeof(double);
    const auto strides    = std::vector<size_t>{sizeDouble * 3, sizeDouble};

    try
    {
        auto positions_array = array_d(ssize_t(nAtoms) * 3, &pos[0]);

        const auto positions_array_reshaped = py::array(
            py::buffer_info(
                positions_array.mutable_data(),            // Pointer to data
                sizeDouble,                                // Size of one scalar
                py::format_descriptor<double>::format(),   // Data type
                2,        // Number of dimensions
                shape,    // Shape (N, 3)
                strides   // Strides
            )
        );

        return positions_array_reshaped;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief get the cell of the ASE Atoms object
 *
 * @param simBox
 *
 * @return py::array_t<double>
 *
 * @throw py::error_already_set if the construction of the array fails
 */
py::array_t<double> ASEQMRunner::aseCell(const SimulationBox &simBox) const
{
    const auto boxDimension = simBox.getBoxDimensions();
    const auto boxAngles    = simBox.getBoxAngles();

    std::array<double, 6> box_array = {
        boxDimension[0],
        boxDimension[1],
        boxDimension[2],
        boxAngles[0],
        boxAngles[1],
        boxAngles[2]
    };

    try
    {
        const auto box_array_ = array_d(6, &box_array[0]);

        return box_array_;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief get the periodic boundary conditions of the ASE Atoms object
 *
 * @return py::array_t<bool>
 *
 * @throw py::error_already_set if the construction of the array fails
 */
py::array_t<bool> ASEQMRunner::asePBC() const
{
    std::array<bool, 3> pbc_array;

    switch (_periodicity)
    {
        case NON_PERIODIC: pbc_array = {false, false, false}; break;
        case X: pbc_array = {true, false, false}; break;
        case Y: pbc_array = {false, true, false}; break;
        case Z: pbc_array = {false, false, true}; break;
        case XY: pbc_array = {true, true, false}; break;
        case XZ: pbc_array = {true, false, true}; break;
        case YZ: pbc_array = {false, true, true}; break;
        case XYZ: pbc_array = {true, true, true}; break;
        default: pbc_array = {false, false, false}; break;
    }

    try
    {
        return py::array_t<bool>(3, pbc_array.data());
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief get the atomic numbers of the atoms in the ASE Atoms object
 *
 * @param simBox
 *
 * @return py::array_t<int>
 *
 * @throw py::error_already_set if the construction of the array fails
 */
py::array_t<int> ASEQMRunner::aseAtomicNumbers(
    const SimulationBox &simBox
) const
{
    std::vector<int> atomicNumbers;
    atomicNumbers.reserve(simBox.getNumberOfQMAtoms());

    for (const auto atomicNumber : simBox.getQMAtomicNumbers())
        atomicNumbers.emplace_back(atomicNumber);

    try
    {
        const auto atomicNumbers_ =
            array_i(atomicNumbers.size(), atomicNumbers.data());

        return atomicNumbers_;
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}