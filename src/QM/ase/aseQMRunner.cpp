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

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <thread>

#include "box.hpp"         // for simulationBox::Periodicity
#include "constants.hpp"   // for _DEG_TO_RAD_
#include "physicalData.hpp"
#include "qmSettings.hpp"   // for QMSettings
#include "simulationBox.hpp"

using enum simulationBox::Periodicity;

using QM::AseQMRunner;
using namespace simulationBox;
using namespace physicalData;
using namespace constants;
using namespace settings;

using array_d = pybind11::array_t<double>;
using array_i = pybind11::array_t<int>;

namespace
{
    /**
     * @brief get the positions of the atoms in the ASE Atoms object
     *
     * @param simBox
     *
     * @return py::array
     *
     * @throw py::error_already_set if the construction of the array fails
     */
    [[nodiscard]]
    pybind11::array asePositions(const SimulationBox &simBox)
    {
        const auto nAtoms = simBox.getNumberOfQMAtoms();
        const auto pos    = simBox.getFlattenedQMPositions();

        const auto shape      = std::vector<size_t>{nAtoms, 3};
        const auto sizeDouble = sizeof(double);
        const auto strides    = std::vector<size_t>{sizeDouble * 3, sizeDouble};

        try
        {
            auto positions_array =
                array_d(static_cast<ssize_t>(nAtoms) * 3, &pos[0]);

            const auto positions_array_reshaped = pybind11::array(
                pybind11::buffer_info(
                    positions_array.mutable_data(),   // Pointer to data
                    sizeDouble,                       // Size of one scalar
                    pybind11::format_descriptor<double>::format(
                    ),        // Data type
                    2,        // Number of dimensions
                    shape,    // Shape (N, 3)
                    strides   // Strides
                )
            );

            return positions_array_reshaped;
        }
        catch (const pybind11::error_already_set &)
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
     * @return pybind11::array_t<double>
     *
     * @throw pybind11::error_already_set if the construction of the array fails
     */
    [[nodiscard]]
    pybind11::array_t<double> aseCell(const SimulationBox &simBox)
    {
        const auto boxDimension = simBox.getBoxDimensions();
        const auto boxAngles    = simBox.getBoxAngles();

        constexpr auto                   boxArraySize = 6;
        std::array<double, boxArraySize> box_array    = {
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
        catch (const pybind11::error_already_set &)
        {
            ::PyErr_Print();
            throw;
        }
    }

    /**
     * @brief get the periodic boundary conditions of the ASE Atoms object
     *
     * @return pybind11::array_t<bool>
     *
     * @throw pybind11::error_already_set if the construction of the array fails
     */
    [[nodiscard]]
    pybind11::array_t<bool> asePBC(simulationBox::Periodicity periodicity)
    {
        std::array<bool, 3> pbc_array{true, true, true};

        switch (periodicity)
        {
            case NON_PERIODIC: pbc_array = {false, false, false}; break;
            case X: pbc_array = {true, false, false}; break;
            case Y: pbc_array = {false, true, false}; break;
            case Z: pbc_array = {false, false, true}; break;
            case XY: pbc_array = {true, true, false}; break;
            case XZ: pbc_array = {true, false, true}; break;
            case YZ: pbc_array = {false, true, true}; break;
            case XYZ: pbc_array = {true, true, true}; break;
        }

        try
        {
            return pybind11::array_t<bool>(3, pbc_array.data());
        }
        catch (const pybind11::error_already_set &)
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
     * @return pybind11::array_t<int>
     *
     * @throw pybind11::error_already_set if the construction of the array fails
     */
    [[nodiscard]]
    pybind11::array_t<int> aseAtomicNumbers(const SimulationBox &simBox)
    {
        const auto atomicNumbers = simBox.getAtomicNumbers();
        const auto nAtoms        = simBox.getNumberOfAtoms();

        try
        {
            const auto atomicNumbers_ =
                array_i(static_cast<ssize_t>(nAtoms), &atomicNumbers[0]);

            return atomicNumbers_;
        }
        catch (const pybind11::error_already_set &)
        {
            ::PyErr_Print();
            throw;
        }
    }

}   // namespace

/**
 * @class AseQMRunner::AseInterface
 *
 * @brief PIMPL Interface to the ASE QM calculator
 */
struct __attribute__((visibility("default"))) AseQMRunner::AseInterface
{
    pybind11::object calculator;
    pybind11::object atomsModule;
    pybind11::object atoms;

    pybind11::array_t<double> forces;
    pybind11::array_t<double> stress;
};

/**
 * @brief Construct a new AseQMRunner::AseQMRunner object
 *
 * @throw pybind11::error_already_set if the import of the ase.atoms module
 * fails
 */
AseQMRunner::AseQMRunner() : _ase(std::make_unique<AseInterface>())
{
    try
    {
        const auto warningsModule = pybind11::module_::import("warnings");
        warningsModule.attr("filterwarnings")("ignore");

        const auto ioModule      = pybind11::module_::import("io");
        const auto sysModule     = pybind11::module_::import("sys");
        auto       old_stdout    = sysModule.attr("stdout");
        const auto mystdout      = ioModule.attr("StringIO")();
        sysModule.attr("stdout") = mystdout;

        _ase->atomsModule = pybind11::module_::import("ase.atoms");

        sysModule.attr("stdout") = old_stdout;
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

AseQMRunner::~AseQMRunner() = default;

/**
 * @brief run the ASE QM calculation
 *
 * @param simBox
 * @param physicalData
 *
 * @throw QMRunnerException if the calculation takes too long
 */
void AseQMRunner::run(
    SimulationBox &simBox,
    PhysicalData  &physicalData,
    Periodicity    per
)
{
    _periodicity = per;

    std::jthread timeoutThread{[this](const std::stop_token stopToken)
                               { throwAfterTimeout(stopToken); }};

    {
        auto _ = scoped("Build ASE Atoms");
        buildAseAtoms(simBox);
    }

    {
        auto _ = scoped("Execute ASE QM");
        execute();
    }

    {
        auto _ = scoped("Collect ASE Data");
        collectData(simBox, physicalData);
    }

    timeoutThread.request_stop();
}

/**
 * @brief execute the ASE QM calculation
 *
 * @param simBox
 *
 * @throw pybind11::error_already_set if the execution of the ASE QM calculation
 * fails
 */
void AseQMRunner::execute()
{
    try
    {
        _ase->atoms.attr("set_calculator")(_ase->calculator);

        const auto forces = _ase->atoms.attr("get_forces")();
        const auto energy = _ase->atoms.attr("get_potential_energy")();
        const auto stress =
            _ase->atoms.attr("get_stress")(pybind11::arg("voigt") = false);

        _ase->forces = forces.cast<array_d>();
        _energy      = energy.cast<double>();
        _ase->stress = stress.cast<array_d>();
    }
    catch (const pybind11::error_already_set &)
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
void AseQMRunner::collectData(
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
 * @throw pybind11::error_already_set if the collection of the forces fails
 */
void AseQMRunner::collectForces(SimulationBox &simBox) const
{
    const auto nAtoms = simBox.getNumberOfAtoms();

    try
    {
        const auto forces = _ase->forces.unchecked<2>();

        for (size_t i = 0; i < nAtoms; ++i)
        {
            simBox.getAtoms()[i]->setForce(
                {forces(i, 0) * EV_TO_KCAL_PER_MOL,
                 forces(i, 1) * EV_TO_KCAL_PER_MOL,
                 forces(i, 2) * EV_TO_KCAL_PER_MOL}
            );
        }
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }

    if (QMSettings::getRemoveNetForce())
        simBox.removeNetForce();
}

/**
 * @brief collect the energy from the ASE QM calculation
 *
 * @param physicalData
 */
void AseQMRunner::collectEnergy(PhysicalData &physicalData) const
{
    physicalData.setQMEnergy(_energy * EV_TO_KCAL_PER_MOL);
}

/**
 * @brief collect the stress from the ASE QM calculation
 *
 * @param simBox
 * @param physicalData
 *
 * @throw pybind11::error_already_set if the collection of the stress fails
 */
void AseQMRunner::collectStress(
    const SimulationBox &simBox,
    PhysicalData        &data
) const
{
    linearAlgebra::tensor3D stress_;

    try
    {
        const auto stress = _ase->stress.unchecked<2>();

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = 0; j < 3; ++j) stress_[i][j] = -stress(i, j);
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }

    stress_ = stress_ * EV_TO_KCAL_PER_MOL;

    const auto virial = stress_ * simBox.getVolume();

    data.setStressTensor(stress_);
    data.addVirial(virial);
}

/**
 * @brief build the ASE Atoms object
 *
 * @param simBox
 *
 * @throw pybind11::error_already_set if the construction of the Atoms object
 * fails
 */
void AseQMRunner::buildAseAtoms(const SimulationBox &simBox)
{
    try
    {
        const auto positions     = asePositions(simBox);
        const auto cell          = aseCell(simBox);
        const auto pbc           = asePBC(_periodicity);
        const auto atomicNumbers = aseAtomicNumbers(simBox);

        _ase->atoms = _ase->atomsModule.attr("Atoms")(
            pybind11::arg("positions") = positions,
            pybind11::arg("numbers")   = atomicNumbers,
            pybind11::arg("cell")      = cell,
            pybind11::arg("pbc")       = pbc
        );
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}

/**
 * @brief set the ASE calculator
 *
 * @param calculator
 */
void AseQMRunner::setAseCalculator(const pybind11::object &calculator)
{
    _ase->calculator = calculator;
}
