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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>

#include <chrono>
#include <memory>

#include "ase/aseFennolRunner.hpp"
#include "exceptions.hpp"
#include "orthorhombicBox.hpp"
#include "physicalData.hpp"
#include "qmSettings.hpp"
#include "simulationBox.hpp"

namespace
{
    // stand-ins for ase.atoms and fennol.ase: the force call takes
    // FENNIXCalculator.delay seconds
    constexpr auto fakeAseAtoms = R"py(
import time
import numpy as np

class Atoms:
    def __init__(self, positions, numbers, cell, pbc):
        self.n = len(numbers)

    def set_calculator(self, calculator):
        self.calculator = calculator

    def get_forces(self):
        time.sleep(self.calculator.delay)
        return np.zeros((self.n, 3))

    def get_potential_energy(self):
        return 0.0

    def get_stress(self, voigt):
        return np.zeros((3, 3))

class FENNIXCalculator:
    delay = 0.0

    def __init__(self, **kwargs):
        pass
)py";

    // pybind11 types are hidden, so they cannot be fixture members.
    // numpy cannot be re-imported after Py_Finalize: one interpreter per suite
    struct Python
    {
        static inline std::unique_ptr<pybind11::scoped_interpreter> guard;
    };

}   // namespace

class AseQMRunnerTest : public ::testing::Test
{
   protected:
    molsys::SimulationBox       _simulationBox;
    physicalData::PhysicalData  _physicalData;
    std::unique_ptr<QMSettings> _qmSettings;

   public:
    static void SetUpTestSuite()
    {
        Python::guard = std::make_unique<pybind11::scoped_interpreter>();
    }

    static void TearDownTestSuite() { Python::guard.reset(); }

   protected:
    void SetUp() override
    {
        _qmSettings      = std::make_unique<QMSettings>();
        const auto types = pybind11::module_::import("types");
        const auto sys   = pybind11::module_::import("sys");

        auto aseModule             = types.attr("ModuleType")("ase");
        auto aseAtoms              = types.attr("ModuleType")("ase.atoms");
        aseModule.attr("__path__") = pybind11::list();
        aseModule.attr("atoms")    = aseAtoms;

        pybind11::exec(fakeAseAtoms, aseAtoms.attr("__dict__"));

        auto fennolModule             = types.attr("ModuleType")("fennol");
        fennolModule.attr("__path__") = pybind11::list();
        fennolModule.attr("ase")      = aseAtoms;

        sys.attr("modules")["ase"]        = aseModule;
        sys.attr("modules")["ase.atoms"]  = aseAtoms;
        sys.attr("modules")["fennol"]     = fennolModule;
        sys.attr("modules")["fennol.ase"] = aseAtoms;

        constexpr auto          boxEdge = 10.0;
        molsys::OrthorhombicBox box{{boxEdge, boxEdge, boxEdge}};
        _simulationBox.setBox(box, false);
    }

    QM::AseFennolRunner _runner(const double delay)
    {
        pybind11::module_::import("sys")
            .attr("modules")["ase.atoms"]
            .attr("FENNIXCalculator")
            .attr("delay") = delay;

        return QM::AseFennolRunner{*_qmSettings, "model.fnx", false, true};
    }
};

TEST_F(AseQMRunnerTest, timeoutInterruptsPythonAndThrows)
{
    _qmSettings->setQMLoopTimeLimit(0.5);
    auto runner = _runner(10.0);

    const auto start = std::chrono::steady_clock::now();

    try
    {
        runner.run(_simulationBox, _physicalData, molsys::Periodicity::XYZ);
        FAIL() << "Expected the QM timeout to throw";
    }
    catch (const exc::QMRunnerException &error)
    {
        EXPECT_THAT(error.what(), testing::HasSubstr("timeout"));
    }

    EXPECT_LT(
        std::chrono::steady_clock::now() - start,
        std::chrono::seconds(5)
    );

    // the interpreter is usable again
    _qmSettings->setQMLoopTimeLimit(0.0);
    EXPECT_NO_THROW(_runner(0.0).run(
        _simulationBox,
        _physicalData,
        molsys::Periodicity::XYZ
    ));
}

TEST_F(AseQMRunnerTest, finishedStepWithinTimeLimitDoesNotThrow)
{
    _qmSettings->setQMLoopTimeLimit(10.0);
    auto runner = _runner(0.0);

    EXPECT_NO_THROW(
        runner.run(_simulationBox, _physicalData, molsys::Periodicity::XYZ)
    );
}
