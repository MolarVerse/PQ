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

#include "testQMSetupAse.hpp"   // for TestQMSetupAse

#include <gtest/gtest.h>   // for Test, TestInfo (ptr only), InitGoogleTest, RUN_ALL_TESTS

#include <string>   // for allocator, basic_string

#ifdef WITH_ASE
#include "aseFennolRunner.hpp"   // for AseFennolRunner
#include "aseMaceRunner.hpp"     // for AseMaceRunner
#include "pybind11/embed.h"      // for scoped_interpreter
#endif

#include "exceptions.hpp"         // for InputFileException
#include "gtest/gtest.h"          // for Message, TestPartResult
#include "qmRunner.hpp"           // for QMRunner
#include "throwWithMessage.hpp"   // for ASSERT_THROW_MSG

#ifdef WITH_ASE
TEST_F(TestQMSetupAse, setupAseDftbplus3OB)
{
    QMSettings::setSlakosType("3ob");
    QMSettings::setUseThirdOrderDftb(true);
    QMSettings::setUseDispersionCorrection(true);
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        3ob");
    getline(file, line);
    // clang-format off
    std::string skPath {__SLAKOS_DIR__ + string(QMSettings::getSlakosType()) + "/skfiles/"};
    EXPECT_EQ(line, "         sk file path:         " + skPath);
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: on");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  on");
}

TEST_F(TestQMSetupAse, setupAseDftbplus3OBno3rdOrder)
{
    QMSettings::setUseThirdOrderDftb(false);
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setSlakosType("3ob");
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        3ob");
    getline(file, line);
    // clang-format off
    std::string skPath {__SLAKOS_DIR__ + string(QMSettings::getSlakosType()) + "/skfiles/"};
    EXPECT_EQ(line, "         sk file path:         " + skPath);
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: off");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  off");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    // clang-format off
    EXPECT_EQ(line, "WARNING: 3ob approach has been chosen while disabling 3rd order DFTB. This setup is not recommended.");
    // clang-format on
}

TEST_F(TestQMSetupAse, setupAseDftbplus3OBCustomHubbardDerivs)
{
    QMSettings::setSlakosType("3ob");
    QMSettings::setUseThirdOrderDftb(true);
    QMSettings::setHubbardDerivs({{"H", -0.3}});
    QMSettings::setIsHubbardDerivsSet(true);
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        3ob");
    getline(file, line);
    // clang-format off
    std::string skPath {__SLAKOS_DIR__ + string(QMSettings::getSlakosType()) + "/skfiles/"};
    EXPECT_EQ(line, "         sk file path:         " + skPath);
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: off");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  on");
    getline(file, line);
    // clang-format off
    EXPECT_EQ(line, "         Hubbard derivatives:  H: -0.3");
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    // clang-format off
    EXPECT_EQ(line, "WARNING: 3ob approach has been chosen while setting custom Hubbard derivatives. This setup is not recommended.");
    // clang-format on
}

TEST_F(TestQMSetupAse, setupAseDftbplusMatsci)
{
    QMSettings::setSlakosType("matsci");
    QMSettings::setUseDispersionCorrection(true);
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        matsci");
    getline(file, line);
    // clang-format off
    std::string skPath {__SLAKOS_DIR__ + string(QMSettings::getSlakosType()) + "/skfiles/"};
    EXPECT_EQ(line, "         sk file path:         " + skPath);
    // clang-format on
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: on");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  off");
}
#endif

TEST_F(TestQMSetupAse, setupAseDftbplusCustom)
{
    QMSettings::setSlakosType("custom");
    QMSettings::setSlakosPath("custom/path/");
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        custom");
    getline(file, line);
    EXPECT_EQ(line, "         sk file path:         custom/path/");
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: off");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  off");
}

TEST_F(TestQMSetupAse, setupAseDftbplusCustom3rdOrder)
{
    QMSettings::setSlakosType("custom");
    QMSettings::setSlakosPath("custom/path/");
    QMSettings::setUseDispersionCorrection(true);
    QMSettings::setUseThirdOrderDftb(true);
    QMSettings::setUseThirdOrderDftb(true);
    QMSettings::setIsThirdOrderDftbSet(true);
    QMSettings::setHubbardDerivs({{"H", -0.3}});
    QMSettings::setIsHubbardDerivsSet(true);
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    getline(file, line);
    EXPECT_EQ(line, "         QM runner: ASEDFTBPLUS");
    getline(file, line);
    EXPECT_EQ(line, "");
    getline(file, line);
    EXPECT_EQ(line, "         DFTB approach:        custom");
    getline(file, line);
    EXPECT_EQ(line, "         sk file path:         custom/path/");
    getline(file, line);
    EXPECT_EQ(line, "         Dispersion is turned: on");
    getline(file, line);
    EXPECT_EQ(line, "         3rd order is turned:  on");
    getline(file, line);
    EXPECT_EQ(line, "         Hubbard derivatives:  H: -0.3");
}

#ifdef WITH_ASE
TEST_F(TestQMSetupAse, setupAseFennolRunnerConstructsCalculator)
{
    pybind11::scoped_interpreter guard{};
    const auto                   types = pybind11::module_::import("types");
    const auto                   sys   = pybind11::module_::import("sys");
    const auto                   modules = sys.attr("modules");

    auto aseModule       = types.attr("ModuleType")("ase");
    auto aseAtomsModule  = types.attr("ModuleType")("ase.atoms");
    auto fennolModule    = types.attr("ModuleType")("fennol");
    auto fennolAseModule = types.attr("ModuleType")("fennol.ase");

    aseModule.attr("__path__")    = pybind11::list();
    fennolModule.attr("__path__") = pybind11::list();
    aseModule.attr("atoms")       = aseAtomsModule;
    fennolModule.attr("ase")      = fennolAseModule;

    pybind11::exec(
        R"py(
class FENNIXCalculator:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
)py",
        pybind11::globals(),
        fennolAseModule.attr("__dict__")
    );

    modules["ase"]        = aseModule;
    modules["ase.atoms"]  = aseAtomsModule;
    modules["fennol"]     = fennolModule;
    modules["fennol.ase"] = fennolAseModule;

    ASSERT_NO_THROW({ QM::AseFennolRunner runner("model.fnx", false, true); });

    auto lastKwargs = fennolAseModule.attr("FENNIXCalculator").attr("last_kwargs")
                          .cast<pybind11::dict>();

    EXPECT_EQ(lastKwargs["model"].cast<std::string>(), "model.fnx");
    EXPECT_EQ(lastKwargs["gpu_preprocessing"].cast<bool>(), false);
    EXPECT_EQ(lastKwargs["use_float64"].cast<bool>(), true);

    modules.attr("pop")("fennol.ase", pybind11::none());
    modules.attr("pop")("fennol", pybind11::none());
    modules.attr("pop")("ase.atoms", pybind11::none());
    modules.attr("pop")("ase", pybind11::none());
}
#endif

#ifdef WITH_ASE
TEST_F(TestQMSetupAse, setupAseMaceRunnerConstructsCalculator)
{
    pybind11::scoped_interpreter guard{};
    const auto                   types   = pybind11::module_::import("types");
    const auto                   sys     = pybind11::module_::import("sys");
    const auto                   modules = sys.attr("modules");

    auto aseModule         = types.attr("ModuleType")("ase");
    auto aseAtomsModule    = types.attr("ModuleType")("ase.atoms");
    auto maceModule        = types.attr("ModuleType")("mace");
    auto calculatorsModule = types.attr("ModuleType")("mace.calculators");

    aseModule.attr("__path__")     = pybind11::list();
    maceModule.attr("__path__")    = pybind11::list();
    aseModule.attr("atoms")        = aseAtomsModule;
    maceModule.attr("calculators") = calculatorsModule;

    pybind11::exec(
        R"py(
class MACECalculator:
    last_kwargs = None

    def __init__(self, **kwargs):
        type(self).last_kwargs = kwargs
)py",
        pybind11::globals(),
        calculatorsModule.attr("__dict__")
    );

    modules["ase"]              = aseModule;
    modules["ase.atoms"]        = aseAtomsModule;
    modules["mace"]             = maceModule;
    modules["mace.calculators"] = calculatorsModule;

    ASSERT_NO_THROW({
        QM::AseMaceRunner runner(
            "MACECalculator", "model.model", "float64", true, false
        );
    });

    auto lastKwargs = calculatorsModule.attr("MACECalculator").attr("last_kwargs")
                          .cast<pybind11::dict>();

    EXPECT_EQ(lastKwargs["model"].cast<std::string>(), "model.model");
    EXPECT_EQ(lastKwargs["dispersion"].cast<bool>(), true);
    EXPECT_EQ(lastKwargs["enable_cueq"].cast<bool>(), false);
    EXPECT_EQ(lastKwargs["default_dtype"].cast<std::string>(), "float64");
    EXPECT_EQ(lastKwargs["device"].cast<std::string>(), "cuda");

    modules.attr("pop")("mace.calculators", pybind11::none());
    modules.attr("pop")("mace", pybind11::none());
    modules.attr("pop")("ase.atoms", pybind11::none());
    modules.attr("pop")("ase", pybind11::none());
}

TEST_F(TestQMSetupAse, setupAseMaceWriteInfoFast)
{
    QMSettings::setQMMethod(QMMethod::MACE);
    QMSettings::setMaceMode("fast");
    _qmSetup->setupWriteInfo();

    std::ifstream file("default.log");
    std::string   line;
    std::string   all;
    while (std::getline(file, line)) all += line + "\n";

    EXPECT_NE(all.find("Evaluation mode:       fast"), std::string::npos);
    EXPECT_NE(all.find("cuequivariance-accelerated"), std::string::npos);
}
#endif
