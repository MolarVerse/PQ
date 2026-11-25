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

#ifdef WITH_PYTHON_BINDINGS

#include <Python.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <format>
#include <fstream>
#include <string>
#include <vector>

#include "exceptions.hpp"
#include "selection.hpp"
#include "throwWithMessage.hpp"

TEST(TestPythonSelection, selectsAtomsFromRestartFile)
{
    const auto restartFile =
        std::filesystem::temp_directory_path() / "pqSelection.rst";

    {
        std::ofstream file{restartFile};
        file << "Box 10.0 10.0 10.0\n";
        file << "O 1 1 0.0 0.0 0.0\n";
        file << "H 2 1 1.0 0.0 0.0\n";
        file << "H 3 1 0.0 1.0 0.0\n";
        file << "Na 4 2 5.0 5.0 5.0\n";
    }

    EXPECT_EQ(
        pq_python::select("O", restartFile.string(), ""),
        (std::vector<int>{0})
    );
    EXPECT_EQ(
        pq_python::select("elem(H), Na", restartFile.string(), ""),
        (std::vector<int>{1, 2, 3})
    );

    std::filesystem::remove(restartFile);
}

namespace
{
    void blockImport(const std::string &blocked, const std::string &error)
    {
        if (::Py_IsInitialized() == 0)
        {
            ::Py_Initialize();
        }

        const auto code = std::format(
            "import sys\n"
            "_pqRoots = ('PQAnalysis', '{0}')\n"
            "class _PQBlocker:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name.split('.')[0] == '{0}':\n"
            "            raise {1}\n"
            "_pqBlocker = _PQBlocker()\n"
            "_pqSavedModules = {{name: module for name, module in "
            "sys.modules.items() if name.split('.')[0] in _pqRoots}}\n"
            "for _pqName in _pqSavedModules:\n"
            "    del sys.modules[_pqName]\n"
            "sys.meta_path.insert(0, _pqBlocker)\n",
            blocked,
            error
        );
        ASSERT_EQ(::PyRun_SimpleString(code.c_str()), 0);
    }

    void unblockImport()
    {
        ASSERT_EQ(
            ::PyRun_SimpleString(
                "sys.meta_path.remove(_pqBlocker)\n"
                "for _pqName in [name for name in sys.modules "
                "if name.split('.')[0] in _pqRoots]:\n"
                "    del sys.modules[_pqName]\n"
                "sys.modules.update(_pqSavedModules)\n"
            ),
            0
        );
    }
}   // namespace

TEST(TestPythonSelection, missingPQAnalysisThrows)
{
    blockImport(
        "PQAnalysis",
        "ModuleNotFoundError(f\"No module named '{name}'\", name=name)"
    );

    EXPECT_THROW_MSG(
        pq_python::select("O", "missing.rst", ""),
        exc::InputFileException,
        "Selection \"O\" needs the Python package PQAnalysis, which is not "
        "installed (ModuleNotFoundError: No module named 'PQAnalysis'). "
        "Install it for the Python that PQ was built with: "
        "pip install PQAnalysis"
    );

    unblockImport();
}

TEST(TestPythonSelection, missingDependencyThrows)
{
    blockImport(
        "lark",
        "ModuleNotFoundError(f\"No module named '{name}'\", name=name)"
    );

    EXPECT_THROW_MSG(
        pq_python::select("O", "missing.rst", ""),
        exc::InputFileException,
        "Selection \"O\" could not import \"PQAnalysis.topology.api\" "
        "because the Python module \"lark\" was not found "
        "(ModuleNotFoundError: No module named 'lark')"
    );

    unblockImport();
}

TEST(TestPythonSelection, failingImportThrows)
{
    blockImport("PQAnalysis", "ImportError('broken by test')");

    EXPECT_THROW_MSG(
        pq_python::select("O", "missing.rst", ""),
        exc::InputFileException,
        "Selection \"O\" could not import \"PQAnalysis.topology.api\" "
        "(ImportError: broken by test)"
    );

    unblockImport();
}

#endif   // WITH_PYTHON_BINDINGS
