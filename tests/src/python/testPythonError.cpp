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

#include "pythonError.hpp"

TEST(TestPythonError, importErrorMessageForMissingPQAnalysis)
{
    EXPECT_EQ(
        pq_python::error::importErrorMessage(
            "O",
            "PQAnalysis.topology.api",
            {.description   = "ModuleNotFoundError: No module named 'x'",
             .missingModule = "PQAnalysis"}
        ),
        "Selection \"O\" needs the Python package PQAnalysis, which is not "
        "installed (ModuleNotFoundError: No module named 'x'). Install it "
        "for the Python that PQ was built with: pip install PQAnalysis"
    );
}

TEST(TestPythonError, importErrorMessageForMissingDependency)
{
    EXPECT_EQ(
        pq_python::error::importErrorMessage(
            "O",
            "PQAnalysis.topology.api",
            {.description = "ModuleNotFoundError", .missingModule = "lark"}
        ),
        "Selection \"O\" could not import \"PQAnalysis.topology.api\" because "
        "the Python module \"lark\" was not found (ModuleNotFoundError)"
    );
}

TEST(TestPythonError, importErrorMessageForOtherError)
{
    EXPECT_EQ(
        pq_python::error::importErrorMessage(
            "O",
            "PQAnalysis.topology.api",
            {.description = "ImportError: broken", .missingModule = ""}
        ),
        "Selection \"O\" could not import \"PQAnalysis.topology.api\" "
        "(ImportError: broken)"
    );
}

TEST(TestPythonError, takePythonError)
{
    if (::Py_IsInitialized() == 0)
    {
        ::Py_Initialize();
    }

    const auto noError = pq_python::error::takePythonError();
    EXPECT_EQ(noError.description, "no Python exception was set");
    EXPECT_EQ(noError.missingModule, "");

    ::PyErr_SetString(::PyExc_ValueError, "bad value");
    const auto valueError = pq_python::error::takePythonError();
    EXPECT_EQ(valueError.description, "ValueError: bad value");
    EXPECT_EQ(valueError.missingModule, "");
    EXPECT_EQ(::PyErr_Occurred(), nullptr);

    EXPECT_EQ(::PyImport_ImportModule("pqMissingTestModule"), nullptr);
    const auto importError = pq_python::error::takePythonError();
    EXPECT_EQ(
        importError.description,
        "ModuleNotFoundError: No module named 'pqMissingTestModule'"
    );
    EXPECT_EQ(importError.missingModule, "pqMissingTestModule");
}

#endif   // WITH_PYTHON_BINDINGS
