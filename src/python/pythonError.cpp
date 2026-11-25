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

#include "pythonError.hpp"

#include <format>
#include <string>

/**
 * @brief take ownership of a new Python reference
 *
 * @param object may be nullptr
 * @return PyObjectPtr
 */
pq_python::error::PyObjectPtr pq_python::error::own(PyObject *object)
{
    return {object, &::Py_DecRef};
}

/**
 * @brief take the raised Python exception and clear it
 *
 * @return PythonError
 */
pq_python::error::PythonError pq_python::error::takePythonError()
{
    const auto error = own(::PyErr_GetRaisedException());
    if (error == nullptr)
    {
        return {
            .description   = "no Python exception was set",
            .missingModule = ""
        };
    }

    PythonError result{
        .description   = Py_TYPE(error.get())->tp_name,
        .missingModule = "",
    };

    const auto  text = own(::PyObject_Str(error.get()));
    const char *utf8 =
        text == nullptr ? nullptr : ::PyUnicode_AsUTF8(text.get());
    if (utf8 != nullptr)
    {
        result.description = std::format("{}: {}", result.description, utf8);
    }

    if (::PyErr_GivenExceptionMatches(
            error.get(),
            ::PyExc_ModuleNotFoundError
        ) != 0)
    {
        const auto  name = own(::PyObject_GetAttrString(error.get(), "name"));
        const char *nameUtf8 =
            name == nullptr || ::PyUnicode_Check(name.get()) == 0
                ? nullptr
                : ::PyUnicode_AsUTF8(name.get());
        if (nameUtf8 != nullptr)
        {
            result.missingModule = nameUtf8;
        }
    }

    ::PyErr_Clear();
    return result;
}

/**
 * @brief message for a failed import of the selection module
 *
 * @param selection
 * @param moduleName
 * @param pythonError
 * @return std::string
 */
std::string pq_python::error::importErrorMessage(
    const std::string &selection,
    const std::string &moduleName,
    const PythonError &pythonError
)
{
    if (pythonError.missingModule == "PQAnalysis")
    {
        return std::format(
            "Selection \"{}\" needs the Python package PQAnalysis, which "
            "is not installed ({}). Install it for the Python that PQ was "
            "built with: pip install PQAnalysis",
            selection,
            pythonError.description
        );
    }

    if (!pythonError.missingModule.empty())
    {
        return std::format(
            "Selection \"{}\" could not import \"{}\" because the Python "
            "module \"{}\" was not found ({})",
            selection,
            moduleName,
            pythonError.missingModule,
            pythonError.description
        );
    }

    return std::format(
        R"(Selection "{}" could not import "{}" ({}))",
        selection,
        moduleName,
        pythonError.description
    );
}

#endif   // WITH_PYTHON_BINDINGS
