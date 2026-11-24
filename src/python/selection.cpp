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

#include "selection.hpp"

#include <Python.h>

#include <format>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "exceptions.hpp"

namespace
{
    using PyObjectPtr = std::unique_ptr<PyObject, decltype(&::Py_DecRef)>;

    constexpr const char *SELECTION_MODULE = "PQAnalysis.topology.api";
    constexpr const char *SELECTION_FUNCTION =
        "selection_from_restart_file_as_list";

    PyObjectPtr own(PyObject *object) { return {object, &::Py_DecRef}; }

    struct PythonError
    {
        std::string description;
        std::string missingModule;
    };

    PythonError takePythonError()
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
            result.description =
                std::format("{}: {}", result.description, utf8);
        }

        if (::PyErr_GivenExceptionMatches(
                error.get(),
                ::PyExc_ModuleNotFoundError
            ) != 0)
        {
            const auto name =
                own(::PyObject_GetAttrString(error.get(), "name"));
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

    std::string importErrorMessage(
        const std::string &selection,
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
                SELECTION_MODULE,
                pythonError.missingModule,
                pythonError.description
            );
        }

        return std::format(
            "Selection \"{}\" could not import \"{}\" ({})",
            selection,
            SELECTION_MODULE,
            pythonError.description
        );
    }
}   // namespace

/**
 * @brief call selection tool of PQAnalysis.topology
 *
 * @param selection
 * @param restartFileName
 * @param moldescriptorFileName
 * @return std::vector<int>
 *
 * @throw exc::InputFileException with the failed step and the Python
 * exception if the selection cannot be run
 */
std::vector<int> pq_python::select(
    const std::string &selection,
    const std::string &restartFileName,
    const std::string &moldescriptorFileName
)
{
    // keep the interpreter alive: numpy cannot be imported again after
    // Py_Finalize, and PQ may already own it (pybind11 in apps/PQ.cpp)
    if (::Py_IsInitialized() == 0)
    {
        ::Py_Initialize();
    }

    const auto module = own(::PyImport_ImportModule(SELECTION_MODULE));
    if (module == nullptr)
    {
        throw exc::InputFileException(
            importErrorMessage(selection, takePythonError())
        );
    }

    const auto function =
        own(::PyObject_GetAttrString(module.get(), SELECTION_FUNCTION));
    if (function == nullptr)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" could not load \"{}\" from \"{}\" ({})",
                selection,
                SELECTION_FUNCTION,
                SELECTION_MODULE,
                takePythonError().description
            )
        );
    }
    if (::PyCallable_Check(function.get()) == 0)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" could not call \"{}.{}\": it is a \"{}\", "
                "not a function",
                selection,
                SELECTION_MODULE,
                SELECTION_FUNCTION,
                Py_TYPE(function.get())->tp_name
            )
        );
    }

    std::vector<std::string> argStrings{selection, restartFileName};
    if (!moldescriptorFileName.empty() &&
        std::ifstream{moldescriptorFileName}.good())
    {
        argStrings.push_back(moldescriptorFileName);
    }

    const auto args =
        own(::PyTuple_New(static_cast<Py_ssize_t>(argStrings.size())));
    bool argsSet = args != nullptr;
    for (size_t index = 0; argsSet && index < argStrings.size(); ++index)
    {
        PyObject *item = ::PyUnicode_FromString(argStrings[index].c_str());
        argsSet        = item != nullptr && ::PyTuple_SetItem(
                                         args.get(),
                                         static_cast<Py_ssize_t>(index),
                                         item
                                     ) == 0;
    }
    if (!argsSet)
    {
        throw exc::InputFileException(
            std::format(
                "Could not pass selection \"{}\" to Python ({})",
                selection,
                takePythonError().description
            )
        );
    }

    const auto value = own(::PyObject_CallObject(function.get(), args.get()));
    if (value == nullptr)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" failed in \"{}.{}\" ({})",
                selection,
                SELECTION_MODULE,
                SELECTION_FUNCTION,
                takePythonError().description
            )
        );
    }

    const auto iter = own(::PyObject_GetIter(value.get()));
    if (iter == nullptr)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" returned no list of atom indices ({})",
                selection,
                takePythonError().description
            )
        );
    }

    std::vector<int> result;
    while (const auto item = own(::PyIter_Next(iter.get())))
    {
        const auto index = ::PyLong_AsLong(item.get());
        if (index == -1 && ::PyErr_Occurred() != nullptr)
        {
            break;
        }
        result.push_back(static_cast<int>(index));
    }

    if (::PyErr_Occurred() != nullptr)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" returned an invalid atom index list ({})",
                selection,
                takePythonError().description
            )
        );
    }

    return result;
}

#endif   // WITH_PYTHON_BINDINGS
