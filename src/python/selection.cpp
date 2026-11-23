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

    std::string takePythonError()
    {
        const auto error = own(::PyErr_GetRaisedException());
        if (error == nullptr)
        {
            return "unknown Python error";
        }

        const auto  text = own(::PyObject_Str(error.get()));
        const char *utf8 =
            text == nullptr ? nullptr : ::PyUnicode_AsUTF8(text.get());
        if (utf8 == nullptr)
        {
            ::PyErr_Clear();
            return Py_TYPE(error.get())->tp_name;
        }

        return std::format("{}: {}", Py_TYPE(error.get())->tp_name, utf8);
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
 * @throw exc::InputFileException if PQAnalysis cannot be imported or the
 * selection fails
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
            std::format(
                "Selection \"{}\" needs the Python package PQAnalysis, but "
                "\"{}\" could not be imported ({}). Install it into the "
                "Python PQ was built with: pip install PQAnalysis",
                selection,
                SELECTION_MODULE,
                takePythonError()
            )
        );
    }

    const auto function =
        own(::PyObject_GetAttrString(module.get(), SELECTION_FUNCTION));
    if (function == nullptr || ::PyCallable_Check(function.get()) == 0)
    {
        const auto reason =
            function == nullptr ? takePythonError() : "not a callable object";
        throw exc::InputFileException(
            std::format(
                "\"{}.{}\" is not usable ({}). Update PQAnalysis in the "
                "Python PQ was built with: pip install --upgrade PQAnalysis",
                SELECTION_MODULE,
                SELECTION_FUNCTION,
                reason
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
                takePythonError()
            )
        );
    }

    const auto value = own(::PyObject_CallObject(function.get(), args.get()));
    const auto iter =
        value == nullptr ? own(nullptr) : own(::PyObject_GetIter(value.get()));
    if (iter == nullptr)
    {
        throw exc::InputFileException(
            std::format(
                "Selection \"{}\" failed in PQAnalysis ({})",
                selection,
                takePythonError()
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
                takePythonError()
            )
        );
    }

    return result;
}

#endif   // WITH_PYTHON_BINDINGS
