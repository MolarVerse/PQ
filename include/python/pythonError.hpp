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

#ifndef PYTHON__PYTHON_ERROR_HPP
#define PYTHON__PYTHON_ERROR_HPP

#include <Python.h>

#include <memory>
#include <string>

namespace pq_python::error
{
    using PyObjectPtr = std::unique_ptr<PyObject, decltype(&::Py_DecRef)>;

    /**
     * @brief the raised Python exception as text
     *
     * @details missingModule is the name of a ModuleNotFoundError, else empty
     */
    struct PythonError
    {
        std::string description;
        std::string missingModule;
    };

    PyObjectPtr own(PyObject *object);

    PythonError takePythonError();

    std::string importErrorMessage(
        const std::string &selection,
        const std::string &moduleName,
        const PythonError &pythonError
    );
}   // namespace pq_python::error

#endif   // PYTHON__PYTHON_ERROR_HPP
