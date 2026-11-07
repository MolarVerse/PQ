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

#include "interrupt.hpp"

#ifdef WITH_PYBIND11
#include <pybind11/pybind11.h>
#endif

namespace utilities
{
    /**
     * @brief throw if a signal recorded by embedded Python is pending
     *
     * @throw pybind11::error_already_set e.g. KeyboardInterrupt
     */
    void throwIfInterrupted()
    {
#ifdef WITH_PYBIND11
        if (::Py_IsInitialized() != 0 && ::PyErr_CheckSignals() != 0)
            throw pybind11::error_already_set();
#endif
    }
}   // namespace utilities
