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

#include "aseDftbRunner.hpp"

#include "pybind11/embed.h"

using QM::AseDftbRunner;

/**
 * @brief Construct a new AseDftbRunner::AseDftbRunner object
 *
 * @param slakos
 *
 * @throw py::error_already_set if the import of the mace module fails
 */
AseDftbRunner::AseDftbRunner(const std::string &slakos) : ASEQMRunner()
{
    try
    {
        const py::module_ calculator =
            py::module_::import("ase.calculators.dftb.Dftb");

        const py::dict calculatorArgs;

        if (slakos == "3ob" || slakos == "matsci")
        {
            const std::string slakosDir    = SLAKOS_DIR + slakos + "/skfiles/";
            calculatorArgs["slakos"] = slakosDir.c_str();
        }
        else
            calculatorArgs["slakos"] = slakos.c_str();

        _calculator = calculator(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}
