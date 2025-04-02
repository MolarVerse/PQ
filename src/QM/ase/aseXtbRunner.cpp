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

#include "aseXtbRunner.hpp"

#include "hubbardDerivMap.hpp"
#include "pybind11/embed.h"

using QM::AseXtbRunner;
using namespace constants;

/**
 * @brief Construct a new AseDftbRunner::AseDftbRunner object
 *
 * @param slakos
 *
 * @throw py::error_already_set if the import of the mace module fails
 */
AseXtbRunner::AseXtbRunner(const std::string &method) : ASEQMRunner()
{
    try
    {
        const py::module_ calculator =
            py::module_::import("ase.calculators.dftb");

        const py::dict calculatorArgs;

        calculatorArgs["Hamiltonian_"]       = "xTB";
        calculatorArgs["Hamiltonian_Method"] = method;

        // default would be 1, which is incompatible with DFTB3
        calculatorArgs["ParserOptions_ParserVersion"] = "12";
        // SCC = "Yes" is mandatory for SCC cycles to be performed
        calculatorArgs["Hamiltonian_SCC"]              = "Yes";
        calculatorArgs["Hamiltonian_SCCTolerance"]     = "1e-6";
        calculatorArgs["Hamiltonian_MaxSCCIterations"] = "250";
        calculatorArgs["kpts"] = py::make_tuple(1, 1, 1);
        _calculator            = calculator.attr("Dftb")(**calculatorArgs);
    }
    catch (const py::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}