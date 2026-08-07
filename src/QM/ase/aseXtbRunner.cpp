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

using QM::AseXtbRunner;

/**
 * @brief Construct a new AseXtbRunner::AseXtbRunner object
 *
 * @param slakos
 *
 * @throw pybind11::error_already_set if the import of the mace module fails
 */
AseXtbRunner::AseXtbRunner(const std::string &method) : AseQMRunner()
{
    try
    {
        const pybind11::module_ calculator =
            pybind11::module_::import("ase.calculators.dftb");

        const pybind11::dict calculatorArgs;

        calculatorArgs["Hamiltonian_"]       = "xTB";
        calculatorArgs["Hamiltonian_Method"] = method;

        // default would be 1, which is incompatible with DFTB3
        calculatorArgs["ParserOptions_ParserVersion"] = "12";
        // SCC = "Yes" is mandatory for SCC cycles to be performed
        calculatorArgs["Hamiltonian_SCC"]              = "Yes";
        calculatorArgs["Hamiltonian_SCCTolerance"]     = "1e-6";
        calculatorArgs["Hamiltonian_MaxSCCIterations"] = "250";
        calculatorArgs["kpts"] = pybind11::make_tuple(1, 1, 1);
        setAseCalculator(calculator.attr("Dftb")(**calculatorArgs));
    }
    catch (const pybind11::error_already_set &)
    {
        ::PyErr_Print();
        throw;
    }
}
