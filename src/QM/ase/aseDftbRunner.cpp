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

#include <pybind11/embed.h>

#include "qmSettings.hpp"

namespace QM
{

    /**
     * @brief Construct a new AseDftbRunner::AseDftbRunner object
     *
     * @param settings The QM settings to configure the ASE DFTB runner with,
     * including method-specific parameters such as the DFTB version and any
     * relevant computational options
     * @param dftbSettings The DFTB+ specific settings including the Slakos
     * path, whether third order corrections should be applied, the Hubbard
     * derivatives, and whether dispersion corrections should be included
     *
     * @throw pybind11::error_already_set if the import of the mace module fails
     */
    AseDftbRunner::AseDftbRunner(
        const QMSettings   &settings,
        const DftbSettings &dftbSettings
    )
        : AseQMRunner(settings)
    {
        try
        {
            const pybind11::module_ calculator =
                pybind11::module_::import("ase.calculators.dftb");

            const pybind11::dict calculatorArgs;

            calculatorArgs["slako_dir"] = dftbSettings.getSlakosPath().c_str();

            if (QMSettings::useDispersionCorr())
            {
                calculatorArgs["Hamiltonian_Dispersion"]            = "DftD3{";
                calculatorArgs["Hamiltonian_Dispersion_Damping_a1"] = "0.746";
                calculatorArgs["Hamiltonian_Dispersion_Damping_a2"] = "4.191";
                calculatorArgs["Hamiltonian_Dispersion_Damping"] =
                    "BeckeJohnson{";
                calculatorArgs["Hamiltonian_Dispersion_s6"] = "1.0";
                calculatorArgs["Hamiltonian_Dispersion_s8"] = "3.209";
            }

            if (dftbSettings.useThirdOrderDftb())
            {
                const auto hubbDerivDict = dftbSettings.getHubbardDerivs();
                if (!hubbDerivDict.has_value())
                {
                    throw std::runtime_error(
                        "Third order DFTB requested but Hubbard derivatives "
                        "are not provided."
                    );
                }

                calculatorArgs["Hamiltonian_ThirdOrderFull"] = "Yes";
                calculatorArgs["Hamiltonian_hubbardderivs_"] = "";
                for (const auto &[key, value] : hubbDerivDict.value())
                {
                    auto _key = "Hamiltonian_hubbardderivs_" + key;
                    calculatorArgs[_key.c_str()] = value;
                }
            }
            // default would be 1, which is incompatible with DFTB3
            calculatorArgs["ParserOptions_ParserVersion"] = "12";
            // SCC = "Yes" is mandatory for SCC cycles to be performed
            calculatorArgs["Hamiltonian_SCC"]              = "Yes";
            calculatorArgs["Hamiltonian_SCCTolerance"]     = "1e-6";
            calculatorArgs["Hamiltonian_MaxSCCIterations"] = "250";
            calculatorArgs["kpts"] = pybind11::make_tuple(1, 1, 1);
            _setAseCalculator(calculator.attr("Dftb")(**calculatorArgs));
        }
        catch (const pybind11::error_already_set &)
        {
            ::PyErr_Print();
            throw;
        }
    }

}   // namespace QM
