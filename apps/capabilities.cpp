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

#include "capabilities.hpp"

#include <cstdint>
#include <iomanip>
#include <ostream>
#include <string_view>

#include "defaults.hpp"
#include "systemInfo.hpp"

namespace
{
    constexpr bool _WITH_ASE_              = PQ_BUILD_WITH_ASE;
    constexpr bool _WITH_MPI_              = PQ_BUILD_WITH_MPI;
    constexpr bool _WITH_KOKKOS_           = PQ_BUILD_WITH_KOKKOS;
    constexpr bool _WITH_PYTHON_BINDINGS_  = PQ_BUILD_WITH_PYTHON_BINDINGS;
    constexpr bool _WITH_PYTHON_EMBEDDING_ = PQ_BUILD_WITH_PYTHON_EMBEDDING;

    void writeJsonString(std::ostream &output, const std::string_view value)
    {
        output << '"';
        for (const auto character : value)
        {
            if ('"' == character || '\\' == character)
                output << '\\';
            output << character;
        }
        output << '"';
    }
}   // namespace

/**
 * @brief Writes the versioned machine-readable PQ capabilities.
 *
 * @param output
 */
void cli::writeCapabilities(std::ostream &output)
{
    const auto flags     = output.flags();
    const auto precision = output.precision();
    output << std::boolalpha << std::setprecision(15);

    output << "{\n"
           << "  \"schema\": \"pq.capabilities\",\n"
           << "  \"schema_version\": 1,\n"
           << "  \"version\": ";
    writeJsonString(output, sysinfo::_VERSION_);
    output << ",\n"
           << "  \"build\": {\n"
           << "    \"ase\": " << _WITH_ASE_ << ",\n"
           << "    \"mpi\": " << _WITH_MPI_ << ",\n"
           << "    \"kokkos\": " << _WITH_KOKKOS_ << ",\n"
           << "    \"python_bindings\": " << _WITH_PYTHON_BINDINGS_ << ",\n"
           << "    \"python_embedding\": " << _WITH_PYTHON_EMBEDDING_ << "\n"
           << "  },\n"
           << "  \"input\": {\n"
           << "    \"job_types\": [\n"
           << "      \"mm-md\", \"mm-hessian\", \"mm-opt\", \"qm-md\", "
              "\"qm-rpmd\"\n"
           << "    ],\n"
           << "    \"qm_programs\": [\n"
           << "      \"dftbplus\", \"pyscf\", \"turbomole\"";
    if (_WITH_ASE_)
        output << ", \"ase_dftbplus\", \"ase_xtb\", \"fennol\", \"mace\", "
                  "\"mace_mp\", \"mace_off\"";
    output << "\n"
           << "    ],\n"
           << "    \"thermostats\": [\n"
           << "      \"none\", \"berendsen\", \"velocity_rescaling\", "
              "\"langevin\", \"nh-chain\"\n"
           << "    ],\n"
           << "    \"manostats\": [\n"
           << "      \"none\", \"berendsen\", \"stochastic_rescaling\"\n"
           << "    ],\n"
           << "    \"pressure_isotropies\": [\n"
           << "      \"isotropic\", \"xy\", \"xz\", \"yz\", "
              "\"anisotropic\", \"full_anisotropic\"\n"
           << "    ],\n"
           << "    \"parameters\": {\n"
           << "      \"random_seed\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 0, "
              "\"maximum\": "
           << UINT32_MAX << "\n"
           << "      },\n"
           << "      \"t_relaxation\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps\", \"default\": "
           << defaults::_BERENDSEN_THERMOSTAT_RELAX_TIME_ << "\n"
           << "      },\n"
           << "      \"friction\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps^-1\", "
              "\"default\": "
           << defaults::_LANGEVIN_THERMOSTAT_FRICTION_ / 1.0e12 << "\n"
           << "      },\n"
           << "      \"nh-chain_length\": {\n"
           << "        \"type\": \"integer\", \"default\": "
           << defaults::_NH_CHAIN_LENGTH_DEFAULT_ << "\n"
           << "      },\n"
           << "      \"coupling_frequency\": {\n"
           << "        \"type\": \"number\", \"unit\": \"cm^-1\", "
              "\"default\": "
           << defaults::_NH_COUPLING_FREQ_ << "\n"
           << "      },\n"
           << "      \"p_relaxation\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps\", \"default\": "
           << defaults::_BERENDSEN_MANOSTAT_RELAX_TIME_ << "\n"
           << "      },\n"
           << "      \"compressibility\": {\n"
           << "        \"type\": \"number\", \"unit\": \"bar^-1\", "
              "\"default\": "
           << defaults::_COMPRESSIBILITY_WATER_DEFAULT_ << "\n"
           << "      },\n"
           << "      \"rcoulomb\": {\n"
           << "        \"type\": \"number\", \"unit\": \"angstrom\", "
              "\"default\": "
           << defaults::_COULOMB_CUT_OFF_DEFAULT_ << "\n"
           << "      }\n"
           << "    }\n"
           << "  }\n"
           << "}\n";

    output.flags(flags);
    output.precision(precision);
}
