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

#include <climits>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string_view>

#include "constants/conversionFactors.hpp"
#include "defaults.hpp"
#include "externalQMScripts.hpp"
#include "systemInfo.hpp"

namespace
{
    constexpr bool _WITH_ASE_              = PQ_BUILD_WITH_ASE;
    constexpr bool _WITH_MPI_              = PQ_BUILD_WITH_MPI;
    constexpr bool _WITH_KOKKOS_           = PQ_BUILD_WITH_KOKKOS;
    constexpr bool _WITH_PYTHON_BINDINGS_  = PQ_BUILD_WITH_PYTHON_BINDINGS;
    constexpr bool _WITH_PYTHON_EMBEDDING_ = PQ_BUILD_WITH_PYTHON_EMBEDDING;
    constexpr bool _SHARED_                = PQ_BUILD_SHARED;
    constexpr bool _STATIC_                = PQ_BUILD_STATIC;
    constexpr bool _WITH_SINGULARITY_      = PQ_BUILD_WITH_SINGULARITY;

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

    void writeExternalQMCapabilities(std::ostream &output)
    {
        output << "    \"external_qm\": {\n"
               << "      \"script_mode\": \""
               << ((_STATIC_ || _WITH_SINGULARITY_) ? "full_path_only"
                                                    : "bundled_or_full_path")
               << "\",\n"
               << "      \"programs\": {\n";

        for (size_t methodIndex = 0;
             methodIndex < cli::_EXTERNAL_QM_METHODS_.size();
             ++methodIndex)
        {
            const auto method = cli::_EXTERNAL_QM_METHODS_[methodIndex];
            output << "        ";
            writeJsonString(output, cli::externalQMProgramName(method));
            output << ": {\n"
                   << "          \"recommended_script\": ";

            const auto recommended = cli::recommendedExternalQMScript(method);
            if (recommended.empty())
                output << "null";
            else
                writeJsonString(output, recommended);

            output << ",\n"
                   << "          \"scripts\": [\n";

            const auto scripts = cli::externalQMScripts(method);
            for (size_t scriptIndex = 0; scriptIndex < scripts.size();
                 ++scriptIndex)
            {
                const auto &script = scripts[scriptIndex];
                output << "            {\"name\": ";
                writeJsonString(output, script.name);
                output << ", \"label\": ";
                writeJsonString(output, script.label);

                if (!script.requiredFileKeyword.empty())
                {
                    output << ", \"required_file_keywords\": [";
                    writeJsonString(output, script.requiredFileKeyword);
                    output << ']';
                }

                if (!script.requiredWorkingFile.empty())
                {
                    output << ", \"required_working_files\": [";
                    writeJsonString(output, script.requiredWorkingFile);
                    output << ']';
                }

                output << '}'
                       << (scriptIndex + 1 == scripts.size() ? "\n" : ",\n");
            }

            output << "          ]\n"
                   << "        }"
                   << (methodIndex + 1 == cli::_EXTERNAL_QM_METHODS_.size()
                           ? "\n"
                           : ",\n");
        }

        output << "      }\n"
               << "    }";
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
    output << std::boolalpha
           << std::setprecision(std::numeric_limits<double>::max_digits10);

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
           << "    \"python_embedding\": " << _WITH_PYTHON_EMBEDDING_ << ",\n"
           << "    \"shared\": " << _SHARED_ << ",\n"
           << "    \"singularity\": " << _WITH_SINGULARITY_ << "\n"
           << "  },\n"
           << "  \"cli\": {\n"
           << "    \"input_validation\": {\n"
           << "      \"schema\": \"pq.validation\",\n"
           << "      \"schema_version\": 1,\n"
           << "      \"formats\": [\"text\", \"json\"],\n"
           << "      \"scopes\": [\"portable\", \"installed\"]\n"
           << "    }\n"
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
           << "    ],\n";
    writeExternalQMCapabilities(output);
    output << ",\n"
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
           << "      \"nstep\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 1, \"maximum\": "
           << INT_MAX << "\n"
           << "      },\n"
           << "      \"timestep\": {\n"
           << "        \"type\": \"number\", \"unit\": \"fs\", "
              "\"exclusive_minimum\": 0\n"
           << "      },\n"
           << "      \"output_freq\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 0, \"maximum\": "
           << INT_MAX << "\n"
           << "      },\n"
           << "      \"random_seed\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 0, "
              "\"maximum\": "
           << UINT32_MAX << "\n"
           << "      },\n"
           << "      \"temp\": {\n"
           << "        \"type\": \"number\", \"unit\": \"K\", "
              "\"minimum\": 0\n"
           << "      },\n"
           << "      \"start_temp\": {\n"
           << "        \"type\": \"number\", \"unit\": \"K\", "
              "\"minimum\": 0\n"
           << "      },\n"
           << "      \"end_temp\": {\n"
           << "        \"type\": \"number\", \"unit\": \"K\", "
              "\"minimum\": 0\n"
           << "      },\n"
           << "      \"temp_ramp_steps\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 0, \"maximum\": "
           << INT_MAX << "\n"
           << "      },\n"
           << "      \"temp_ramp_frequency\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 1, \"maximum\": "
           << INT_MAX << "\n"
           << "      },\n"
           << "      \"t_relaxation\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps\", "
              "\"exclusive_minimum\": 0, \"maximum\": "
           << std::numeric_limits<double>::max() / constants::_PS_TO_FS_
           << ", \"minimum_from\": {\"parameter\": \"timestep\", "
              "\"factor\": 0.001}, \"default\": "
           << defaults::_BERENDSEN_THERMOSTAT_RELAX_TIME_ << "\n"
           << "      },\n"
           << "      \"friction\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps^-1\", "
              "\"minimum\": 0, \"maximum\": "
           << std::numeric_limits<double>::max() / 1.0e12 << ", \"default\": "
           << defaults::_LANGEVIN_THERMOSTAT_FRICTION_ / 1.0e12 << "\n"
           << "      },\n"
           << "      \"nh-chain_length\": {\n"
           << "        \"type\": \"integer\", \"minimum\": 1, \"maximum\": "
           << INT_MAX
           << ", \"default\": " << defaults::_NH_CHAIN_LENGTH_DEFAULT_ << "\n"
           << "      },\n"
           << "      \"coupling_frequency\": {\n"
           << "        \"type\": \"number\", \"unit\": \"cm^-1\", "
              "\"minimum\": 0, \"maximum\": "
           << std::sqrt(std::numeric_limits<double>::max()) /
                  constants::_PER_CM_TO_HZ_
           << ", \"default\": " << defaults::_NH_COUPLING_FREQ_ << "\n"
           << "      },\n"
           << "      \"pressure\": {\n"
           << "        \"type\": \"number\", \"unit\": \"bar\"\n"
           << "      },\n"
           << "      \"p_relaxation\": {\n"
           << "        \"type\": \"number\", \"unit\": \"ps\", "
              "\"exclusive_minimum\": 0, \"maximum\": "
           << std::numeric_limits<double>::max() / constants::_PS_TO_FS_
           << ", \"minimum_from\": {\"parameter\": \"timestep\", "
              "\"factor\": 0.001}, \"default\": "
           << defaults::_BERENDSEN_MANOSTAT_RELAX_TIME_ << "\n"
           << "      },\n"
           << "      \"compressibility\": {\n"
           << "        \"type\": \"number\", \"unit\": \"bar^-1\", "
              "\"minimum\": 0, \"default\": "
           << defaults::_COMPRESSIBILITY_WATER_DEFAULT_ << "\n"
           << "      },\n"
           << "      \"density\": {\n"
           << "        \"type\": \"number\", \"unit\": \"kg/L\", "
              "\"exclusive_minimum\": 0\n"
           << "      },\n"
           << "      \"rcoulomb\": {\n"
           << "        \"type\": \"number\", \"unit\": \"angstrom\", "
              "\"minimum\": 0, "
              "\"default\": "
           << defaults::_COULOMB_CUT_OFF_DEFAULT_ << "\n"
           << "      }\n"
           << "    }\n"
           << "  }\n"
           << "}\n";

    output.flags(flags);
    output.precision(precision);
}
