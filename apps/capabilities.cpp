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
#include <initializer_list>
#include <iomanip>
#include <limits>
#include <ostream>
#include <string_view>

#include "constants/conversionFactors.hpp"
#include "defaults.hpp"
#include "externalQMScripts.hpp"
#include "jsonOutput.hpp"
#include "systemInfo.hpp"

namespace
{
    constexpr bool withAse             = PQ_BUILD_WITH_ASE;
    constexpr bool withMpi             = PQ_BUILD_WITH_MPI;
    constexpr bool withPythonBindings  = PQ_BUILD_WITH_PYTHON_BINDINGS;
    constexpr bool withPythonEmbedding = PQ_BUILD_WITH_PYTHON_EMBEDDING;
    constexpr bool sharedBuild         = PQ_BUILD_SHARED;
    constexpr bool staticBuild         = PQ_BUILD_STATIC;
    constexpr bool withSingularity     = PQ_BUILD_WITH_SINGULARITY;

    void writeStringArray(
        cli::JsonWriter                              &json,
        const std::string_view                        key,
        const std::initializer_list<std::string_view> values
    )
    {
        json.beginArray(key);
        for (const auto value : values) json.value(value);
        json.endArray();
    }

    void beginParameter(
        cli::JsonWriter       &json,
        const std::string_view name,
        const std::string_view type,
        const std::string_view unit
    )
    {
        json.beginObject(name);
        json.value("type", type);
        if (!unit.empty())
            json.value("unit", unit);
    }

    void beginParameter(
        cli::JsonWriter       &json,
        const std::string_view name,
        const std::string_view type
    )
    {
        beginParameter(json, name, type, "");
    }

    void writeBuildCapabilities(cli::JsonWriter &json)
    {
        json.beginObject("build");
        json.value("ase", withAse);
        json.value("mpi", withMpi);
        json.value("python_bindings", withPythonBindings);
        json.value("python_embedding", withPythonEmbedding);
        json.value("shared", sharedBuild);
        json.value("static", staticBuild);
        json.value("singularity", withSingularity);
        json.endObject();
    }

    void writeCliCapabilities(cli::JsonWriter &json)
    {
        json.beginObject("cli");
        json.beginObject("input_validation");
        json.value("schema", "pq.validation");
        json.value("schema_version", 1);
        writeStringArray(json, "formats", {"text", "json"});
        writeStringArray(json, "scopes", {"portable", "installed"});
        json.endObject();
        json.endObject();
    }

    void writeExternalQMCapabilities(cli::JsonWriter &json)
    {
        json.beginObject("external_qm");
        json.value(
            "script_mode",
            (staticBuild || withSingularity) ? "full_path_only"
                                             : "bundled_or_full_path"
        );
        json.beginObject("programs");

        for (const auto method : cli::externalQMMethods)
        {
            json.beginObject(cli::externalQMProgramName(method));

            const auto recommended = cli::recommendedExternalQMScript(method);
            if (recommended.empty())
                json.value("recommended_script", nullptr);
            else
                json.value("recommended_script", recommended);

            json.beginArray("scripts");
            for (const auto &script : cli::externalQMScripts(method))
            {
                json.beginObject();
                json.value("name", script.name);
                json.value("label", script.label);

                if (!script.requiredFileKeyword.empty())
                    writeStringArray(
                        json,
                        "required_file_keywords",
                        {script.requiredFileKeyword}
                    );

                if (!script.requiredWorkingFile.empty())
                    writeStringArray(
                        json,
                        "required_working_files",
                        {script.requiredWorkingFile}
                    );

                json.endObject();
            }
            json.endArray();
            json.endObject();
        }

        json.endObject();
        json.endObject();
    }

    void writeParameters(cli::JsonWriter &json)
    {
        json.beginObject("parameters");

        beginParameter(json, "nstep", "integer");
        json.value("minimum", 1);
        json.value("maximum", INT_MAX);
        json.endObject();

        beginParameter(json, "timestep", "number", "fs");
        json.value("exclusive_minimum", 0);
        json.endObject();

        beginParameter(json, "output_freq", "integer");
        json.value("minimum", 0);
        json.value("maximum", INT_MAX);
        json.endObject();

        beginParameter(json, "random_seed", "integer");
        json.value("minimum", 0);
        json.value("maximum", UINT32_MAX);
        json.endObject();

        for (const auto name : {"temp", "start_temp", "end_temp"})
        {
            beginParameter(json, name, "number", "K");
            json.value("minimum", 0);
            json.endObject();
        }

        beginParameter(json, "temp_ramp_steps", "integer");
        json.value("minimum", 0);
        json.value("maximum", INT_MAX);
        json.endObject();

        beginParameter(json, "temp_ramp_frequency", "integer");
        json.value("minimum", 1);
        json.value("maximum", INT_MAX);
        json.endObject();

        beginParameter(json, "t_relaxation", "number", "ps");
        json.value("exclusive_minimum", 0);
        json.value(
            "maximum",
            std::numeric_limits<double>::max() / constants::PS_TO_FS
        );
        json.beginObject("minimum_from");
        json.value("parameter", "timestep");
        json.value("factor", constants::FS_TO_PS);
        json.endObject();
        json.value("default", defaults::BERENDSEN_THERMOSTAT_RELAX_TIME);
        json.endObject();

        beginParameter(json, "friction", "number", "ps^-1");
        json.value("minimum", 0);
        json.value(
            "maximum",
            std::numeric_limits<double>::max() /
                defaults::MAX_FRICTION_CONVERSION
        );
        json.value(
            "default",
            defaults::LANGEVIN_THERMOSTAT_FRICTION /
                defaults::MAX_FRICTION_CONVERSION
        );
        json.endObject();

        beginParameter(json, "nh-chain_length", "integer");
        json.value("minimum", 1);
        json.value("maximum", INT_MAX);
        json.value("default", defaults::NH_CHAIN_LENGTH_DEFAULT);
        json.endObject();

        beginParameter(json, "coupling_frequency", "number", "cm^-1");
        json.value("minimum", 0);
        json.value(
            "maximum",
            std::sqrt(std::numeric_limits<double>::max()) /
                constants::PER_CM_TO_HZ
        );
        json.value("default", defaults::NH_COUPLING_FREQ);
        json.endObject();

        beginParameter(json, "pressure", "number", "bar");
        json.endObject();

        beginParameter(json, "p_relaxation", "number", "ps");
        json.value("exclusive_minimum", 0);
        json.value(
            "maximum",
            std::numeric_limits<double>::max() / constants::PS_TO_FS
        );
        json.beginObject("minimum_from");
        json.value("parameter", "timestep");
        json.value("factor", constants::FS_TO_PS);
        json.endObject();
        json.value("default", defaults::BERENDSEN_MANOSTAT_RELAX_TIME);
        json.endObject();

        beginParameter(json, "compressibility", "number", "bar^-1");
        json.value("minimum", 0);
        json.value("default", defaults::COMPRESSIBILITY_WATER_DEFAULT);
        json.endObject();

        beginParameter(json, "density", "number", "kg/L");
        json.value("exclusive_minimum", 0);
        json.endObject();

        beginParameter(json, "rcoulomb", "number", "angstrom");
        json.value("minimum", 0);
        json.value("default", defaults::COULOMB_CUT_OFF_DEFAULT);
        json.endObject();

        json.endObject();
    }

    void writeInputCapabilities(cli::JsonWriter &json)
    {
        json.beginObject("input");
        writeStringArray(
            json,
            "job_types",
            {"mm-md", "mm-hessian", "mm-opt", "qm-md", "qm-rpmd"}
        );

        json.beginArray("qm_programs");
        for (const auto program : {"dftbplus", "pyscf", "turbomole"})
            json.value(program);
        if (withAse)
            for (const auto program :
                 {"ase_dftbplus",
                  "ase_xtb",
                  "fennol",
                  "mace",
                  "mace_mp",
                  "mace_off"})
                json.value(program);
        json.endArray();

        writeExternalQMCapabilities(json);
        writeStringArray(
            json,
            "thermostats",
            {"none", "berendsen", "velocity_rescaling", "langevin", "nh-chain"}
        );
        writeStringArray(
            json,
            "manostats",
            {"none", "berendsen", "stochastic_rescaling"}
        );
        writeStringArray(
            json,
            "pressure_isotropies",
            {"isotropic", "xy", "xz", "yz", "anisotropic", "full_anisotropic"}
        );
        writeParameters(json);
        json.endObject();
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
    output << std::setprecision(std::numeric_limits<double>::max_digits10);

    auto json = JsonWriter(output);
    json.beginObject();
    json.value("schema", "pq.capabilities");
    json.value("schema_version", 2);
    json.value("version", sysinfo::VERSION);
    writeBuildCapabilities(json);
    writeCliCapabilities(json);
    writeInputCapabilities(json);
    json.endObject();
    output << '\n';

    output.flags(flags);
    output.precision(precision);
}
