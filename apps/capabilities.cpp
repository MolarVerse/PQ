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

#include <array>
#include <cstdint>
#include <iomanip>
#include <ostream>
#include <span>
#include <string_view>

#include "defaults.hpp"
#include "jsonOutput.hpp"
#include "systemInfo.hpp"

namespace
{
    constexpr bool _WITH_ASE_              = PQ_BUILD_WITH_ASE;
    constexpr bool _WITH_MPI_              = PQ_BUILD_WITH_MPI;
    constexpr bool _WITH_KOKKOS_           = PQ_BUILD_WITH_KOKKOS;
    constexpr bool _WITH_PYTHON_BINDINGS_  = PQ_BUILD_WITH_PYTHON_BINDINGS;
    constexpr bool _WITH_PYTHON_EMBEDDING_ = PQ_BUILD_WITH_PYTHON_EMBEDDING;

    constexpr auto _JOB_TYPES_ = std::array<std::string_view, 5>{
        "mm-md",
        "mm-hessian",
        "mm-opt",
        "qm-md",
        "qm-rpmd"
    };
    constexpr auto _QM_PROGRAMS_ = std::array<std::string_view, 3>{
        "dftbplus",
        "pyscf",
        "turbomole"
    };
    constexpr auto _ASE_QM_PROGRAMS_ = std::array<std::string_view, 6>{
        "ase_dftbplus",
        "ase_xtb",
        "fennol",
        "mace",
        "mace_mp",
        "mace_off"
    };
    constexpr auto _THERMOSTATS_ = std::array<std::string_view, 5>{
        "none",
        "berendsen",
        "velocity_rescaling",
        "langevin",
        "nh-chain"
    };
    constexpr auto _MANOSTATS_ = std::array<std::string_view, 3>{
        "none",
        "berendsen",
        "stochastic_rescaling"
    };
    constexpr auto _PRESSURE_ISOTROPIES_ = std::array<std::string_view, 6>{
        "isotropic",
        "xy",
        "xz",
        "yz",
        "anisotropic",
        "full_anisotropic"
    };

    void writeStringArray(
        cli::JsonWriter                         &json,
        const std::string_view                   key,
        const std::span<const std::string_view> values
    )
    {
        json.beginArray(key);
        for (const auto value : values) json.value(value);
        json.endArray();
    }

    void writeParameters(cli::JsonWriter &json)
    {
        json.beginObject("parameters");

        json.beginObject("random_seed");
        json.value("type", "integer");
        json.value("minimum", 0);
        json.value("maximum", UINT32_MAX);
        json.endObject();

        json.beginObject("t_relaxation");
        json.value("type", "number");
        json.value("unit", "ps");
        json.value("default", defaults::_BERENDSEN_THERMOSTAT_RELAX_TIME_);
        json.endObject();

        json.beginObject("friction");
        json.value("type", "number");
        json.value("unit", "ps^-1");
        json.value(
            "default",
            defaults::_LANGEVIN_THERMOSTAT_FRICTION_ / 1.0e12
        );
        json.endObject();

        json.beginObject("nh-chain_length");
        json.value("type", "integer");
        json.value("default", defaults::_NH_CHAIN_LENGTH_DEFAULT_);
        json.endObject();

        json.beginObject("coupling_frequency");
        json.value("type", "number");
        json.value("unit", "cm^-1");
        json.value("default", defaults::_NH_COUPLING_FREQ_);
        json.endObject();

        json.beginObject("p_relaxation");
        json.value("type", "number");
        json.value("unit", "ps");
        json.value("default", defaults::_BERENDSEN_MANOSTAT_RELAX_TIME_);
        json.endObject();

        json.beginObject("compressibility");
        json.value("type", "number");
        json.value("unit", "bar^-1");
        json.value("default", defaults::_COMPRESSIBILITY_WATER_DEFAULT_);
        json.endObject();

        json.beginObject("rcoulomb");
        json.value("type", "number");
        json.value("unit", "angstrom");
        json.value("default", defaults::_COULOMB_CUT_OFF_DEFAULT_);
        json.endObject();

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
    output << std::setprecision(15);

    auto json = JsonWriter(output);
    json.beginObject();
    json.value("schema", "pq.capabilities");
    json.value("schema_version", 1);
    json.value("version", sysinfo::_VERSION_);

    json.beginObject("build");
    json.value("ase", _WITH_ASE_);
    json.value("mpi", _WITH_MPI_);
    json.value("kokkos", _WITH_KOKKOS_);
    json.value("python_bindings", _WITH_PYTHON_BINDINGS_);
    json.value("python_embedding", _WITH_PYTHON_EMBEDDING_);
    json.endObject();

    json.beginObject("input");
    writeStringArray(json, "job_types", _JOB_TYPES_);

    json.beginArray("qm_programs");
    for (const auto program : _QM_PROGRAMS_) json.value(program);
    if (_WITH_ASE_)
        for (const auto program : _ASE_QM_PROGRAMS_) json.value(program);
    json.endArray();

    writeStringArray(json, "thermostats", _THERMOSTATS_);
    writeStringArray(json, "manostats", _MANOSTATS_);
    writeStringArray(json, "pressure_isotropies", _PRESSURE_ISOTROPIES_);
    writeParameters(json);
    json.endObject();
    json.endObject();
    output << '\n';

    output.flags(flags);
    output.precision(precision);
}
