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
#include <initializer_list>
#include <ostream>
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

    void writeStringArray(
        cli::JsonWriter                                  &json,
        const std::string_view                            key,
        const std::initializer_list<std::string_view> values
    )
    {
        json.beginArray(key);
        for (const auto value : values) json.value(value);
        json.endArray();
    }

    template <typename T>
    void writeParameter(
        cli::JsonWriter       &json,
        const std::string_view name,
        const std::string_view type,
        const std::string_view unit,
        const T                defaultValue
    )
    {
        json.beginObject(name);
        json.value("type", type);
        if (!unit.empty()) json.value("unit", unit);
        json.value("default", defaultValue);
        json.endObject();
    }

    void writeParameters(cli::JsonWriter &json)
    {
        json.beginObject("parameters");

        json.beginObject("random_seed");
        json.value("type", "integer");
        json.value("minimum", 0);
        json.value("maximum", UINT32_MAX);
        json.endObject();

        writeParameter(
            json,
            "t_relaxation",
            "number",
            "ps",
            defaults::_BERENDSEN_THERMOSTAT_RELAX_TIME_
        );
        writeParameter(
            json,
            "friction",
            "number",
            "ps^-1",
            defaults::_LANGEVIN_THERMOSTAT_FRICTION_ / 1.0e12
        );
        writeParameter(
            json,
            "nh-chain_length",
            "integer",
            "",
            defaults::_NH_CHAIN_LENGTH_DEFAULT_
        );
        writeParameter(
            json,
            "coupling_frequency",
            "number",
            "cm^-1",
            defaults::_NH_COUPLING_FREQ_
        );
        writeParameter(
            json,
            "p_relaxation",
            "number",
            "ps",
            defaults::_BERENDSEN_MANOSTAT_RELAX_TIME_
        );
        writeParameter(
            json,
            "compressibility",
            "number",
            "bar^-1",
            defaults::_COMPRESSIBILITY_WATER_DEFAULT_
        );
        writeParameter(
            json,
            "rcoulomb",
            "number",
            "angstrom",
            defaults::_COULOMB_CUT_OFF_DEFAULT_
        );

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
    writeStringArray(
        json,
        "job_types",
        {"mm-md", "mm-hessian", "mm-opt", "qm-md", "qm-rpmd"}
    );

    json.beginArray("qm_programs");
    for (const auto program : {"dftbplus", "pyscf", "turbomole"})
        json.value(program);
    if (_WITH_ASE_)
        for (const auto program : {
                 "ase_dftbplus",
                 "ase_xtb",
                 "fennol",
                 "mace",
                 "mace_mp",
                 "mace_off"
             })
            json.value(program);
    json.endArray();

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
    json.endObject();
    output << '\n';

    output.flags(flags);
    output.precision(precision);
}
