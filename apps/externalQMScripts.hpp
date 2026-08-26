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

#ifndef _EXTERNAL_QM_SCRIPTS_HPP_

#define _EXTERNAL_QM_SCRIPTS_HPP_

#include <algorithm>
#include <array>
#include <span>
#include <string_view>

#include "qmSettings.hpp"

namespace cli
{
    struct ExternalQMScriptInfo
    {
        std::string_view name;
        std::string_view label;
        std::string_view requiredFileKeyword;
        std::string_view requiredWorkingFile;
    };

    inline constexpr auto dftbPlusScripts = std::array{ExternalQMScriptInfo{
        .name                = "dftbplus_periodic_stress",
        .label               = "DFTB+ periodic stress",
        .requiredFileKeyword = "dftb_file",
        .requiredWorkingFile = ""
    }};

    inline constexpr auto pyScfScripts = std::array{
        ExternalQMScriptInfo{
            .name                = "pyscf_hf.py",
            .label               = "UHF / STO-3G",
            .requiredFileKeyword = "",
            .requiredWorkingFile = ""
        },
        ExternalQMScriptInfo{
            .name                = "pyscf_mp2.py",
            .label               = "UMP2 / 6-311++G**",
            .requiredFileKeyword = "",
            .requiredWorkingFile = ""
        }
    };

    inline constexpr auto turbomoleScripts = std::array{ExternalQMScriptInfo{
        .name                = "turbomole_ricc2",
        .label               = "RI-MP2",
        .requiredFileKeyword = "",
        .requiredWorkingFile = "tm_define.template"
    }};

    inline constexpr auto externalQMMethods = std::array{
        settings::QMMethod::DFTBPLUS,
        settings::QMMethod::PYSCF,
        settings::QMMethod::TURBOMOLE
    };

    inline std::span<const ExternalQMScriptInfo> externalQMScripts(
        const settings::QMMethod method
    )
    {
        using enum settings::QMMethod;

        switch (method)
        {
            case DFTBPLUS: return dftbPlusScripts;
            case PYSCF: return pyScfScripts;
            case TURBOMOLE: return turbomoleScripts;

            case NONE:
            case ASEDFTBPLUS:
            case ASEXTB:
            case MACE:
            case FENNOL: return {};
        }

        return {};
    }

    inline std::string_view externalQMProgramName(
        const settings::QMMethod method
    )
    {
        using enum settings::QMMethod;

        switch (method)
        {
            case DFTBPLUS: return "dftbplus";
            case PYSCF: return "pyscf";
            case TURBOMOLE: return "turbomole";

            case NONE:
            case ASEDFTBPLUS:
            case ASEXTB:
            case MACE:
            case FENNOL: return "";
        }

        return "";
    }

    inline std::string_view recommendedExternalQMScript(
        const settings::QMMethod method
    )
    {
        using enum settings::QMMethod;

        switch (method)
        {
            case DFTBPLUS: return dftbPlusScripts.front().name;
            case TURBOMOLE: return turbomoleScripts.front().name;

            case NONE:
            case ASEDFTBPLUS:
            case ASEXTB:
            case PYSCF:
            case MACE:
            case FENNOL: return "";
        }

        return "";
    }

    inline bool isExternalQMScript(
        const settings::QMMethod method,
        const std::string_view   script
    )
    {
        const auto scripts = externalQMScripts(method);

        return std::ranges::any_of(
            scripts,
            [script](const auto &candidate) { return candidate.name == script; }
        );
    }
}   // namespace cli

#endif   // _EXTERNAL_QM_SCRIPTS_HPP_
