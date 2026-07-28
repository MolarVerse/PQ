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

    inline constexpr auto _DFTBPLUS_SCRIPTS_ = std::array{ExternalQMScriptInfo{
        "dftbplus_periodic_stress",
        "DFTB+ periodic stress",
        "dftb_file",
        ""
    }};
    inline constexpr auto _PYSCF_SCRIPTS_    = std::array{
        ExternalQMScriptInfo{"pyscf_hf.py", "UHF / STO-3G", "", ""},
        ExternalQMScriptInfo{"pyscf_mp2.py", "UMP2 / 6-311++G**", "", ""}
    };
    inline constexpr auto _TURBOMOLE_SCRIPTS_ = std::array{ExternalQMScriptInfo{
        "turbomole_rimp2",
        "RI-MP2",
        "",
        "tm_define.template"
    }};

    inline constexpr auto _EXTERNAL_QM_METHODS_ = std::array{
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
            case DFTBPLUS: return _DFTBPLUS_SCRIPTS_;
            case PYSCF: return _PYSCF_SCRIPTS_;
            case TURBOMOLE: return _TURBOMOLE_SCRIPTS_;

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
            case DFTBPLUS: return _DFTBPLUS_SCRIPTS_.front().name;
            case TURBOMOLE: return _TURBOMOLE_SCRIPTS_.front().name;

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
