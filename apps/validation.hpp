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

#ifndef _CLI_VALIDATION_HPP_

#define _CLI_VALIDATION_HPP_

#include <cstddef>
#include <iosfwd>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "commandLineArgs.hpp"

namespace cli
{
    enum class ValidationSeverity
    {
        WARNING,
        ERROR
    };

    struct ValidationDiagnostic
    {
        ValidationSeverity    severity = ValidationSeverity::ERROR;
        std::string           message;
        std::optional<size_t> lineNumber;
    };

    struct ValidationResult
    {
        bool                              valid = true;
        std::string                       inputFile;
        ValidationScope                   scope = ValidationScope::INSTALLED;
        std::vector<ValidationDiagnostic> diagnostics;
    };

    [[nodiscard]] ValidationResult validateInputFile(
        std::string_view inputFile,
        ValidationScope  scope = ValidationScope::INSTALLED
    );

    void writeValidationJson(
        const ValidationResult &result,
        std::ostream           &output
    );
    void writeValidationText(
        const ValidationResult &result,
        std::ostream           &output,
        std::ostream           &error
    );
}   // namespace cli

#endif   // _CLI_VALIDATION_HPP_
