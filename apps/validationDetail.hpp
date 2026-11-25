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

#ifndef APPS__VALIDATION_DETAIL_HPP
#define APPS__VALIDATION_DETAIL_HPP

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mstd/utils.hpp>
#include <optional>
#include <ostream>
#include <streambuf>
#include <string_view>

#include "enums/jobtype.hpp"
#include "enums/qm.hpp"

enum class ValidationScope : std::uint8_t;   // forward declaration
class QMSettings;                            // forward declaration
class ExternalQMSettings;                    // forward declaration
struct Settings;                             // forward declaration

namespace input
{
    class InputFileReader;   // forward declaration
}   // namespace input

namespace cli
{
    enum class ValidationSeverity : std::uint8_t;   // forward declaration
    struct ValidationResult;                        // forward declaration
}   // namespace cli

namespace cli::detail
{
    /**
     * @brief redirects a stream to another buffer until destruction
     */
    class ScopedStreamRedirect
    {
       private:
        std::ostream   &_stream;
        std::streambuf *_original;

       public:
        ScopedStreamRedirect(std::ostream &stream, std::streambuf *replacement);

        MSTD_NO_COPY(ScopedStreamRedirect);
        MSTD_NO_MOVE(ScopedStreamRedirect);

        ~ScopedStreamRedirect();
    };

    /**
     * @brief installed and build-tree location of a runtime asset
     */
    struct Files
    {
        std::filesystem::path installedRelativePath;
        std::filesystem::path buildPath;
    };

    [[nodiscard]] cli::ValidationResult invalidResult(
        std::string_view      inputFile,
        std::string_view      message,
        std::optional<size_t> lineNumber,
        ValidationScope       scope
    );

    void requireFile(
        const std::filesystem::path &fileName,
        std::string_view             description
    );
    void requireDirectory(
        const std::filesystem::path &directoryName,
        std::string_view             description
    );

    [[nodiscard]] bool isRemoteResource(std::string_view value);

    [[nodiscard]] std::filesystem::path runtimeAssetPath(const Files &files);
    [[nodiscard]] std::filesystem::path bundledQMScriptPath(
        std::string_view script
    );
    [[nodiscard]] std::filesystem::path bundledSlakosPath(SlakosType type);

    void validateExternalQMScriptSelection(
        const QMSettings         &qmSettings,
        const ExternalQMSettings &externalQMSettings,
        JobType                   jobType
    );
    void validateInstalledExternalQMScript(
        const QMSettings         &qmSettings,
        const ExternalQMSettings &externalQMSettings,
        JobType                   jobType
    );
    void validateInputDependencies(const Settings &settings);
    void validateEffectiveFiles(const Settings &settings);
    void validateCompiledCapabilities(
        const QMSettings &qmSettings,
        JobType           jobType
    );

    void appendWarnings(
        const input::InputFileReader &reader,
        cli::ValidationResult        &result,
        const Settings               &settings
    );

    [[nodiscard]] std::string_view string(cli::ValidationSeverity severity);
    [[nodiscard]] std::string_view string(ValidationScope scope);
}   // namespace cli::detail

#endif   // APPS__VALIDATION_DETAIL_HPP
