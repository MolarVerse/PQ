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

#include "validation.hpp"

#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <format>
#include <iomanip>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#elif defined(__linux__)
#include <unistd.h>
#endif

#include "engine.hpp"
#include "exceptions.hpp"
#include "externalQMScripts.hpp"
#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "inputFileReader.hpp"
#include "manostatSettings.hpp"
#include "qmSettings.hpp"
#include "settings.hpp"
#include "thermostatSettings.hpp"

namespace
{
    class ScopedStreamRedirect
    {
       private:
        std::ostream   &_stream;
        std::streambuf *_original;

       public:
        ScopedStreamRedirect(std::ostream &stream, std::streambuf *replacement)
            : _stream(stream), _original(stream.rdbuf(replacement))
        {
        }

        ~ScopedStreamRedirect() { _stream.rdbuf(_original); }
    };

    cli::ValidationResult invalidResult(
        const std::string_view      inputFile,
        const std::string_view      message,
        const std::optional<size_t> lineNumber,
        const ValidationScope       scope
    )
    {
        return {
            .valid       = false,
            .inputFile   = std::string(inputFile),
            .scope       = scope,
            .diagnostics = {
                {cli::ValidationSeverity::ERROR,
                 std::string(message),
                 lineNumber}
            }
        };
    }

    void requireFile(
        const std::filesystem::path &fileName,
        const std::string_view       description
    )
    {
        if (!std::filesystem::is_regular_file(fileName))
            throw customException::InputFileException(
                std::format(
                    "{} \"{}\" does not exist or is not a regular file",
                    description,
                    fileName.string()
                )
            );
    }

    void requireDirectory(
        const std::filesystem::path &directoryName,
        const std::string_view       description
    )
    {
        if (!std::filesystem::is_directory(directoryName))
            throw customException::InputFileException(
                std::format(
                    "{} \"{}\" does not exist or is not a directory",
                    description,
                    directoryName.string()
                )
            );
    }

    bool isRemoteResource(const std::string_view value)
    {
        return value.starts_with("https://") || value.starts_with("http://");
    }

    std::filesystem::path executablePath()
    {
#if defined(_WIN32)
        auto buffer = std::vector<wchar_t>(1024);
        while (true)
        {
            const auto length = GetModuleFileNameW(
                nullptr,
                buffer.data(),
                static_cast<DWORD>(buffer.size())
            );
            if (length == 0)
                break;
            if (length < buffer.size() - 1)
                return std::filesystem::weakly_canonical(buffer.data());
            buffer.resize(buffer.size() * 2);
        }
#elif defined(__APPLE__)
        auto size = uint32_t{0};
        _NSGetExecutablePath(nullptr, &size);
        auto buffer = std::vector<char>(size);
        if (_NSGetExecutablePath(buffer.data(), &size) == 0)
            return std::filesystem::weakly_canonical(buffer.data());
#elif defined(__linux__)
        auto buffer = std::vector<char>(1024);
        while (true)
        {
            const auto length =
                readlink("/proc/self/exe", buffer.data(), buffer.size());
            if (length < 0)
                break;
            if (static_cast<size_t>(length) < buffer.size())
                return std::filesystem::weakly_canonical(
                    std::filesystem::path(
                        std::string(buffer.data(), static_cast<size_t>(length))
                    )
                );
            buffer.resize(buffer.size() * 2);
        }
#endif

        return {};
    }

    std::filesystem::path runtimeAssetPath(
        const std::filesystem::path &installedRelativePath,
        const std::filesystem::path &buildPath
    )
    {
        const auto executable = executablePath();
        if (executable.empty())
            return buildPath;

        std::error_code error;
        const auto      buildExecutableDirectory =
            std::filesystem::weakly_canonical(PQ_BUILD_EXECUTABLE_DIR, error);

        if (!error && executable.parent_path() == buildExecutableDirectory)
            return buildPath;

        return executable.parent_path().parent_path() / "share" / "PQ" /
               installedRelativePath;
    }

    std::filesystem::path bundledQMScriptPath(const std::string_view script)
    {
        return runtimeAssetPath(
            std::filesystem::path("scripts") / script,
            std::filesystem::path(PQ_BUILD_QM_SCRIPT_DIR) / script
        );
    }

    std::filesystem::path bundledSlakosPath(const settings::SlakosType type)
    {
        const auto name = settings::string(type);

        return runtimeAssetPath(
            std::filesystem::path("slakos") / name / "skfiles",
            std::filesystem::path(PQ_BUILD_SLAKOS_DIR) / name / "skfiles"
        );
    }

    void validateExternalQMScriptSelection()
    {
        using settings::QMSettings;
        using settings::Settings;

        if (!Settings::isQMActivated() || !QMSettings::isExternalQMRunner())
            return;

        const auto script         = QMSettings::getQMScript();
        const auto fullPathScript = QMSettings::getQMScriptFullPath();

        if (script.empty() && fullPathScript.empty())
            throw customException::InputFileException(
                "No qm_script provided. Please provide a qm_script in the "
                "input file."
            );

        if (!script.empty() && !fullPathScript.empty())
            throw customException::InputFileException(
                "\"qm_script\" and \"qm_script_full_path\" are mutually "
                "exclusive"
            );

        if (!script.empty() &&
            !cli::isExternalQMScript(QMSettings::getQMMethod(), script))
            throw customException::InputFileException(
                std::format(
                    "Bundled QM script \"{}\" is not available for {}",
                    script,
                    cli::externalQMProgramName(QMSettings::getQMMethod())
                )
            );
    }

    void validateInstalledExternalQMScript()
    {
        using settings::QMSettings;
        using settings::Settings;

        if (!Settings::isQMActivated() || !QMSettings::isExternalQMRunner())
            return;

        const auto script         = QMSettings::getQMScript();
        const auto fullPathScript = QMSettings::getQMScriptFullPath();

        if ((PQ_BUILD_STATIC || PQ_BUILD_WITH_SINGULARITY) &&
            fullPathScript.empty())
            throw customException::InputFileException(
                "This PQ build requires \"qm_script_full_path\" for "
                "external QM programs"
            );

        if (!fullPathScript.empty())
        {
            requireFile(fullPathScript, "QM script");
            return;
        }

        requireFile(bundledQMScriptPath(script), "Bundled QM script");

        const auto scripts  = cli::externalQMScripts(QMSettings::getQMMethod());
        const auto selected = std::ranges::find_if(
            scripts,
            [&script](const auto &candidate)
            { return candidate.name == script; }
        );

        if (selected != scripts.end() && !selected->requiredWorkingFile.empty())
            requireFile(
                selected->requiredWorkingFile,
                "Required QM working file"
            );
    }

    void validateInputDependencies(engine::Engine &engine)
    {
        using settings::FileSettings;
        using settings::ForceFieldSettings;

        if (engine.isConstraintsActivated() || ForceFieldSettings::isActive())
        {
            if (!FileSettings::isTopologyFileNameSet())
                throw customException::InputFileException(
                    "Topology file needed for requested simulation setup"
                );
        }

        if (ForceFieldSettings::isActive() &&
            !FileSettings::isParameterFileNameSet())
            throw customException::InputFileException(
                "Parameter file needed for requested simulation setup"
            );

        if (engine.getConstraints().isMShakeActive() &&
            FileSettings::getMShakeFileName().empty())
            throw customException::InputFileException(
                "M-SHAKE file needed for requested simulation setup"
            );

        validateExternalQMScriptSelection();
    }

    void validateEffectiveFiles(engine::Engine &engine)
    {
        using settings::FileSettings;
        using settings::ForceFieldSettings;
        using settings::ManostatSettings;
        using settings::ManostatType;
        using settings::QMMethod;
        using settings::QMSettings;
        using settings::Settings;
        using settings::SlakosType;

        requireFile(FileSettings::getStartFileName(), "Start file");

        if (FileSettings::isRingPolymerStartFileNameSet())
            requireFile(
                FileSettings::getRingPolymerStartFileName(),
                "Ring-polymer start file"
            );

        if (FileSettings::isIntraNonBondedFileNameSet())
            requireFile(
                FileSettings::getIntraNonBondedFileName(),
                "Intra non-bonded file"
            );

        if (Settings::isMMActivated() ||
            ManostatSettings::getManostatType() != ManostatType::NONE)
            requireFile(
                FileSettings::getMolDescriptorFileName(),
                "Moldescriptor file"
            );

        if (Settings::isMMActivated() &&
            !engine.isForceFieldNonCoulombicsActivated())
            requireFile(FileSettings::getGuffDatFileName(), "Guff file");

        if (engine.isConstraintsActivated() || ForceFieldSettings::isActive())
            requireFile(FileSettings::getTopologyFileName(), "Topology file");

        if (ForceFieldSettings::isActive())
            requireFile(FileSettings::getParameterFilename(), "Parameter file");

        if (FileSettings::isMShakeFileNameSet())
            requireFile(FileSettings::getMShakeFileName(), "M-SHAKE file");

        if (!Settings::isQMActivated())
            return;

        const auto method = QMSettings::getQMMethod();

        if (method == QMMethod::DFTBPLUS)
            requireFile(FileSettings::getDFTBFileName(), "DFTB setup file");

        if (method == QMMethod::ASEDFTBPLUS &&
            QMSettings::getSlakosType() != SlakosType::NONE)
        {
            const auto slakosType = QMSettings::getSlakosType();
            if (slakosType == SlakosType::CUSTOM)
                requireDirectory(
                    QMSettings::getSlakosPath(),
                    "Slater-Koster directory"
                );
            else
                requireDirectory(
                    bundledSlakosPath(slakosType),
                    "Built-in Slater-Koster directory"
                );
        }

        if (method == QMMethod::FENNOL)
            requireFile(QMSettings::getFennolModelPath(), "FeNNol model file");

        if (method == QMMethod::MACE &&
            QMSettings::getMaceModel() == settings::MaceModel::CUSTOM)
        {
            const auto modelPath = QMSettings::getMaceModelPath();
            if (!isRemoteResource(modelPath))
                requireFile(modelPath, "MACE model file");
        }

        validateInstalledExternalQMScript();
    }

    void validateCompiledCapabilities()
    {
        if (PQ_BUILD_WITH_ASE || !settings::Settings::isQMActivated())
            return;

        const auto method = settings::QMSettings::getQMMethod();
        if (method == settings::QMMethod::ASEDFTBPLUS ||
            method == settings::QMMethod::ASEXTB ||
            method == settings::QMMethod::FENNOL ||
            method == settings::QMMethod::MACE)
            throw customException::InputFileException(
                std::format(
                    "QM method {} requires ASE support, but this PQ build "
                    "does not include it",
                    settings::string(method)
                )
            );
    }

    void appendWarnings(
        const input::InputFileReader &reader,
        cli::ValidationResult        &result
    )
    {
        using settings::ManostatSettings;
        using settings::ManostatType;
        using settings::ThermostatSettings;
        using settings::ThermostatType;

        if (reader.getKeywordSet("mace_model_size"))
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "\"mace_model_size\" is deprecated; use \"mace_model\"",
                 std::nullopt}
            );

        if (ThermostatSettings::getThermostatType() ==
                ThermostatType::NOSE_HOOVER &&
            ThermostatSettings::getNoseHooverCouplingFrequency() == 0.0)
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero Nose-Hoover coupling frequency disables thermostat "
                 "coupling",
                 std::nullopt}
            );

        if (ThermostatSettings::getThermostatType() ==
                ThermostatType::LANGEVIN &&
            ThermostatSettings::getFriction() == 0.0)
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero Langevin friction disables thermostat coupling",
                 std::nullopt}
            );

        if (ManostatSettings::getManostatType() != ManostatType::NONE &&
            ManostatSettings::getCompressibility() == 0.0)
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero compressibility disables cell response",
                 std::nullopt}
            );
    }

    void writeJsonString(std::ostream &output, const std::string_view value)
    {
        output << '"';

        for (const auto character : value)
        {
            switch (character)
            {
                case '"': output << "\\\""; break;
                case '\\': output << "\\\\"; break;
                case '\b': output << "\\b"; break;
                case '\f': output << "\\f"; break;
                case '\n': output << "\\n"; break;
                case '\r': output << "\\r"; break;
                case '\t': output << "\\t"; break;
                default:
                    if (static_cast<unsigned char>(character) < 0x20)
                    {
                        const auto flags = output.flags();
                        const auto fill  = output.fill();
                        output << "\\u" << std::hex << std::setw(4)
                               << std::setfill('0')
                               << static_cast<unsigned int>(
                                      static_cast<unsigned char>(character)
                                  );
                        output.flags(flags);
                        output.fill(fill);
                    }
                    else
                        output << character;
            }
        }

        output << '"';
    }

    std::string_view string(const cli::ValidationSeverity severity)
    {
        return severity == cli::ValidationSeverity::WARNING ? "warning"
                                                            : "error";
    }

    std::string_view string(const ValidationScope scope)
    {
        return scope == ValidationScope::PORTABLE ? "portable" : "installed";
    }
}   // namespace

cli::ValidationResult cli::validateInputFile(
    const std::string_view inputFile,
    const ValidationScope  scope
)
{
    auto engine = std::unique_ptr<engine::Engine>();

    std::ostringstream   parserOutput;
    ScopedStreamRedirect redirect(std::cout, parserOutput.rdbuf());

    try
    {
        input::readJobType(std::string(inputFile), engine);

        input::InputFileReader reader(
            inputFile,
            *engine,
            scope == ValidationScope::INSTALLED,
            false
        );
        reader.read();
        reader.postProcess();
        reader.validateInputConfiguration();
        validateInputDependencies(*engine);
        if (scope == ValidationScope::INSTALLED)
        {
            validateCompiledCapabilities();
            validateEffectiveFiles(*engine);
        }

        auto result = ValidationResult{
            .valid     = true,
            .inputFile = std::string(inputFile),
            .scope     = scope
        };
        appendWarnings(reader, result);
        return result;
    }
    catch (const customException::CustomException &exception)
    {
        return invalidResult(
            inputFile,
            exception.getMessage(),
            exception.getLineNumber(),
            scope
        );
    }
}

void cli::writeValidationJson(
    const ValidationResult &result,
    std::ostream           &output
)
{
    const auto flags = output.flags();

    output << "{\n"
           << "  \"schema\": \"pq.validation\",\n"
           << "  \"schema_version\": 1,\n"
           << "  \"valid\": " << std::boolalpha << result.valid << ",\n"
           << "  \"input\": ";
    writeJsonString(output, result.inputFile);
    output << ",\n"
           << "  \"scope\": ";
    writeJsonString(output, string(result.scope));
    output << ",\n"
           << "  \"diagnostics\": [";

    if (result.diagnostics.empty())
    {
        output << "]\n"
               << "}\n";
        output.flags(flags);
        return;
    }

    for (size_t index = 0; index < result.diagnostics.size(); ++index)
    {
        const auto &diagnostic = result.diagnostics[index];
        output << (index == 0 ? "\n" : ",\n") << "    {\n"
               << "      \"severity\": ";
        writeJsonString(output, string(diagnostic.severity));
        output << ",\n"
               << "      \"message\": ";
        writeJsonString(output, diagnostic.message);
        output << ",\n"
               << "      \"file\": ";
        writeJsonString(output, result.inputFile);
        output << ",\n"
               << "      \"line\": ";
        if (diagnostic.lineNumber.has_value())
            output << diagnostic.lineNumber.value();
        else
            output << "null";
        output << "\n"
               << "    }";
    }

    output << '\n'
           << "  ]\n"
           << "}\n";
    output.flags(flags);
}

void cli::writeValidationText(
    const ValidationResult &result,
    std::ostream           &output,
    std::ostream           &error
)
{
    if (result.valid)
    {
        output << "Valid PQ input: " << result.inputFile << '\n';

        for (const auto &diagnostic : result.diagnostics)
            error << "Warning: " << diagnostic.message << '\n';

        return;
    }

    error << "Invalid PQ input: " << result.diagnostics.front().message;
    if (result.diagnostics.front().lineNumber.has_value())
    {
        const auto line = std::format(
            "line {}",
            result.diagnostics.front().lineNumber.value()
        );
        if (!result.diagnostics.front().message.contains(line))
            error << " (" << line << ')';
    }
    error << '\n';
}
