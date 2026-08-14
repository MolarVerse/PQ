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
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <string_view>

#include "engine.hpp"
#include "exceptions.hpp"
#include "executablePath.hpp"
#include "externalQMScripts.hpp"
#include "fileSettings.hpp"
#include "forceFieldSettings.hpp"
#include "inputFileReader.hpp"
#include "jsonOutput.hpp"
#include "manostatSettings.hpp"
#include "mathUtilities.hpp"
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

        ScopedStreamRedirect(const ScopedStreamRedirect &)            = delete;
        ScopedStreamRedirect &operator=(const ScopedStreamRedirect &) = delete;
        ScopedStreamRedirect(ScopedStreamRedirect &&)                 = delete;
        ScopedStreamRedirect &operator=(ScopedStreamRedirect &&)      = delete;

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
        {
            throw customException::InputFileException(
                std::format(
                    "{} \"{}\" does not exist or is not a regular file",
                    description,
                    fileName.string()
                )
            );
        }
    }

    void requireDirectory(
        const std::filesystem::path &directoryName,
        const std::string_view       description
    )
    {
        if (!std::filesystem::is_directory(directoryName))
        {
            throw customException::InputFileException(
                std::format(
                    "{} \"{}\" does not exist or is not a directory",
                    description,
                    directoryName.string()
                )
            );
        }
    }

    bool isRemoteResource(const std::string_view value)
    {
        return value.starts_with("https://") || value.starts_with("http://");
    }

    std::filesystem::path runtimeAssetPath(
        const std::filesystem::path &installedRelativePath,
        const std::filesystem::path &buildPath
    )
    {
        const auto executable = utilities::executablePath();
        if (executable.empty())
            return buildPath;

        std::error_code error;
        const auto      buildExecutableDirectory =
            std::filesystem::weakly_canonical(PQ_BUILD_EXECUTABLE_DIR, error);

        if (!error && executable.parent_path() == buildExecutableDirectory)
            return buildPath;

        return utilities::installedDataPath(installedRelativePath);
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
        {
            throw customException::InputFileException(
                "No qm_script provided. Please provide a qm_script in the "
                "input file."
            );
        }

        if (!script.empty() && !fullPathScript.empty())
        {
            throw customException::InputFileException(
                "\"qm_script\" and \"qm_script_full_path\" are mutually "
                "exclusive"
            );
        }

        if (!script.empty() &&
            !cli::isExternalQMScript(QMSettings::getQMMethod(), script))
        {
            throw customException::InputFileException(
                std::format(
                    "Bundled QM script \"{}\" is not available for {}",
                    script,
                    cli::externalQMProgramName(QMSettings::getQMMethod())
                )
            );
        }
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
        {
            throw customException::InputFileException(
                "This PQ build requires \"qm_script_full_path\" for "
                "external QM programs"
            );
        }

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
        {
            requireFile(
                selected->requiredWorkingFile,
                "Required QM working file"
            );
        }
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

        if (engine.getConstraints()->isMShakeActive() &&
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
        {
            requireFile(
                FileSettings::getRingPolymerStartFileName(),
                "Ring-polymer start file"
            );
        }

        if (FileSettings::isIntraNonBondedFileNameSet())
        {
            requireFile(
                FileSettings::getIntraNonBondedFileName(),
                "Intra non-bonded file"
            );
        }

        if (Settings::isMMActivated() ||
            ManostatSettings::getManostatType() != ManostatType::NONE)
        {
            requireFile(
                FileSettings::getMolDescriptorFileName(),
                "Moldescriptor file"
            );
        }

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
            {
                requireDirectory(
                    QMSettings::getSlakosPath(),
                    "Slater-Koster directory"
                );
            }
            else
            {
                requireDirectory(
                    bundledSlakosPath(slakosType),
                    "Built-in Slater-Koster directory"
                );
            }
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
        {
            throw customException::InputFileException(
                std::format(
                    "QM method {} requires ASE support, but this PQ build "
                    "does not include it",
                    settings::string(method)
                )
            );
        }
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
        {
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "\"mace_model_size\" is deprecated; use \"mace_model\"",
                 std::nullopt}
            );
        }
        if (ThermostatSettings::getThermostatType() ==
                ThermostatType::NOSE_HOOVER &&
            utilities::isZero(
                ThermostatSettings::getNoseHooverCouplingFrequency()
            ))
        {
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero Nose-Hoover coupling frequency disables thermostat "
                 "coupling",
                 std::nullopt}
            );
        }

        if (ThermostatSettings::getThermostatType() ==
                ThermostatType::LANGEVIN &&
            utilities::isZero(ThermostatSettings::getFriction()))
        {
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero Langevin friction disables thermostat coupling",
                 std::nullopt}
            );
        }

        if (ManostatSettings::getManostatType() != ManostatType::NONE &&
            utilities::isZero(ManostatSettings::getCompressibility()))
        {
            result.diagnostics.push_back(
                {cli::ValidationSeverity::WARNING,
                 "A zero compressibility disables cell response",
                 std::nullopt}
            );
        }
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

cli::ValidationResult cli::validateInputFile(std::string_view inputFile)
{
    return validateInputFile(inputFile, ValidationScope::INSTALLED);
}

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
            .valid       = true,
            .inputFile   = std::string(inputFile),
            .scope       = scope,
            .diagnostics = {}
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
    auto json = JsonWriter(output);
    json.beginObject();
    json.value("schema", "pq.validation");
    json.value("schema_version", 1);
    json.value("valid", result.valid);
    json.value("input", result.inputFile);
    json.value("scope", string(result.scope));
    json.beginArray("diagnostics");

    for (const auto &diagnostic : result.diagnostics)
    {
        json.beginObject();
        json.value("severity", string(diagnostic.severity));
        json.value("message", diagnostic.message);
        json.value("file", result.inputFile);
        if (diagnostic.lineNumber.has_value())
            json.value("line", diagnostic.lineNumber.value());
        else
            json.value("line", nullptr);
        json.endObject();
    }

    json.endArray();
    json.endObject();
    output << '\n';
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
