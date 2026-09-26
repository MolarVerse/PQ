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

#include "filesInputParser.hpp"

#include <mstd/file.hpp>
#include <utility>

#include "fileSettings.hpp"   // for FileSettings
#include "inputKeyAdapter.hpp"
#include "keyRegistry.hpp"

using namespace input;
using namespace exc;
using namespace settings;

namespace
{
    /**
     * @brief Add a file keyword to the parser
     *
     * @tparam T The type of the file
     * @param metaData The metadata for the key
     * @param setValue The function to set the value
     * @param parser The input file parser
     * @param registry The key registry
     * @param isRequired Flag indicating whether the key is required
     */
    template <typename T>
    void addFileKeywordImpl(
        const KeyMetadata &metaData,
        const auto        &setValue,
        auto              &parser,
        auto              &registry,
        bool               isRequired
    )
    {
        auto keyRegistry = [&, metaData, setValue]<typename U>()
        { return KeyRegistry<U>{.metadata = metaData, .onSet = setValue}; };

        parser.addKeyword(
            metaData.name,
            adapt(registry.registerKey(keyRegistry.template operator()<T>())),
            isRequired
        );
    }

    /**
     * @brief Add a file keyword to the parser, choosing between std::string and
     * File based on validation flag
     *
     * @param metaData The metadata for the key
     * @param setValue The function to set the value
     * @param parser The input file parser
     * @param registry The key registry
     * @param isRequired Flag indicating whether the key is required
     * @param validateFilePaths Flag indicating whether to validate file paths
     *
     */
    void addFileKeyword(
        const KeyMetadata &metaData,
        const auto        &setValue,
        auto              &parser,
        auto              &registry,
        bool               isRequired,
        bool               validateFilePaths
    )
    {
        if (!validateFilePaths)
        {
            addFileKeywordImpl<std::string>(
                metaData,
                setValue,
                parser,
                registry,
                isRequired
            );
        }
        else
        {
            addFileKeywordImpl<mstd::File>(
                metaData,
                setValue,
                parser,
                registry,
                isRequired
            );
        }
    }
}   // namespace

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser
 * Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) intra-nonBonded_file "<string>"
 * 2) topology_file "<string>" 3) parameter_file "<string>" 4) start_file
 * "<string>" (required) 5) rpmd_start_file "<string>" 6) moldescriptor_file
 * "<string>" 7) guff_path "<string>" (deprecated) 8) guff_file "<string>" 9)
 * mshake_file "<string>" 10) dftb_file "<string>" 11) turbomole_file "<string>"
 *
 * @param intraNonBonded
 */
FilesInputParser::FilesInputParser(
    std::shared_ptr<intraNonBonded::IntraNonBonded> intraNonBonded
)
    : FilesInputParser(std::move(intraNonBonded), true)
{
}

/**
 * @brief Construct a new Input File Parser Non Coulomb Type:: Input File Parser
 * Non Coulomb Type object
 *
 * @details following keywords are added to the _keywordFuncMap,
 * _keywordRequiredMap and _keywordCountMap: 1) intra-nonBonded_file "<string>"
 * 2) topology_file "<string>" 3) parameter_file "<string>" 4) start_file
 * "<string>" (required) 5) rpmd_start_file "<string>" 6) moldescriptor_file
 * "<string>" 7) guff_path "<string>" (deprecated) 8) guff_file "<string>" 9)
 * mshake_file "<string>" 10) dftb_file "<string>" 11) turbomole_file "<string>"
 *
 * @param intraNonBonded
 * @param validateFilePaths
 */
FilesInputParser::FilesInputParser(
    std::shared_ptr<intraNonBonded::IntraNonBonded> intraNonBonded,
    bool                                            validateFilePaths
)
    : InputFileParser(),
      _intraNonBonded(std::move(intraNonBonded)),
      _validateFilePaths(validateFilePaths)
{
    addIntraNonBondedFileKey();
    addTopologyFileKey();
    addParameterFileKey();
    addStartFileKey();
    addRingPolymerStartFileKey();
    addMoldescriptorFileKey();
    addGuffDatFileKey();
    addGuffPathKey();
    addMShakeFileKey();
    addDFTBFileKey();
    addTMFileKey();
}

/**
 * @brief Adds the intra-nonBonded_file keyword to the parser
 *
 * This keyword specifies the file containing intra non bonded combinations.
 */
void FilesInputParser::addIntraNonBondedFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "intra-nonBonded_file",
        .title       = "Intra Non Bonded File",
        .description = "File containing intra non bonded combinations"
    };

    const auto setValue =
        [intraNonBonded = _intraNonBonded]<typename T>(const T &value) -> void
    {
        intraNonBonded->activate();

        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setIntraNonBondedFileName(value);
        else
            FileSettings::setIntraNonBondedFileName(value.fileName());

        FileSettings::setIsIntraNonBondedFileNameSet();
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the topology_file keyword to the parser
 *
 * This keyword specifies the file containing the topology of the system.
 */
void FilesInputParser::addTopologyFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "topology_file",
        .title       = "Topology File",
        .description = "File containing the topology of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setTopologyFileName(value);
        else
            FileSettings::setTopologyFileName(value.fileName());

        FileSettings::setIsTopologyFileNameSet();
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the parameter_file keyword to the parser
 *
 * This keyword specifies the file containing the parameters of the system.
 */
void FilesInputParser::addParameterFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "parameter_file",
        .title       = "Parameter File",
        .description = "File containing the parameters of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setParameterFileName(value);
        else
            FileSettings::setParameterFileName(value.fileName());

        FileSettings::setIsParameterFileNameSet();
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the start_file keyword to the parser
 *
 * This keyword specifies the file containing the start configuration of the
 * system.
 */
void FilesInputParser::addStartFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "start_file",
        .title       = "Start File",
        .description = "File containing the start configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setStartFileName(value);
        else
            FileSettings::setStartFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        true,
        _validateFilePaths
    );
}

/**
 * @brief Adds the ring_polymer_start_file keyword to the parser
 *
 * This keyword specifies the file containing the ring polymer start
 * configuration of the system.
 */
void FilesInputParser::addRingPolymerStartFileKey()
{
    const auto metaData = KeyMetadata{
        .name  = "rpmd_start_file",
        .title = "Ring Polymer Start File",
        .description =
            "File containing the ring polymer start configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setRingPolymerStartFileName(value);
        else
            FileSettings::setRingPolymerStartFileName(value.fileName());

        FileSettings::setIsRingPolymerStartFileNameSet();
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the moldescriptor_file keyword to the parser
 *
 * This keyword specifies the file containing the molecular descriptor of the
 * system.
 */
void FilesInputParser::addMoldescriptorFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "moldescriptorFile_name",
        .title       = "Mol Descriptor File",
        .description = "File containing the molecular descriptor of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setMolDescriptorFileName(value);
        else
            FileSettings::setMolDescriptorFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the guffdat_file keyword to the parser
 *
 * This keyword specifies the file containing the guff dat configuration of the
 * system.
 */
void FilesInputParser::addGuffDatFileKey()
{
    const auto metaData = KeyMetadata{
        .name  = "guffdat_file",
        .title = "Guff Dat File",
        .description =
            "File containing the guff dat configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setGuffDatFileName(value);
        else
            FileSettings::setGuffDatFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the mshake_file keyword to the parser
 *
 * This keyword specifies the file containing the MShake configuration of the
 * system.
 */
void FilesInputParser::addMShakeFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "mshake_file",
        .title       = "MShake File",
        .description = "File containing the MShake configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setMShakeFileName(value);
        else
            FileSettings::setMShakeFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the dftb_file keyword to the parser
 *
 * This keyword specifies the file containing the DFTB configuration of the
 * system.
 */
void FilesInputParser::addDFTBFileKey()
{
    const auto metaData = KeyMetadata{
        .name        = "dftb_file",
        .title       = "DFTB File",
        .description = "File containing the DFTB configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setDFTBFileName(value);
        else
            FileSettings::setDFTBFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the tm_file keyword to the parser
 *
 * @details registers the "tm_file" key and associates it with the appropriate
 * callback that sets the Turbomole file name in the settings.
 */
void FilesInputParser::addTMFileKey()
{
    const auto metaData = KeyMetadata{
        .name  = "turbomole_file",
        .title = "Turbomole File",
        .description =
            "File containing the Turbomole configuration of the system"
    };

    const auto setValue = []<typename T>(const T &value) -> void
    {
        if constexpr (std::is_same_v<T, std::string>)
            FileSettings::setTMFileName(value);
        else
            FileSettings::setTMFileName(value.fileName());
    };

    addFileKeyword(
        metaData,
        setValue,
        *this,
        _getRegistry(),
        false,
        _validateFilePaths
    );
}

/**
 * @brief Adds the deprecated guff_path keyword to the parser
 *
 * @details registers the "guff_path" key as deprecated and associates it
 * with the appropriate callback that throws an exception when used.
 */
void FilesInputParser::addGuffPathKey()
{
    const auto deprecatedKey = DeprecatedInputKey(
        "guff_path",
        "The \"guff_path\" keyword is deprecated. Please use \"guffdat_file\" "
        "instead."
    );

    _getRegistry().registerDeprecatedKey(deprecatedKey);

    addKeyword(deprecatedKey.getKey(), adapt(deprecatedKey), false);
}
