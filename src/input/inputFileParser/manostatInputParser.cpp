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

#include "manostatInputParser.hpp"

#include <limits>
#include <optional>

#include "constants/conversionFactors.hpp"
#include "customValidator.hpp"
#include "defaults.hpp"
#include "inputKeyAdapter.hpp"
#include "keyMetaData.hpp"
#include "manostatSettings.hpp"
#include "rangeValidator.hpp"
#include "references.hpp"
#include "referencesOutput.hpp"

namespace input
{

    /**
     * @brief Construct a new Input File Parser Manostat:: Input File Parser
     * Manostat object
     *
     * @details following keywords are added to the _keywordFuncMap,
     * _keywordRequiredMap and _keywordCountMap: 1) manostat "<string>" 2)
     * pressure
     * "<double>" (only required if manostat is not none) 3) p_relaxation
     * "<double>" 4) compressibility "<double>"
     */
    ManostatInputParser::ManostatInputParser()
    {
        addManostatKey();
        addPressureKey();
        addManostatRelaxationTimeKey();
        addCompressibilityKey();
        addIsotropyKey();
        addFixedAxisKey();
    }

    /**
     * @brief Add the manostat key to the input parser
     *
     * @details This function registers the "manostat" key with the input
     * parser, including its metadata, default value, and on-set behavior.
     */
    void ManostatInputParser::addManostatKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "manostat",
            .title = "Manostat type",
            .description =
                "Specifies the type of manostat to be used in the simulation",
        };

        const auto setValue = [](ManostatType manostatType)
        {
            switch (manostatType)
            {
                case ManostatType::BERENDSEN:
                    references::ReferencesOutput::addReferenceFile(
                        references::BERENDSEN_FILE
                    );
                    break;
                case ManostatType::STOCHASTIC_RESCALING:
                    references::ReferencesOutput::addReferenceFile(
                        references::STOCHASTIC_RESCALING_FILE
                    );
                    break;
                case ManostatType::NONE: break;
            }
            settings::ManostatSettings::setManostatType(manostatType);
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<ManostatType>{
                .metadata     = metaData,
                .defaultValue = ManostatType::NONE,
                .onSet        = setValue,
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the pressure key to the input parser
     *
     * @details This function registers the "pressure" key with the input
     * parser, including its metadata, default value, and on-set behavior.
     */
    void ManostatInputParser::addPressureKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "pressure",
            .title       = "Pressure",
            .description = "Specifies the pressure used in the simulation",
        };

        const auto setValue = [](double pressure)
        { settings::ManostatSettings::setTargetPressure(pressure); };

        const RangeValidator<double> validator{std::nullopt, std::nullopt};

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata   = metaData,
                .onSet      = setValue,
                .validators = {makeShared(validator)},
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the manostat relaxation time key to the input parser
     *
     * @details This function registers the "manostat_relaxation_time" key with
     * the input parser, including its metadata, default value, and on-set
     * behavior.
     */
    void ManostatInputParser::addManostatRelaxationTimeKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "p_relaxation",
            .title       = "Manostat Relaxation Time",
            .description = "Specifies the relaxation time of the manostat",
        };

        const auto setValue = [](double relaxationTime)
        { settings::ManostatSettings::setTauManostat(relaxationTime); };

        const auto maxValidator = CustomValidator<double>{
            [](const double &value)
            { return value <= std::numeric_limits<double>::max() / PS_TO_FS; },
            "Relaxation time of manostat is too large to represent in "
            "femtoseconds"
        };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata     = metaData,
                .defaultValue = 1.0,
                .onSet        = setValue,
                .validators =
                    {makeShared(PositiveGTDoubleValidator),
                     makeShared(maxValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the compressibility key to the input parser
     *
     * @details This function registers the "compressibility" key with
     * the input parser, including its metadata, default value, and on-set
     * behavior.
     */
    void ManostatInputParser::addCompressibilityKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "compressibility",
            .title = "Compressibility",
            .description =
                "Specifies the isothermal compressibility used in the "
                "simulation",
        };

        const auto setValue = [](double compressibility)
        { settings::ManostatSettings::setCompressibility(compressibility); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<double>{
                .metadata     = metaData,
                .defaultValue = defaults::COMPRESSIBILITY_WATER_DEFAULT,
                .onSet        = setValue,
                .validators   = {makeShared(PositiveGTDoubleValidator)}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Add the isotropy key to the input parser
     *
     * @details This function registers the "isotropy" key with
     * the input parser, including its metadata, default value, and on-set
     * behavior.
     */
    void ManostatInputParser::addIsotropyKey()
    {
        const auto metaData = KeyMetadata{
            .name  = "isotropy",
            .title = "Isotropy",
            .description =
                "Specifies the isotropy of the manostat (isotropic, "
                "semi-isotropic, anisotropic, full_anisotropic)",
        };

        const auto setValue = [](Isotropy isotropy)
        { settings::ManostatSettings::setIsotropy(isotropy); };

        auto &key = _getRegistry().registerKey(
            KeyRegistry<Isotropy>{
                .metadata     = metaData,
                .defaultValue = Isotropy::ISOTROPIC,
                .onSet        = setValue,
                .validators   = {}
            }
        );

        addKeyword(metaData.name, adapt(key), false);
    }

    /**
     * @brief Adds the fixed axis key to the input parser
     *
     * This key allows the user to specify the fixed axis in the simulation.
     */
    void ManostatInputParser::addFixedAxisKey()
    {
        const auto metaData = KeyMetadata{
            .name        = "fixed_axis",
            .title       = "Fixed axis",
            .description = "Specifies the fixed axis in the simulation",
        };

        const auto setValue = [](FixedAxis fixedAxis)
        { settings::ManostatSettings::setFixedAxis(fixedAxis); };

        auto &fixedAxisKey = _getRegistry().registerKey(
            KeyRegistry<FixedAxis>{
                .metadata     = metaData,
                .defaultValue = FixedAxis::NONE,
                .onSet        = setValue,
            }
        );

        addKeyword(metaData.name, adapt(fixedAxisKey), false);
    }

}   // namespace input
