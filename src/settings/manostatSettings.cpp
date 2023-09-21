#include "manostatSettings.hpp"

#include "stringUtilities.hpp"

using settings::ManostatSettings;

/**
 * @brief return string of manostatType
 *
 * @param manostatType
 */
std::string settings::string(const settings::ManostatType &manostatType)
{
    switch (manostatType)
    {
    case settings::ManostatType::BERENDSEN: return "berendsen";
    default: return "none";
    }
}

/**
 * @brief sets the manostatType to enum in settings
 *
 * @param manostatType
 */
void ManostatSettings::setManostatType(const std::string_view &manostatType)
{
    const auto manostatTypeToLower = utilities::toLowerCopy(manostatType);

    if (manostatTypeToLower == "berendsen")
        _manostatType = settings::ManostatType::BERENDSEN;
    else
        _manostatType = settings::ManostatType::NONE;
}