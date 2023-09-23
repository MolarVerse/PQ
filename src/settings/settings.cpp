#include "settings.hpp"

#include "stringUtilities.hpp"

using settings::Settings;

/**
 * @brief sets the jobtype to enum in settings
 *
 * @param jobtype
 */
void Settings::setJobtype(const std::string_view jobtype)
{
    const auto jobtypeToLower = utilities::toLowerCopy(jobtype);

    if (jobtypeToLower == "mmmd")
        _jobtype = settings::JobType::MM_MD;
    else if (jobtypeToLower == "qmmd")
        _jobtype = settings::JobType::QM_MD;
    else if (jobtypeToLower == "ring_polymer_qmmd")
        _jobtype = settings::JobType::RING_POLYMER_QM_MD;
    else
        _jobtype = settings::JobType::NONE;
}