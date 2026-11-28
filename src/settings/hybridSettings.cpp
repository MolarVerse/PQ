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

#include "hybridSettings.hpp"

#include <format>

#include "exceptions.hpp"
#include "fileSettings.hpp"

#ifdef PYTHON_ENABLED
#include "selection.hpp"
#endif

/**
 * @brief resolve the pending innerRegionCenter selection expression
 *
 * @param fileSettings
 * @param jobType
 *
 * @throw exc::InputFileException if the selection is invalid or empty
 */
void HybridSettings::finalize(const FileSettings &fileSettings, JobType jobType)
{
    _isActive = isHybridJobtype(jobType);

    if (!_innerRegionCenterExpression.has_value())
        return;

    const auto expression = *_innerRegionCenterExpression;
    _innerRegionCenterExpression.reset();

#ifdef PYTHON_ENABLED
    auto selection = pq_python::select(
        expression,
        fileSettings.get_startFile(),
        fileSettings.get_molDescriptorFile()
    );

    std::ranges::sort(selection);
    const auto [first, last] = std::ranges::unique(selection);
    selection.erase(first, last);

    if (selection.empty() || selection.front() < 0)
        throw exc::InputFileException(
            std::format(
                "Invalid selection \"{}\" for \"inner_region_center\": "
                "the selection is empty or contains negative atom indices",
                expression
            )
        );

    _innerRegionCenter.setByUser(
        std::vector<size_t>(selection.begin(), selection.end())
    );
#else
    static_cast<void>(fileSettings);
    throw exc::InputFileException(
        std::format(
            "Selection \"{}\" for \"inner_region_center\" requires "
            "PQ to be built with Python bindings",
            expression
        )
    );
#endif
}

/**
 * @brief check if the hybrid settings are active based on the job type
 *
 * @return bool
 */
bool HybridSettings::isActive() const { return _isActive; }
