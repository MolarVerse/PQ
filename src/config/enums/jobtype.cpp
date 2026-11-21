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

#include "enums/jobtype.hpp"

/**
 * @brief Returns true if the jobtype does not use any MM type simulations
 *
 * @return true/false if the jobtype does not use any MM type simulations
 *
 */
bool isQMOnlyJobtype(JobType jobtype)
{
    return (jobtype & JobType::QM_MM) == JobType::QM;
}

/**
 * @brief Returns true if the jobtype does not use any QM type simulations
 *
 * @return true/false if the jobtype does not use any QM type simulations
 *
 */
bool isMMOnlyJobtype(JobType jobtype)
{
    return (jobtype & JobType::QM_MM) == JobType::MM;
}

/**
 * @brief Returns true if the jobtype is a hybrid type simulation
 *
 * @return true/false if the jobtype is a hybrid type simulation
 *
 */
bool isHybridJobtype(JobType jobtype)
{
    return (jobtype & JobType::QM_MM) == JobType::QM_MM;
}

/**
 * @brief Returns true if the jobtype performs an MD simulation
 *
 * @return true/false
 *
 */
bool isMDJobType(JobType jobtype)
{
    return (jobtype & JobType::MD) == JobType::MD;
}

/**
 * @brief Returns true if the jobtype does is based on optimization
 *
 * @return true/false
 *
 */
bool isOptJobType(JobType jobtype)
{
    return (jobtype & JobType::OPT) == JobType::OPT;
}

/**
 * @brief Returns true if the MM simulations are activated
 *
 * @return true/false
 *
 */
bool isMMActivated(JobType jobtype)
{
    return (jobtype & JobType::MM) == JobType::MM;
}

/**
 * @brief Returns true if the QM simulations are activated
 *
 * @return true/false
 *
 */
bool isQMActivated(JobType jobtype)
{
    return (jobtype & JobType::QM) == JobType::QM;
}

/**
 * @brief Returns true if the ring polymer MD simulations are activated
 *
 * @return true/false
 *
 */
bool isRingPolymerMDActivated(JobType jobtype)
{
    return (jobtype & JobType::RING_POLYMER) == JobType::RING_POLYMER;
}
