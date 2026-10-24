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

#ifndef _JOBTYPE_ENUM_HPP_
#define _JOBTYPE_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum JobType
 *
 * @brief enum class to store the type of the job
 *
 */
enum class JobType : std::uint8_t;

#define U(X) static_cast<unsigned>(X)

#define JOB_TYPE_LIST(X)                                   \
    X(MM, 1U << 0U)                                        \
    X(QM, 1U << 1U)                                        \
    X(MD, 1U << 2U)                                        \
    X(OPT, 1U << 3U)                                       \
    X(HESSIAN, 1U << 4U)                                   \
    X(RING_POLYMER, 1U << 5U)                              \
    X(QM_MD, U(QM) | U(MD))                                \
    X(MM_MD, U(MM) | U(MD))                                \
    X(QMMM_MD, U(QM) | U(MM) | U(MD))                      \
    X(RING_POLYMER_QM_MD, U(RING_POLYMER) | U(QM) | U(MD)) \
    X(MM_OPT, U(MM) | U(OPT))                              \
    X(MM_HESSIAN, U(MM) | U(HESSIAN))

MSTD_ENUM_BITFLAG(JobType, std::uint8_t, JOB_TYPE_LIST)

#undef JOB_TYPE_LIST
#undef U

#endif   // _JOBTYPE_ENUM_HPP_
