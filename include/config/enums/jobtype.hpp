#ifndef _JOBTYPE_ENUM_HPP_
#define _JOBTYPE_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>
#include <mstd/enum/enum_string.hpp>

/**
 * @enum JobType
 *
 * @brief enum class to store the type of the job
 *
 */
enum class JobType : std::uint8_t;

#define JOB_TYPE_LIST(X)  \
    X(MM_MD)              \
    X(QM_MD)              \
    X(QMMM_MD)            \
    X(RING_POLYMER_QM_MD) \
    X(MM_OPT)             \
    X(MM_HESSIAN)         \
    X(NONE)

MSTD_ENUM(JobType, std::uint8_t, JOB_TYPE_LIST)

#undef JOB_TYPE_LIST

#endif   // _JOBTYPE_ENUM_HPP_
