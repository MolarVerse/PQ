#ifndef _HYBRID_ENUM_HPP_
#define _HYBRID_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>

/**
 * @enum SmoothingMethod
 *
 * @brief enum class to store the type of smoothing method
 *
 */
enum class SmoothingMethod : std::uint8_t;

#define SMOOTHING_METHOD_LIST(X) \
    X(HOTSPOT)                   \
    X(EXACT)

MSTD_ENUM(SmoothingMethod, std::uint8_t, SMOOTHING_METHOD_LIST)

#undef SMOOTHING_METHOD_LIST

/**
 * @enum QMForceDist
 *
 * @brief enum class to store the type of force distribution of the QM
 * method in hotspot smoothing
 *
 */
enum class QMForceDist : std::uint8_t;

#define QM_FORCE_DIST_LIST(X) \
    X(NONE)                   \
    X(EQUAL)                  \
    X(RANDOM)                 \
    X(DISTANCE_WEIGHTED)

MSTD_ENUM(QMForceDist, std::uint8_t, QM_FORCE_DIST_LIST)

#undef QM_FORCE_DIST_LIST

#endif   // _HYBRID_ENUM_HPP_
