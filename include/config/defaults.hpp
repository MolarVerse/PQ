#ifndef _DEFAULTS_HPP_

#define _DEFAULTS_HPP_

#include <string>

/**
 * @brief namespace containing all default values
 *
 */
namespace defaults
{
    constexpr char _MOLDESCRIPTOR_FILENAME_DEFAULT_[] = "moldescriptor.dat";
    constexpr char _GUFF_FILENAME_DEFAULT_[]          = "guff.dat";

    constexpr double _COULOMB_CUT_OFF_DEFAULT_           = 12.5;
    constexpr double _WOLF_PARAMETER_DEFAULT_            = 0.25;
    constexpr char   _COULOMB_LONG_RANGE_TYPE_DEFAULT_[] = "none";

    constexpr char _NONCOULOMB_TYPE_DEFAULT_[] = "none";

    constexpr double _COMPRESSIBILITY_WATER_DEFAULT_ = 4.591e-5;
    constexpr double _TAU_MANOSTAT_DEFAULT_          = 1.0;

    constexpr size_t _SHAKE_MAX_ITER_DEFAULT_  = 20;
    constexpr size_t _RATTLE_MAX_ITER_DEFAULT_ = 20;

    constexpr double _SHAKE_TOLERANCE_DEFAULT_  = 1e-8;
    constexpr double _RATTLE_TOLERANCE_DEFAULT_ = 1e-8;

    constexpr size_t _NUMBER_OF_CELLS_DEFAULT_ = 7;
}   // namespace defaults

#endif   // _DEFAULTS_HPP_