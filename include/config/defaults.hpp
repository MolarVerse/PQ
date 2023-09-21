#ifndef _DEFAULTS_HPP_

#define _DEFAULTS_HPP_

#include <string>

/**
 * @brief namespace containing all default values
 *
 */
namespace defaults
{
    static constexpr char   _MOLDESCRIPTOR_FILENAME_DEFAULT_[] = "moldescriptor.dat";
    static constexpr char   _GUFF_FILENAME_DEFAULT_[]          = "guff.dat";
    static constexpr size_t _NUMBER_OF_GUFF_ENTRIES_           = 28;

    static constexpr char _RESTART_FILENAME_DEFAULT_[]                 = "default.rst";
    static constexpr char _ENERGY_FILENAME_DEFAULT_[]                  = "default.en";
    static constexpr char _MOMENTUM_FILENAME_DEFAULT_[]                = "default.mom";
    static constexpr char _TRAJECTORY_FILENAME_DEFAULT_[]              = "default.xyz";
    static constexpr char _VELOCITY_FILENAME_DEFAULT_[]                = "default.vel";
    static constexpr char _FORCE_FILENAME_DEFAULT_[]                   = "default.force";
    static constexpr char _CHARGE_FILENAME_DEFAULT_[]                  = "default.charge";
    static constexpr char _LOG_FILENAME_DEFAULT_[]                     = "default.out";
    static constexpr char _INFO_FILENAME_DEFAULT_[]                    = "default.info";
    static constexpr char _RING_POLYMER_RESTART_FILENAME_DEFAULT_[]    = "default.rpmd.rst";
    static constexpr char _RING_POLYMER_TRAJECTORY_FILENAME_DEFAULT_[] = "default.rpmd.xyz";
    static constexpr char _RING_POLYMER_VELOCITY_FILENAME_DEFAULT_[]   = "default.rpmd.vel";
    static constexpr char _RING_POLYMER_FORCE_FILENAME_DEFAULT_[]      = "default.rpmd.force";
    static constexpr char _RING_POLYMER_CHARGE_FILENAME_DEFAULT_[]     = "default.rpmd.charge";

    static constexpr double _COULOMB_CUT_OFF_DEFAULT_           = 12.5;   // in Angstrom
    static constexpr double _SCALE_14_COULOMB_DEFAULT_          = 1.0;
    static constexpr double _SCALE_14_VAN_DER_WAALS_DEFAULT_    = 1.0;
    static constexpr double _WOLF_PARAMETER_DEFAULT_            = 0.25;     // TODO: add unit
    static constexpr char   _COULOMB_LONG_RANGE_TYPE_DEFAULT_[] = "none";   // default no coulomb long range correction
    static constexpr char   _NON_COULOMB_TYPE_DEFAULT_[]        = "guff";   // default is guff

    static constexpr bool   _CONSTRAINTS_ARE_ACTIVE_DEFAULT_ = false;
    static constexpr size_t _SHAKE_MAX_ITER_DEFAULT_         = 20;
    static constexpr size_t _RATTLE_MAX_ITER_DEFAULT_        = 20;
    static constexpr double _SHAKE_TOLERANCE_DEFAULT_        = 1e-8;
    static constexpr double _RATTLE_TOLERANCE_DEFAULT_       = 1e-8 * 1e12;

    static constexpr bool   _CELL_LIST_IS_ACTIVE_DEFAULT_ = false;   // default is brute force routine
    static constexpr size_t _NUMBER_OF_CELLS_DEFAULT_     = 7;       // for each dimension

    static constexpr size_t _NOSE_HOOVER_CHAIN_LENGTH_DEFAULT_     = 3;        // default value for nose hoover chain length
    static constexpr double _BERENDSEN_THERMOSTAT_RELAXATION_TIME_ = 0.1;      // in ps
    static constexpr double _LANGEVIN_THERMOSTAT_FRICTION_         = 1.0e11;   // in s^-1
    static constexpr double _NOSE_HOOVER_COUPLING_FREQUENCY_       = 1.0e3;    // in cm^-1

    static constexpr double _BERENDSEN_MANOSTAT_RELAXATION_TIME_ = 1.0;        // in ps
    static constexpr double _COMPRESSIBILITY_WATER_DEFAULT_      = 4.591e-5;   // default value for berendsen manostat

}   // namespace defaults

#endif   // _DEFAULTS_HPP_