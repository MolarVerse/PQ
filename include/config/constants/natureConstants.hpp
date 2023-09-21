#ifndef _NATURE_CONSTANTS_HPP_

#define _NATURE_CONSTANTS_HPP_

#include <cmath>

namespace constants
{
    /**
     * @brief avogadro number in mol⁻¹
     */
    static constexpr double _AVOGADRO_NUMBER_ = 6.022140857e23;

    /**
     * @brief bohr radius in m
     */
    static constexpr double _BOHR_RADIUS_IN_METER_ = 5.2917721067e-11;

    /**
     * @brief Planck constant in J s
     */
    static constexpr double _PLANCK_CONSTANT_         = 6.626070040e-34;
    static constexpr double _REDUCED_PLANCK_CONSTANT_ = _PLANCK_CONSTANT_ / (2.0 * M_PI);

    /**
     * @brief Boltzmann constant in J K⁻¹
     * @brief universal gas constant in J mol⁻¹ K⁻¹
     */
    static constexpr double _BOLTZMANN_CONSTANT_     = 1.38064852e-23;
    static constexpr double _UNIVERSAL_GAS_CONSTANT_ = _BOLTZMANN_CONSTANT_ * _AVOGADRO_NUMBER_;

    /**
     * @brief electron charge in C
     */
    static constexpr double _ELECTRON_CHARGE_ = 1.6021766208e-19;

    /**
     * @brief electron charge squared in C²
     */
    static constexpr double _ELECTRON_CHARGE_SQUARED_ = _ELECTRON_CHARGE_ * _ELECTRON_CHARGE_;

    /**
     * @brief electron mass in kg
     */
    static constexpr double _ELECTRON_MASS_ = 9.10938356e-31;

    /**
     * @brief permittivity of vacuum in F/m
     */
    static constexpr double _PERMITTIVITY_VACUUM_ = 8.854187817e-12;

    /**
     * @brief speed of light in m/s
     */
    static constexpr double _SPEED_OF_LIGHT_ = 299792458.0;

}   // namespace constants

#endif   // _NATURE_CONSTANTS_HPP_