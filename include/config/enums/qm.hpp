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

#ifndef _QM_ENUM_HPP_
#define _QM_ENUM_HPP_

#include <cstdint>
#include <mstd/enum.hpp>
#include <mstd/enum/enum_string.hpp>

/**
 * @brief enum QMMethod
 *
 */
enum class QMMethod : std::uint8_t;

#define QM_METHOD_LIST(X) \
    X(NONE)               \
    X(DFTBPLUS)           \
    X(ASE_DFTBPLUS)       \
    X(ASE_XTB)            \
    X(PYSCF)              \
    X(TURBOMOLE)          \
    X(MACE)               \
    X(FENNOL)

MSTD_ENUM(QMMethod, std::uint8_t, QM_METHOD_LIST)

/**
 * @brief Input aliases for QMMethod enum
 *
 * @tparam QMMethod The enum type for which input aliases are defined.
 */
namespace mstd
{
    template <>
    struct EnumAliases<QMMethod>
    {
        static constexpr auto value = mstd::makeAliases<QMMethod>(
            {{"mace_mp", QMMethod::MACE},
             {"mace_off", QMMethod::MACE},
             {"mace_anicc", QMMethod::MACE},
             {"mace_ani", QMMethod::MACE}}
        );
    };
}   // namespace mstd

#undef QM_METHOD_LIST

/**
 * @brief enum MaceModel
 *
 */
enum class MaceModel : std::uint8_t;

#define MACE_MODEL_LIST(X) \
    X(SMALL)               \
    X(MEDIUM)              \
    X(LARGE)               \
    X(SMALL_0B)            \
    X(MEDIUM_0B)           \
    X(SMALL_0B2)           \
    X(MEDIUM_0B2)          \
    X(LARGE_0B2)           \
    X(MEDIUM_0B3)          \
    X(MEDIUM_MPA_0)        \
    X(MEDIUM_OMAT_0)       \
    X(CUSTOM)

MSTD_ENUM(MaceModel, std::uint8_t, MACE_MODEL_LIST)

#undef MACE_MODEL_LIST

/**
 * @brief enum MaceModelType
 */
enum class MaceModelType : std::uint8_t;

#define MACE_MODEL_TYPE_LIST(X) \
    X(MACE_MP)                  \
    X(MACE_OFF)                 \
    X(MACE_ANICC)

MSTD_ENUM(MaceModelType, std::uint8_t, MACE_MODEL_TYPE_LIST)

namespace mstd
{
    /**
     * @brief Input aliases for MaceModelType enum
     *
     * @tparam MaceModelType The enum type for which input aliases are defined.
     */
    template <>
    struct EnumAliases<MaceModelType>
    {
        static constexpr auto value = makeAliases<MaceModelType>(
            {{"mace", MaceModelType::MACE_MP},
             {"mace_ani", MaceModelType::MACE_ANICC}}
        );
    };
}   // namespace mstd

#undef MACE_MODEL_TYPE_LIST

/**
 * @brief enum MaceMode
 *
 * @details enum class for the MACE evaluation mode / kernel backend
 */
enum class MaceMode : std::uint8_t;

#define MACE_MODE_LIST(X) \
    X(ACCURATE)           \
    X(FAST)

MSTD_ENUM(MaceMode, std::uint8_t, MACE_MODE_LIST)

#undef MACE_MODE_LIST

/**
 * @brief enum XtbMethod
 */
enum class XtbMethod : std::uint8_t;

#define XTB_METHOD_LIST(X) \
    X(GFN1)                \
    X(GFN2)                \
    X(IPEA1)

MSTD_ENUM(XtbMethod, std::uint8_t, XTB_METHOD_LIST)

namespace mstd
{
    /**
     * @brief Input aliases for XtbMethod enum
     *
     * @details We definitely need these here for the external connections to
     * recognize the xTB methods correctly.
     *
     * @tparam XtbMethod The enum type for which input aliases are defined.
     */
    template <>
    struct EnumNames<XtbMethod>
    {
        static constexpr auto value = makeNames<XtbMethod>(
            {{XtbMethod::GFN1, "GFN1-xTB"},
             {XtbMethod::GFN2, "GFN2-xTB"},
             {XtbMethod::IPEA1, "IPEA1-xTB"}}
        );
    };

    /**
     * @brief Input aliases for XtbMethod enum
     *
     * @tparam XtbMethod The enum type for which input aliases are defined.
     */
    template <>
    struct EnumAliases<XtbMethod>
    {
        static constexpr auto value = makeAliases<XtbMethod>(
            {{"gfn1_xTB", XtbMethod::GFN1},
             {"gfn2_xTB", XtbMethod::GFN2},
             {"ipea1_xTB", XtbMethod::IPEA1}}
        );
    };
}   // namespace mstd

#undef XTB_METHOD_LIST

/**
 * @brief enum SlakosType
 */
enum class SlakosType : std::uint8_t;

#define SLAKOS_TYPE_LIST(X) \
    X(NONE)                 \
    X(THREEOB)              \
    X(MATSCI)               \
    X(CUSTOM)

MSTD_ENUM(SlakosType, std::uint8_t, SLAKOS_TYPE_LIST)

namespace mstd
{
    template <>
    struct EnumNames<SlakosType>
    {
        static constexpr auto value = makeNames<SlakosType>(
            {{SlakosType::NONE, "none"},
             {SlakosType::THREEOB, "3ob"},
             {SlakosType::MATSCI, "matsci"},
             {SlakosType::CUSTOM, "custom"}}
        );
    };
}   // namespace mstd

#undef SLAKOS_TYPE_LIST

/**
 * @brief enum QMCharges
 */
enum class QMCharges : std::uint8_t;

#define QM_CHARGES_LIST(X) \
    X(QM)                  \
    X(MM)

MSTD_ENUM(QMCharges, std::uint8_t, QM_CHARGES_LIST)

#undef QM_CHARGES_LIST

#endif   // _QM_ENUM_HPP_
