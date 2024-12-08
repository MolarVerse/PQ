#ifndef __TRICLINIC_BOX_INL__
#define __TRICLINIC_BOX_INL__

#include "triclinicBox.hpp"

#ifdef __PQ_DEBUG__
#include "debug.hpp"
#endif

namespace simulationBox
{

    /**
     * @brief image triclinic
     *
     * @param boxDimensions
     * @param x
     * @param y
     * @param z
     * @param tx
     * @param ty
     * @param tz
     */
    static inline void imageTriclinic(
        const Real* const boxParams,
        Real&             x,
        Real&             y,
        Real&             z,
        Real&             tx,
        Real&             ty,
        Real&             tz
    )
    {
#ifdef __PQ_DEBUG__
        if (config::Debug::useDebug(config::DebugLevel::BOX_DEBUG))
        {
        }
#endif
        const auto unitBoxX =
            ::round(boxParams[9] * x + boxParams[10] * y + boxParams[11] * z);
        const auto unitBoxY =
            ::round(boxParams[12] * x + boxParams[13] * y + boxParams[14] * z);
        const auto unitBoxZ =
            ::round(boxParams[15] * x + boxParams[16] * y + boxParams[17] * z);

        tx = boxParams[0] * unitBoxX + boxParams[1] * unitBoxY +
             boxParams[2] * unitBoxZ;
        ty = boxParams[3] * unitBoxX + boxParams[4] * unitBoxY +
             boxParams[5] * unitBoxZ;
        tz = boxParams[6] * unitBoxX + boxParams[7] * unitBoxY +
             boxParams[8] * unitBoxZ;

        x += tx;
        y += ty;
        z += tz;
    }

}   // namespace simulationBox

#endif   // __TRICLINIC_BOX_INL__