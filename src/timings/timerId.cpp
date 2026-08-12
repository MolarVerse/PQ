#include "timerId.hpp"

/**
 * @brief convert TimerId to string
 *
 * @param id
 * @return std::string
 */
std::string toString(TimerId id)
{
    if (id == TimerId::CellList)
        return "Cell List";

    if (id == TimerId::PhysicalData)
        return "Physical Data";

    if (id == TimerId::ResetKinetics)
        return "Reset Kinetics";

    if (id == TimerId::WaterIntraPotential)
        return "Water Intra Potential";

    if (id == TimerId::WaterInterPotential)
        return "Water Inter Potential";

    if (id == TimerId::QMEngine)
        return "QM Engine";

    return TimerIdMeta::toString(id);
}
