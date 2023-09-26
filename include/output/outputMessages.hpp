#include <string>   // for operator==

namespace output
{
    static constexpr char _WARNING_[] = "WARNING: ";
    static constexpr char _INFO_[]    = "INFO:    ";
    static constexpr char _OUTPUT_[]  = "         ";

    std::string header();
    std::string endedNormally();

    std::string initialMomentumMessage(const double initialMomentum);

    std::string elapsedTimeMessage(const double elapsedTime);

}   // namespace output