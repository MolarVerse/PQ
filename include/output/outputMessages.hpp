#include <string>   // for operator==

namespace output
{
    std::string header();
    std::string endedNormally();

    std::string initialMomentumMessage(const double initialMomentum);

    std::string elapsedTimeMessage(const double elapsedTime);
}