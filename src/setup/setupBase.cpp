#include "setupBase.hpp"

#include "stdoutOutput.hpp"

namespace setup
{

    /**
     * @brief Writes a setup warning to the log and stdout
     *
     * @param log The log output object
     * @param message The warning message
     */
    void SetupBase::writeSetupWarning(
        out::LogOutput    &log,
        const std::string &message
    )
    {
        log.writeSetupWarning(message);
        out::StdoutOutput::writeSetupWarning(message);
    }

}   // namespace setup
