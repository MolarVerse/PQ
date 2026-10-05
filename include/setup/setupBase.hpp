#ifndef _SETUP_BASE_HPP_
#define _SETUP_BASE_HPP_

#include "logOutput.hpp"

namespace setup
{
    /**
     * @brief Base class for setup-related classes
     *
     */
    class SetupBase
    {
       public:
        virtual ~SetupBase() = default;

       protected:
        static void writeSetupWarning(
            out::LogOutput    &log,
            const std::string &message
        );
    };
}   // namespace setup

#endif   // _SETUP_BASE_HPP_
