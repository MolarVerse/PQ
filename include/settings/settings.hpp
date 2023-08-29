#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

#include "defaults.hpp"

#include <cstddef>       // for size_t
#include <string>        // for string, allocator
#include <string_view>   // for basic_string_view, string_view

namespace settings
{
    /**
     * @class Settings
     *
     * @brief Stores the general settings of the simulation
     *
     */
    class Settings
    {
      private:
        // resetKineticsSettings for later setup

        static inline std::string _jobtype;   // no default value

      public:
        [[nodiscard]] static std::string getJobtype() { return _jobtype; }

        static void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_