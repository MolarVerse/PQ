#ifndef _SETTINGS_HPP_

#define _SETTINGS_HPP_

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
        static inline std::string _jobtype;   // no default value

        static inline bool _isMM = false;
        static inline bool _isQM = false;

      public:
        Settings()  = default;
        ~Settings() = default;

        /***************************
         * standard getter methods *
         ***************************/

        [[nodiscard]] static std::string getJobtype() { return _jobtype; }

        [[nodiscard]] static bool getIsMM() { return _isMM; }
        [[nodiscard]] static bool getIsQM() { return _isQM; }

        /***************************
         * standard setter methods *
         ***************************/

        static void setJobtype(const std::string_view jobtype) { _jobtype = jobtype; }

        static void setIsMM(const bool isMM) { _isMM = isMM; }
        static void setIsQM(const bool isQM) { _isQM = isQM; }
    };

}   // namespace settings

#endif   // _SETTINGS_HPP_