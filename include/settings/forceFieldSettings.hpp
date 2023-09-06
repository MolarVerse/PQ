#ifndef _FORCE_FIELD_SETTINGS_HPP_

#define _FORCE_FIELD_SETTINGS_HPP_

namespace settings
{
    /**
     * @class ForceFieldSettings
     *
     * @brief static class to store settings of the force field
     *
     */
    class ForceFieldSettings
    {
      private:
        static inline bool _active = false;

      public:
        /********************
         * standard getters *
         ********************/

        [[nodiscard]] static bool isActive() { return _active; }

        /********************
         * standard setters *
         ********************/

        static void activate() { _active = true; }
        static void deactivate() { _active = false; }
    };

}   // namespace settings

#endif   // _FORCE_FIELD_SETTINGS_HPP_