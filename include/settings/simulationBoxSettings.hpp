#ifndef _SIMULATION_BOX_SETTINGS_HPP_

#define _SIMULATION_BOX_SETTINGS_HPP_

namespace settings
{
    /**
     * @class SimulationBoxSettings
     *
     * @brief static class to store settings of the simulation box
     *
     */
    class SimulationBoxSettings
    {
      private:
        static inline bool isDensitySet = false;
        static inline bool isBoxSet     = false;

      public:
        SimulationBoxSettings()  = delete;
        ~SimulationBoxSettings() = delete;

        static void setDensitySet(const bool densitySet) { isDensitySet = densitySet; }
        static void setBoxSet(const bool boxSet) { isBoxSet = boxSet; }

        [[nodiscard]] static bool getDensitySet() { return isDensitySet; }
        [[nodiscard]] static bool getBoxSet() { return isBoxSet; }
    };
}   // namespace settings

#endif   // _SIMULATION_BOX_SETTINGS_HPP_