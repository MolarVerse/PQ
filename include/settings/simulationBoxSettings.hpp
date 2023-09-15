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
        static inline bool _isDensitySet = false;
        static inline bool _isBoxSet     = false;

        static inline bool _initializeVelocities = false;

      public:
        SimulationBoxSettings()  = delete;
        ~SimulationBoxSettings() = delete;

        static void setDensitySet(const bool densitySet) { _isDensitySet = densitySet; }
        static void setBoxSet(const bool boxSet) { _isBoxSet = boxSet; }
        static void setInitializeVelocities(const bool initializeVelocities) { _initializeVelocities = initializeVelocities; }

        [[nodiscard]] static bool getDensitySet() { return _isDensitySet; }
        [[nodiscard]] static bool getBoxSet() { return _isBoxSet; }
        [[nodiscard]] static bool getInitializeVelocities() { return _initializeVelocities; }
    };
}   // namespace settings

#endif   // _SIMULATION_BOX_SETTINGS_HPP_