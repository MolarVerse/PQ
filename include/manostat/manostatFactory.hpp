#ifndef _MANOSTAT_FACTORY_HPP_
#define _MANOSTAT_FACTORY_HPP_

#include <memory>

class ManostatSettings;

namespace manostat
{
    class Manostat;

    /**
     * @brief Factory class for creating Manostat instances
     *
     * This class provides a static method to create Manostat objects based on
     * the provided ManostatSettings. It prevents direct instantiation.
     */
    class ManostatFactory
    {
       public:
        [[nodiscard]]
        static std::unique_ptr<Manostat> createManostat(
            const ManostatSettings& settings
        );

       private:
        ManostatFactory() = default;

        static std::unique_ptr<Manostat> _setupBerendsenManostat(
            const ManostatSettings& settings
        );
        static std::unique_ptr<Manostat> _setupStochasticRescalingManostat(
            const ManostatSettings& settings
        );
    };
}   // namespace manostat

#endif   // _MANOSTAT_FACTORY_HPP_
