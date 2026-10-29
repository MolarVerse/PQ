#include "manostatFactory.hpp"

#include <utility>

#include "berendsenManostat.hpp"
#include "manostat.hpp"
#include "manostatSettings.hpp"
#include "stochasticRescalingManostat.hpp"

namespace manostat
{
    /**
     * @brief Creates a Manostat instance based on the provided settings
     *
     * @param settings The settings used to configure the Manostat
     * @return std::unique_ptr<Manostat> A unique pointer to the created
     * Manostat instance
     */
    std::unique_ptr<Manostat> ManostatFactory::createManostat(
        const ManostatSettings& settings
    )
    {
        switch (settings.getManostatType())
        {
            case ManostatType::BERENDSEN:
                return _setupBerendsenManostat(settings);
                break;
            case ManostatType::STOCHASTIC_RESCALING:
                return _setupStochasticRescalingManostat(settings);
                break;
            case ManostatType::NONE:
                return std::make_unique<manostat::Manostat>(settings);
                break;
        }

        std::unreachable();
    }

    /**
     * @brief setup berendsen manostat
     *
     * @param settings The settings used to configure the Manostat
     * @return std::unique_ptr<Manostat> A unique pointer to the created
     * Manostat instance
     */
    std::unique_ptr<Manostat> ManostatFactory::_setupBerendsenManostat(
        const ManostatSettings& settings
    )
    {
        switch (settings.getIsotropy())
        {
            using enum Isotropy;

            case SEMI_ISOTROPIC_XY:
            case SEMI_ISOTROPIC_XZ:
            case SEMI_ISOTROPIC_YZ:
                return std::make_unique<
                    manostat::SemiIsotropicBerendsenManostat>(settings);
                break;

            case ANISOTROPIC:
                return std::make_unique<manostat::AnisotropicBerendsenManostat>(
                    settings
                );
                break;

            case FULL_ANISOTROPIC:
                return std::make_unique<
                    manostat::FullAnisotropicBerendsenManostat>(settings);
                break;

            case ISOTROPIC:
                return std::make_unique<manostat::BerendsenManostat>(settings);
        }

        std::unreachable();
    }

    /**
     * @brief setup stochastic rescaling manostat
     *
     * @param settings The settings used to configure the Manostat
     * @return std::unique_ptr<Manostat> A unique pointer to the created
     * Manostat instance
     */
    std::unique_ptr<Manostat> ManostatFactory::
        _setupStochasticRescalingManostat(const ManostatSettings& settings)
    {
        switch (settings.getIsotropy())
        {
            using enum Isotropy;

            case SEMI_ISOTROPIC_XY:
            case SEMI_ISOTROPIC_XZ:
            case SEMI_ISOTROPIC_YZ:
                return std::make_unique<
                    manostat::SemiIsotropicStochasticRescalingManostat>(settings
                );
                break;

            case ANISOTROPIC:
                return std::make_unique<
                    manostat::AnisotropicStochasticRescalingManostat>(settings);
                break;

            case FULL_ANISOTROPIC:
                return std::make_unique<
                    manostat::FullAnisotropicStochasticRescalingManostat>(
                    settings
                );
                break;

            case ISOTROPIC:
                return std::make_unique<manostat::StochasticRescalingManostat>(
                    settings
                );
        }

        std::unreachable();
    }

}   // namespace manostat
