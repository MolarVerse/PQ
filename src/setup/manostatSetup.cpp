#include "manostatSetup.hpp"

#include "constants.hpp"    // for _PS_TO_FS_
#include "engine.hpp"       // for Engine
#include "exceptions.hpp"   // for InputFileException, customException
#include "manostat.hpp"     // for BerendsenManostat, Manostat, manostat
#include "settings.hpp"     // for Settings
#include "timings.hpp"      // for Timings

#include <string>        // for operator==
#include <string_view>   // for string_view

using namespace setup;
using namespace customException;
using namespace manostat;

/**
 * @brief wrapper for setupManostat
 *
 */
void setup::setupManostat(engine::Engine &engine)
{
    ManostatSetup manostatSetup(engine);
    manostatSetup.setup();
}

/**
 * @brief setup manostat
 *
 */
void ManostatSetup::setup()
{
    if (_engine.getSettings().getManostat() == "berendsen")
    {
        if (!_engine.getSettings().getPressureSet())
            throw InputFileException("Pressure not set for Berendsen manostat");

        _engine.makeManostat(BerendsenManostat(_engine.getSettings().getPressure(),
                                               _engine.getSettings().getTauManostat() * constants::_PS_TO_FS_,
                                               _engine.getSettings().getCompressibility()));
    }

    _engine.getManostat().setTimestep(_engine.getTimings().getTimestep());
}