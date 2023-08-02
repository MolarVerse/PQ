#include "manostatSetup.hpp"

#include "constants.hpp"
#include "exceptions.hpp"

using namespace std;
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
 * TODO: include warnings if value set but not used
 *
 */
void ManostatSetup::setup()
{
    if (_engine.getSettings().getManostat() == "berendsen")
    {
        if (!_engine.getSettings().getPressureSet()) throw InputFileException("Pressure not set for Berendsen manostat");

        _engine.makeManostat(BerendsenManostat(_engine.getSettings().getPressure(),
                                               _engine.getSettings().getTauManostat() * constants::_PS_TO_FS_,
                                               _engine.getSettings().getCompressibility()));
    }

    _engine.getManostat().setTimestep(_engine.getTimings().getTimestep());
}