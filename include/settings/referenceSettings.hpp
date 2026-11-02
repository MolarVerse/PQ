#ifndef SETTINGS__REFERENCE_SETTINGS_HPP
#define SETTINGS__REFERENCE_SETTINGS_HPP

#include <string>
#include <unordered_set>

/**
 * @brief Class to manage reference and BibTeX file settings.
 */
class ReferenceSettings
{
   private:
    std::unordered_set<std::string> _referenceFileNames;
    std::unordered_set<std::string> _bibtexFileNames;

   public:
    void addReferenceFile(const std::string& referenceFileName);

    const std::unordered_set<std::string>& getReferenceFileNames() const;
    const std::unordered_set<std::string>& getBibtexFileNames() const;
};

struct ReferenceFiles
{
    // PQ Software
    static constexpr auto* PQ_FILE = "pq.ref";

    // Time Integrators
    static constexpr auto* VELOCITY_VERLET_FILE = "velocity_verlet.ref";

    // Thermostats and Manostats
    static constexpr auto* BERENDSEN_FILE          = "berendsen.ref";
    static constexpr auto* VELOCITY_RESCALING_FILE = "velocity_rescaling.ref";
    static constexpr auto* NOSE_HOOVER_CHAIN_FILE  = "nose_hoover_chain.ref";
    static constexpr auto* LANGEVIN_FILE           = "langevin.ref";
    static constexpr auto* STOCHASTIC_RESCALING_FILE =
        "stochastic_rescaling.ref";

    // QM Programs
    static constexpr auto* DFTBPLUS_FILE  = "dftbplus.ref";
    static constexpr auto* THREEOB_FILE   = "3ob.ref";
    static constexpr auto* MATSCI_FILE    = "matsci.ref";
    static constexpr auto* GFN1_FILE      = "gfn1.ref";
    static constexpr auto* GFN2_FILE      = "gfn2.ref";
    static constexpr auto* IPEA1_FILE     = "ipea1.ref";
    static constexpr auto* PYSCF_FILE     = "pyscf.ref";
    static constexpr auto* TURBOMOLE_FILE = "turbomole.ref";
    static constexpr auto* MACEMP_FILE    = "macemp.ref";
    static constexpr auto* MACEOFF_FILE   = "maceoff.ref";
    static constexpr auto* FENNOL_FILE    = "fennol.ref";

    // Constraint Dynamics
    static constexpr auto* RATTLE_FILE = "rattle.ref";

    // Water Models
    static constexpr auto* SPC_FILE       = "water_model_spc.ref";
    static constexpr auto* SPC_E_FILE     = "water_model_spce.ref";
    static constexpr auto* SPC_FW_FILE    = "water_model_spcfw.ref";
    static constexpr auto* QSPC_FW_FILE   = "water_model_qspcfw.ref";
    static constexpr auto* SPC_DC_FILE    = "water_model_spcdc.ref";
    static constexpr auto* H2O_DC_FILE    = "water_model_h2odc.ref";
    static constexpr auto* TIP3P_FILE     = "water_model_tip3p.ref";
    static constexpr auto* OPC3_FILE      = "water_model_opc3.ref";
    static constexpr auto* SPC_MTR_FILE   = "water_model_spcmtr.ref";
    static constexpr auto* TIP3P_MTR_FILE = "water_model_tip3pmtr.ref";
};

#endif   // SETTINGS__REFERENCE_SETTINGS_HPP
