from pyscf import gto, scf
import warnings

basis = 'sto-3g'

####### DO NOT CHANGE ANYTHING BELOW THIS LINE ########

warnings.filterwarnings("ignore", category=UserWarning)

mol = gto.Mole()
mol.atom = "coords.xyz"
mol.basis = basis
mol.build()

# First run the SCF calculation
uhf = scf.UHF(mol)
uhf.run()

# Get Mulliken charges from the converged SCF
charges = uhf.mulliken_pop()[1]  # [1] gives the charges, [0] gives the individual orbital populations

# Now run the gradient calculation
calc = uhf.nuc_grad_method().as_scanner()
e_tot, grad = calc(mol)

# Write result to the files
with open('qm_forces', 'w') as f:
    print(e_tot, file=f)
    for i in range(len(grad)):
        print(grad[i][0], grad[i][1], grad[i][2], file=f)

with open('qm_charges', 'w') as f:
    for i, charge in enumerate(charges, 1):
        print(f"{i:4d}    {charge:13.10f}", file=f)

