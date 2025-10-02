from pyscf import gto, scf, mp
import warnings

basis = '6-311++g**'

####### DO NOT CHANGE ANYTHING BELOW THIS LINE ########

warnings.filterwarnings("ignore", category=UserWarning)

mol = gto.Mole()
mol.atom = "coords.xyz"
mol.basis = basis
mol.build()

# First run the SCF calculation
uhf = scf.UHF(mol)
uhf.run()

# Then run the MP2 calculation using the converged SCF
mp2 = mp.UMP2(uhf)
mp2.run()

# Get Mulliken charges from the converged SCF (not MP2)
charges = uhf.mulliken_pop()[1]  # [1] gives the charges, [0] gives the individual orbital populations

# Now run the gradient calculation
calc = mp2.nuc_grad_method().as_scanner()
e_tot, grad = calc(mol)

# Write result to the files
with open('qm_forces', 'w') as f:
    print(e_tot, file=f)
    for i in range(len(grad)):
        print(grad[i][0], grad[i][1], grad[i][2], file=f)

with open('qm_charges', 'w') as f:
    for i, charge in enumerate(charges, 1):
        print(f"{i:4d}    {charge:13.10f}", file=f)