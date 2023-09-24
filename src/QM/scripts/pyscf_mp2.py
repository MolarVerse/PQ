from pyscf import gto, scf, mp
import warnings

basis = '6-311++g**'

####### DO NOT CHANGE ANYTHING BELOW THIS LINE ########

warnings.filterwarnings("ignore", category=UserWarning)

mol = gto.Mole()
mol.atom = "coords.xyz"
mol.basis = basis
mol.build()

calc = mp.UMP2(scf.UHF(mol)).nuc_grad_method().as_scanner()

e_tot, grad = calc(mol)

with open('qm_forces', 'w') as f:
    print(e_tot, file=f)
    for i in range(len(grad)):
        print(grad[i][0], grad[i][1], grad[i][2], file=f)
