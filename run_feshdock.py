import os,re
import time



if __name__ == '__main__':
    root=os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    native_path=f'{root}/data/'

    step=100
    n_swarms=10

    pdbname='1BVN'
    rec_pdb = pdbname + '_r_u.pdb'
    lig_pdb = pdbname + '_l_u.pdb'
    complex_pdb = pdbname + '_segid.pdb'

    t0=time.time()

    comd0 = f"python {root}/bin/gen_init_pop.py  {native_path}/{rec_pdb} {native_path}/{lig_pdb} {n_swarms}  {native_path}{pdbname}.out"  
    os.system(comd0)

    comd1 = f'python {root}/bin/setup.py  {native_path}/{rec_pdb}  {native_path}/{lig_pdb} -s {n_swarms}  --noh --now --noxt --anm'  #
    os.system(comd1)

    comd2 = f'python  {root}/bin/optimize.py  {native_path}/setup.json  -c {n_swarms} {step}  -name {pdbname}'
    os.system(comd2)


    comd3 = f'python  {root}/bin/sort.py  -step {step} -nswarms {n_swarms}'
    os.system(comd3)

    comd4 = f'python  {root}/bin/generate_conformations.py  {native_path}/{rec_pdb}  {native_path}/{lig_pdb} {native_path}/scoring_sorted.out'
    os.system(comd4)


    # The final predicted conformations are stored in the final_models directory.
    com6 = f'python  {root}/bin/hcluster.py -pdbname {pdbname} '
    os.system(com6)

    t1=time.time()
    t=round((t1-t0),1)
    print(f" {pdbname} took {t} seconds.")











