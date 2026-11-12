#!/usr/bin/env python

'''
Classical MD active space sleection algorithm for H2O
'''

import numpy as np
import prism.tools.active_space as act_space

# Identify trajectory
xyz_files = act_space.get_geometries(
    directory="xyz_files_water_md_simul/",
    manual=True,
    min=0,
    max=11
)

# Generate guess orbitals
orbital_results = act_space.run_guess_orbitals(
    xyz_files=xyz_files,
    method="rhf",
    basis="cc-pVDZ",
)

alg_results = act_space.run_algorithm(orbital_results)

print(orbital_results)

