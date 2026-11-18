# PyMOL script for 6kqi -- run with: pymol -cq 6kqi_pocket.pml
load $PDBBIND_DIR/v2020-other-PL/6kqi/6kqi_protein.pdb, prot
load $DSDP_HOME/test_dataset/PDBBind_time_split_dataset/6kqi/6kqi_ligand.pdbqt, lig
load $GNNSITE_HOME/runs/c10/6kqi_gnn_out.pdbqt, pose
hide everything
show cartoon, prot
set cartoon_transparency, 0.35, prot
show sticks, lig
color green, lig
show sticks, pose
color marine, pose
select pocket, resi 209A+212A+213A+214A+215A+216A+217A+224A+228A+236A+238A+293A+296A+297A+298A+299A+1120A+1121A+338A+339A
show spheres, pocket
color red, pocket
set sphere_scale, 0.6, pocket
distance d_lig_pose, lig, pose
zoom pocket, 8
bg_color white
ray 900,900
png 6kqi_pocket_pymol.png, dpi=150
quit
