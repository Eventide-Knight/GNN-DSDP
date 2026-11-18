# PyMOL script for 6jao -- run with: pymol -cq 6jao_pocket.pml
load $PDBBIND_DIR/refined-set/6jao/6jao_protein.pdb, prot
load $DSDP_HOME/test_dataset/PDBBind_time_split_dataset/6jao/6jao_ligand.pdbqt, lig
load $GNNSITE_HOME/runs/c10/6jao_gnn_out.pdbqt, pose
hide everything
show cartoon, prot
set cartoon_transparency, 0.35, prot
show sticks, lig
color green, lig
show sticks, pose
color marine, pose
select pocket, resi 14A+44A+45A+46A+70A+117A+118A+119A+171A+174A+175A+178A+179A+248A+250A+252A+284A+285A+286A+287A
show spheres, pocket
color red, pocket
set sphere_scale, 0.6, pocket
distance d_lig_pose, lig, pose
zoom pocket, 8
bg_color white
ray 900,900
png 6jao_pocket_pymol.png, dpi=150
quit
