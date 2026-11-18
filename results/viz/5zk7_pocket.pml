# PyMOL script for 5zk7 -- run with: pymol -cq 5zk7_pocket.pml
load /share/home/huangyy/DSDP/gnn_site/pdbbind_v2020/v2020-other-PL/5zk7/5zk7_protein.pdb, prot
load /share/home/huangyy/DSDP/test_dataset/PDBBind_time_split_dataset/5zk7/5zk7_ligand.pdbqt, lig
load /share/home/huangyy/DSDP/gnn_site/runs/c10/5zk7_gnn_out.pdbqt, pose
hide everything
show cartoon, prot
set cartoon_transparency, 0.35, prot
show sticks, lig
color green, lig
show sticks, pose
color marine, pose
select pocket, resi 45A+46A+47A+56A+62A+63A+73A+75A+76A+79A+101A+102A+107A+110A+111A+145A+147A+148A+168A+169A
show spheres, pocket
color red, pocket
set sphere_scale, 0.6, pocket
distance d_lig_pose, lig, pose
zoom pocket, 8
bg_color white
ray 900,900
png 5zk7_pocket_pymol.png, dpi=150
quit
