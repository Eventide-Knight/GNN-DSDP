# PyMOL script for 6g2o -- run with: pymol -cq 6g2o_pocket.pml
load /share/home/huangyy/DSDP/gnn_site/pdbbind_v2020/refined-set/6g2o/6g2o_protein.pdb, prot
load /share/home/huangyy/DSDP/test_dataset/PDBBind_time_split_dataset/6g2o/6g2o_ligand.pdbqt, lig
load /share/home/huangyy/DSDP/gnn_site/runs/c10/6g2o_gnn_out.pdbqt, pose
hide everything
show cartoon, prot
set cartoon_transparency, 0.35, prot
show sticks, lig
color green, lig
show sticks, pose
color marine, pose
select pocket, resi 275A+277A+284A+312A+313A+314A+315A+321A+333A+336A+367A+370A+371A+372A+373A+376A+381A+392A+393A+394A
show spheres, pocket
color red, pocket
set sphere_scale, 0.6, pocket
distance d_lig_pose, lig, pose
zoom pocket, 8
bg_color white
ray 900,900
png 6g2o_pocket_pymol.png, dpi=150
quit
