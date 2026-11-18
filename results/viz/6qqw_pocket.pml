# PyMOL script for 6qqw -- run with: pymol -cq 6qqw_pocket.pml
load $PDBBIND_DIR/refined-set/6qqw/6qqw_protein.pdb, prot
load $DSDP_HOME/test_dataset/PDBBind_time_split_dataset/6qqw/6qqw_ligand.pdbqt, lig
load $GNNSITE_HOME/runs/6qqw_out.pdbqt, pose
hide everything
show cartoon, prot
set cartoon_transparency, 0.35, prot
show sticks, lig
color green, lig
show sticks, pose
color marine, pose
select pocket, resi 82A+83A+84A+85A+86A+87A+109A+110A+111A+112A+113A+131A+132A+133A+140A+141A+210A+112B+222B+229B
show spheres, pocket
color red, pocket
set sphere_scale, 0.6, pocket
distance d_lig_pose, lig, pose
zoom pocket, 8
bg_color white
ray 900,900
png 6qqw_pocket_pymol.png, dpi=150
quit
