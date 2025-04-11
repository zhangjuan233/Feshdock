#!/usr/bin/env python3
"""
  Feshdock is developed based on modifications to the LightDock source code.
"""
import argparse
import glob
import os
import sys
from prody import parsePDB, confProDy, calcRMSD
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feshdock.util.logger import LoggingManager
from scipy.cluster.hierarchy import linkage, fcluster
import numpy as np
from scipy.spatial.distance import squareform
from feshdock.constants import DATA_PATH,THRESHOLD,DE_OUTPUT,PREDICT_OUTPUT

# Disable ProDy output
confProDy(verbosity="info")
log = LoggingManager.get_logger("hcluster")


def parse_command_line():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog="hcluster")

    parser.add_argument(
        "de_output_file", help="feshdock output file", metavar="de_output_file"
    )

    return parser.parse_args()

def get_backbone_atoms(ids_list, swarm_path,pdbname):
    """Get all backbone atoms (CA or P) of the PDB files specified by the ids_list.
    PDB files follow the format feshdock_ID.pdb where ID is in ids_list
    """
    ca_atoms = {}
    try:
        for struct_id in ids_list:
            pdb_file = swarm_path+ f"/{pdbname}_{struct_id}.pdb"
            log.info(f"Reading CA from {pdb_file}")
            structure = parsePDB(str(pdb_file))
            selection = structure.select("name CA P")
            ca_atoms[struct_id] = selection
    except IOError as e:
        log.error(f"Error found reading a structure: {e}")
        log.error(
            "Did you generate the feshdock structures corresponding to this output file?"
        )
        raise SystemExit()
    return ca_atoms

def hierarchical_clusterize(sorted_ids, swarm_path, pdbname):
    """Clusters the structures identified by the IDs inside sorted_ids list"""
    # Read all structures backbone atoms
    backbone_atoms = get_backbone_atoms(sorted_ids, swarm_path, pdbname)
    # Calculate pairwise RMSD matrix
    rmsd_matrix = np.zeros((len(sorted_ids), len(sorted_ids)))
    for i, id_i in enumerate(sorted_ids):
        for j, id_j in enumerate(sorted_ids[i:]):
            rmsd_value = calcRMSD(backbone_atoms[id_i], backbone_atoms[id_j]).round(4)
            rmsd_matrix[i, i + j] = rmsd_value
            rmsd_matrix[i + j, i] = rmsd_value
    # Use hierarchical clustering to compute linkage matrix
    condensed_matrix = squareform(rmsd_matrix)
    linkage_matrix = linkage(condensed_matrix, method='average')

    # Assign clusters based on a threshold
    clusters = fcluster(linkage_matrix, t=THRESHOLD, criterion='distance')

    # Organize clustered structures into dictionary
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = [sorted_ids[i]]
        else:
            cluster_dict[cluster_id].append(sorted_ids[i])
    return cluster_dict

def sort_by_value_length(item):
    key, value = item
    return len(value)

def rank_cluster(pdbname,clusters):
    sorted_clusters = dict(sorted(clusters.items(), key=sort_by_value_length, reverse=True))
    return  sorted_clusters

if __name__ == "__main__":
    root=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    parser = argparse.ArgumentParser(description='cluster')
    parser.add_argument('-pdbname', required=True, help='pdbname')

    predict_path = os.path.join(root, DATA_PATH, DE_OUTPUT)
    oldpath = predict_path
    newpath = os.path.join(root, DATA_PATH, PREDICT_OUTPUT)

    args=parser.parse_args()
    num=len(glob.glob(oldpath+'/*'))
    pdbname=args.pdbname
    cluster_ids=[(i+1) for i in range(num)]
    clusters = hierarchical_clusterize(cluster_ids, predict_path, pdbname)
    clusters=rank_cluster(pdbname, clusters)

    representative_list=[]
    for key,value in clusters.items():
        representative_id=value[0]
        representative_list.append(representative_id)

    os.mkdir(newpath)
    for i in range(len(representative_list)):
        oldpdb=f'{pdbname}_{representative_list[i]}'+'.pdb'
        newpdb=f'{pdbname}_{i+1}'+'.pdb'
        comd=f'cp {oldpath}/{oldpdb} {newpath}/{newpdb}'
        os.system(comd)

