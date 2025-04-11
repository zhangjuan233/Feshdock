#!/usr/bin/env python3

"""Cluster LightDock final swarm results using BSAS algorithm"""

import argparse
import os
import numpy as np
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from prody import parsePDB, confProDy, calcRMSD
from feshdock.util.logger import LoggingManager
from feshdock.constants import CLUSTER_REPRESENTATIVES_FILE
from scipy.cluster.hierarchy import linkage, fcluster

from scipy.spatial.distance import squareform

# Disable ProDy output
confProDy(verbosity="info")

log = LoggingManager.get_logger("lgd_cluster_bsas")


def parse_command_line():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog="lgd_cluster_bsas")

    parser.add_argument(
        "gso_output_file", help="LightDock output file", metavar="gso_output_file"
    )

    return parser.parse_args()


def get_backbone_atoms(ids_list, swarm_path,pdbname):
    """Get all backbone atoms (CA or P) of the PDB files specified by the ids_list.

    PDB files follow the format lightdock_ID.pdb where ID is in ids_list
    """
    ca_atoms = {}
    try:
        for struct_id in ids_list:
            pdb_file = swarm_path+ f"{pdbname}_{struct_id}.pdb"
            log.info(f"Reading CA from {pdb_file}")
            structure = parsePDB(str(pdb_file))
            selection = structure.select("name CA P")
            ca_atoms[struct_id] = selection
    except IOError as e:
        log.error(f"Error found reading a structure: {e}")
        log.error(
            "Did you generate the LightDock structures corresponding to this output file?"
        )
        raise SystemExit()
    return ca_atoms


def clusterize(sorted_ids, swarm_path,pdbname):
    """Clusters the structures identified by the IDS inside sorted_ids list"""

    clusters_found = 0
    clusters = {clusters_found: [sorted_ids[0]]}

    # Read all structures backbone atoms
    backbone_atoms = get_backbone_atoms(sorted_ids, swarm_path,pdbname)

    for j in sorted_ids[1:]:
        log.info("Glowworm %d with pdb lightdock_%d.pdb" % (j, j))
        in_cluster = False
        for cluster_id in list(clusters.keys()):
            # For each cluster representative
            representative_id = clusters[cluster_id][0]
            rmsd = calcRMSD(backbone_atoms[representative_id], backbone_atoms[j]).round(
                4
            )
            log.info("RMSD between %d and %d is %5.3f" % (representative_id, j, rmsd))
            if rmsd <= 4.0: #只要小于阈值，就加入该簇
                clusters[cluster_id].append(j)
                log.info("Glowworm %d goes into cluster %d" % (j, cluster_id))
                in_cluster = True
                break

        if not in_cluster:
            clusters_found += 1
            clusters[clusters_found] = [j]
            log.info("New cluster %d" % clusters_found)
    return clusters


def hierarchical_clusterize(sorted_ids, swarm_path, pdbname):
    """Clusters the structures identified by the IDs inside sorted_ids list"""
    # Read all structures backbone atoms
    backbone_atoms = get_backbone_atoms(sorted_ids, swarm_path, pdbname) #记录骨架原子
    # Calculate pairwise RMSD matrix  计算距离矩阵
    rmsd_matrix = np.zeros((len(sorted_ids), len(sorted_ids)))
    for i, id_i in enumerate(sorted_ids):
        for j, id_j in enumerate(sorted_ids[i:]):  # 只计算上三角矩阵
            rmsd_value = calcRMSD(backbone_atoms[id_i], backbone_atoms[id_j]).round(4)
            rmsd_matrix[i, i + j] = rmsd_value
            rmsd_matrix[i + j, i] = rmsd_value  # 复制到下三角

    # Use hierarchical clustering to compute linkage matrix
    # 转换为关联矩阵
    condensed_matrix = squareform(rmsd_matrix)
    # 进行层次聚类
    linkage_matrix = linkage(condensed_matrix, method='average')  # 合并两个簇时，平均链接会考虑到两个簇中所有样本之间的平均距离
    # ------聚类树---------
    # dendrogram(linkage_matrix, labels=sorted_ids, leaf_rotation=90, leaf_font_size=8)
    # plt.title('Hierarchical Clustering Dendrogram')
    # plt.xlabel('Structure IDs')
    # plt.ylabel('Lrmsd')
    # plt.show()

    # Assign clusters based on a threshold (adjust as needed)
    threshold = 4.0
    clusters = fcluster(linkage_matrix, t=threshold, criterion='distance')

    # Organize clustered structures into dictionary
    cluster_dict = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = [sorted_ids[i]]
        else:
            cluster_dict[cluster_id].append(sorted_ids[i])
    return cluster_dict



def write_cluster_info(clusters, gso_data, swarm_path):
    """Writes the clustering result"""
    file_name = swarm_path / CLUSTER_REPRESENTATIVES_FILE
    with open(file_name, "w") as output:
        for id_cluster, ids in clusters.items():
            output.write(
                "%d:%d:%8.5f:%d:%s\n"
                % (
                    id_cluster,
                    len(ids),
                    gso_data[ids[0]].scoring,
                    ids[0],
                    "lightdock_%d.pdb" % ids[0],
                )
            )
        log.info(f"Cluster result written to {file_name} file")

def sort_by_value_length(item):
    key, value = item
    return len(value)

def rank_cluster(pdbname,clusters):
    sorted_clusters = dict(sorted(clusters.items(), key=sort_by_value_length, reverse=True))
    # with open(f'../res/{pdbname}_hierarchy.txt', 'w') as f:
    #     i = 1
    #     for key, value in sorted_clusters.items():
    #         if i <= 100:
    #             f.write(f'line{i}   cluster{key}:  respective_id:{value[0]}  size:{len(value)}  all:{str(value)}' + '\n')
    #             i += 1
    return  sorted_clusters

if __name__ == "__main__":
    os.chdir('/home/ck/pycharm_projects/Feshdock-master2/bin')
    path='/home/ck/pycharm_projects/Feshdock-master2/data1/predict_pdbs/'
    oldpath='/home/ck/pycharm_projects/Feshdock-master2/data1/predict_pdbs'
    newpath='/home/ck/pycharm_projects/Feshdock-master2/data1/final_cluster_pdbs'
    # try:
    parser = argparse.ArgumentParser(description='cluster')
    parser.add_argument('-pdbname', required=True, help='PDB')
    parser.add_argument('-n', required=True, help='nums')

    args=parser.parse_args()
    # args.pdbname
    num=int(args.n)
    pdbname=args.pdbname
    cluster_ids=[(i+1) for i in range(num)]
    # clusters = clusterize(cluster_ids, path,pdbname) #顺序聚类
    clusters = hierarchical_clusterize(cluster_ids, path,pdbname)  #层次聚类

    # --------------根据簇大小进行排序-----------------
    clusters=rank_cluster(pdbname, clusters)


    representative_list=[]  #找代表性元素
    # for item in range(len(clusters)):
    for key,value in clusters.items():
        representative_id=value[0]
        representative_list.append(representative_id)

    final_dir='../data1/'+'final_cluster_pdbs'
    os.mkdir(final_dir)
    for i in range(len(representative_list)):
        oldpdb=f'{pdbname}_{representative_list[i]}'+'.pdb'
        newpdb=f'{pdbname}_{i+1}'+'.pdb'
        comd=f'cp {oldpath}/{oldpdb} {newpath}/{newpdb}'
        os.system(comd)

