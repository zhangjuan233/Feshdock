"""The set of swarms of glowworm agents used in the algorithm"""
import copy
import os.path
from operator import attrgetter
from feshdock.de.glowworm import Glowworm

import numpy as np
import math

from feshdock.prep.poses import normalize_vector
from feshdock.mathutil.cython.quaternion import Quaternion

class Swarm(object):
    def __init__(self, landscape_positions, parameters):
        positions_per_glowworm = [[] for _ in range(len(landscape_positions[0]))]
        for function in landscape_positions:
            for glowworm_id, position in enumerate(function):
                positions_per_glowworm[glowworm_id].append(position)
        self.glowworms = [
            Glowworm(positions, parameters) for positions in positions_per_glowworm
        ]
        self.docking = (
            landscape_positions[0][0].__class__.__name__ != "LandscapePosition"
        )
        self.mutatepopulation=[copy.deepcopy(self.glowworms[i]) for i in range(len(self.glowworms))]

    def minimize_best(self):
        """Minimizes the glowworm with better energy using a local non-gradient minimization method"""
        best_glowworm = max(self.glowworms, key=attrgetter("scoring"))
        best_glowworm.minimize()

    def get_size(self):
        """Gets the population size of this swarm of glowworms"""
        return len(self.glowworms)

    def save(self, step, destination_path, file_name=""):
        """Saves actual population status to a file"""
        if file_name:
            dest_file_name = "%s/%s" % (destination_path, file_name)
        else:
            # dest_file_name = "%s/gso_%d.out" % (destination_path, step)
            dest_file_name = "%s/de_%d.out" % (destination_path, step)

        dest_file = open(dest_file_name, "w")
        dest_file.write(str(self))
        dest_file.close()

    def __repr__(self):
        """String representation of the population"""
        if self.docking:
            representation = "#Coordinates  RecID  LigID  Luciferin  Neighbor's number  Vision Range  Scoring\n"
        else:
            representation = (
                "#Coordinates  Luciferin  Neighbor's number  Vision Range  Scoring\n"
            )
        for glowworm in self.glowworms:
            representation += str(glowworm) + "\n"
        return representation

    def by_scoring(self,Glowworm):
        return Glowworm.scoring

    def my_sort(self):
        self.glowworms.sort(key=self.by_scoring,reverse=True)


    def generate_index(self, cur_index, p_size_cur):
        while True:
            index1 = np.random.randint(0, p_size_cur)
            if index1 != cur_index:
                break
        while True:
            index2 = np.random.randint(0, p_size_cur)
            if index2 != cur_index and index2 != index1:
                break
        while True:
            index3 = np.random.randint(0, p_size_cur)
            if index3 != cur_index and index3 != index1 and index3 != index2:
                break
        return index1,index2,index3

    def boundary_contral(self, index):
        #  (0.2Pi）
        while self.mutatepopulation[index].landscape_positions[0].angles[0] < 0:
            self.mutatepopulation[index].landscape_positions[0].angles[0] += 2 * np.pi
        while self.mutatepopulation[index].landscape_positions[0].angles[0] > 2 * np.pi:
            self.mutatepopulation[index].landscape_positions[0].angles[0] -= 2 * np.pi

        while self.mutatepopulation[index].landscape_positions[0].angles[1] > 2 * np.pi:
            self.mutatepopulation[index].landscape_positions[0].angles[1] -= 2 * np.pi

        if np.pi < self.mutatepopulation[index].landscape_positions[0].angles[1] < 2 * np.pi:
            self.mutatepopulation[index].landscape_positions[0].angles[1] = 2 * np.pi - self.mutatepopulation[index].landscape_positions[0].angles[1]
        if -2 * np.pi < self.mutatepopulation[index].landscape_positions[0].angles[1] < -1 * np.pi:
            self.mutatepopulation[index].landscape_positions[0].angles[1] = 2 * np.pi + self.mutatepopulation[index].landscape_positions[0].angles[1]
        if -1 * np.pi < self.mutatepopulation[index].landscape_positions[0].angles[1] < 0:
            self.mutatepopulation[index].landscape_positions[0].angles[1] = -self.mutatepopulation[index].landscape_positions[0].angles[1]

        #  (0,2pi)
        while self.mutatepopulation[index].landscape_positions[0].angles[2] < 0:
            self.mutatepopulation[index].landscape_positions[0].angles[2] += 2 * np.pi
        while self.mutatepopulation[index].landscape_positions[0].angles[2] > 2 * np.pi:
            self.mutatepopulation[index].landscape_positions[0].angles[2] -= 2 * np.pi
        assert 0 < self.mutatepopulation[index].landscape_positions[0].angles[0] < 2 * np.pi
        assert 0 < self.mutatepopulation[index].landscape_positions[0].angles[1] < np.pi
        assert 0 < self.mutatepopulation[index].landscape_positions[0].angles[2] < 2 * np.pi

    def angles_to_quaternion(self, index):
        q_tmp = np.zeros(4)
        q_unit = Quaternion(1, 0, 0, 0)
        if math.fabs(self.mutatepopulation[index].landscape_positions[0].angles[0] - 0.0) < 1e-6:
            self.mutatepopulation[index].landscape_positions[0].rotation = q_unit
        else:
            q_tmp[0] = self.mutatepopulation[index].landscape_positions[0].angles[0] / 2  # w
            q_tmp[1] = math.sin(self.mutatepopulation[index].landscape_positions[0].angles[1]) * math.cos(
                self.mutatepopulation[index].landscape_positions[0].angles[2])
            q_tmp[2] = math.sin(self.mutatepopulation[index].landscape_positions[0].angles[1]) * math.sin(
                self.mutatepopulation[index].landscape_positions[0].angles[2])
            q_tmp[3] = math.cos(self.mutatepopulation[index].landscape_positions[0].angles[1])
            q_update = Quaternion(math.cos(q_tmp[0]), q_tmp[1] * math.sin(q_tmp[0]), q_tmp[2] * math.sin(q_tmp[0]),q_tmp[3] * math.sin(q_tmp[0]))
            q_update.norm()  #
            self.mutatepopulation[index].landscape_positions[0].rotation= q_update

    def cross_mutate(self,step,F,CR):
        p_size_cur = len(self.glowworms)
        for i in range(p_size_cur):
            index1, index2, index3 = self.generate_index(i, p_size_cur)
            j_rand = np.random.randint(0, 26)
            for j in range(26):
                cr_rand = np.random.uniform(0, 1)
                if cr_rand < CR or j == j_rand:
                        if j < 3:
                            self.mutatepopulation[i].landscape_positions[0].translation[j] = \
                                self.glowworms[index1].landscape_positions[0].translation[j] + F * \
                                (self.glowworms[index2].landscape_positions[0].translation[j] -
                                 self.glowworms[index3].landscape_positions[0].translation[j])
                        elif 3 <= j < 6:
                            self.mutatepopulation[i].landscape_positions[0].angles[j - 3] = \
                                self.glowworms[index1].landscape_positions[0].angles[j - 3] + F * \
                                (self.glowworms[index2].landscape_positions[0].angles[j - 3] -
                                 self.glowworms[index3].landscape_positions[0].angles[j - 3])
                        elif 6<= j <16:
                            self.mutatepopulation[i].landscape_positions[0].rec_extent[j - 6] = \
                                    self.glowworms[index1].landscape_positions[0].rec_extent[j - 6] +\
                                    F *(self.glowworms[index2].landscape_positions[0].rec_extent[j - 6]-
                                        self.glowworms[index3].landscape_positions[0].rec_extent[j - 6])
                        else:
                            self.mutatepopulation[i].landscape_positions[0].lig_extent[j - 16] = \
                                self.glowworms[index1].landscape_positions[0].lig_extent[j - 16] + \
                                F * (self.glowworms[index2].landscape_positions[0].lig_extent[j - 16] -
                                     self.glowworms[index3].landscape_positions[0].lig_extent[j - 16])

                else:
                    if j < 3:
                        self.mutatepopulation[i].landscape_positions[0].translation[j] = self.glowworms[i].landscape_positions[0].translation[j]
                    elif 3 <= j < 6:
                        self.mutatepopulation[i].landscape_positions[0].angles[j - 3] = self.glowworms[i].landscape_positions[0].angles[j - 3]
                    elif 6 <= j < 16:
                        self.mutatepopulation[i].landscape_positions[0].rec_extent[j - 6] = self.glowworms[i].landscape_positions[0].rec_extent[j - 6]
                    else:
                        self.mutatepopulation[i].landscape_positions[0].lig_extent[j - 16] = self.glowworms[i].landscape_positions[0].lig_extent[j - 16]

            self.boundary_contral(i)
            self.angles_to_quaternion(i)
            self.mutatepopulation[i].compute_scoring()


    def selet(self, step,cluster_id):
        p_size_cur = len(self.glowworms)
        for i in range(p_size_cur):
            if self.mutatepopulation[i].scoring <= self.glowworms[i].scoring:
                continue
            else:
                self.glowworms[i] = copy.deepcopy(self.mutatepopulation[i])
        self.my_sort()







