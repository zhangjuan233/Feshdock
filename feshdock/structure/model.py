import numpy as np
from feshdock.mathutil.ellipsoid import MinimumVolumeEllipsoid
from feshdock.structure.space import SpacePoints
from feshdock.error.feshdock_errors import MinimumVolumeEllipsoidError


class DockingModel(object):
    """Represents a docking model of a protein molecule"""

    def __init__(
        self,
        objects,
        coordinates,
        restraints=None,
        membrane=None,
        reference_points=None,
        n_modes=None,
        nm_mask=None,
    ):
        self.objects = objects
        if type(coordinates) is list:
            self.coordinates = coordinates
        else:
            self.coordinates = [coordinates]
        # TODO: Calculate only one set of reference points
        if reference_points is None:
            # Reference points calculation. If single matrix error is found, calculate centroid
            try:
                ellipsoid = MinimumVolumeEllipsoid(self.coordinates[0].coordinates)#将椭球体的中心作为参考点
                self.reference_points = np.array([ellipsoid.center.copy()])
                # Suggest the GC to free some memory
            except MinimumVolumeEllipsoidError:#若出现单矩阵错误，则计算质心作为参考点。
                self.reference_points = np.array([np.mean(self.coordinates[0], axis=0)])
        else:
            self.reference_points = reference_points
        self.reference_points = SpacePoints(self.reference_points)
        self.n_modes = n_modes
        self.restraints = restraints
        self.membrane = membrane
        self.nm_mask = nm_mask

    def translate(self, vector):
        """Translates coordinates based on vector"""
        for coordinates in self.coordinates:
            coordinates.translate(vector)
        self.reference_points.translate(vector)

    def rotate(self, q):
        """Rotates coordinates using a quaternion q"""
        for coordinates in self.coordinates:
            coordinates.rotate(q)
        self.reference_points.rotate(q)

    def __len__(self):
        return len(self.coordinates)
