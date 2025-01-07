import numpy as np
import enum
from dataclasses import dataclass

Point = np.ndarray  # a 3D point
Winding = list[Point]  # a convex polygon


@dataclass
class Plane:
  normal: np.ndarray
  dist: float

  def distance_to(self, p: Point):
    return self.normal.dot(p) - self.dist

  def __neg__(self):
    return Plane(-self.normal, -self.dist)


class ProcessStatus(enum.IntEnum):
  NONE = 0
  WORKING = 1
  DONE = 2


class Portal:
  winding: Winding  # a convex polygon
  leaf: int  # index of the leaf where this portal leads
  plane: Plane  # plane normal points towards the leaf
  num_mightsee = 0
  num_cansee = 0
  mightsee: np.ndarray = None  # what leaves are roughly visible
  vis: np.ndarray = None  # what leaves are visible

  def __init__(self, winding, leaf, plane):
    self.winding = winding
    self.leaf = leaf
    self.plane = plane

    points = np.array(self.winding)
    origin = points.mean(axis=0, keepdims=True)
    self.sphere_origin = origin[0]
    self.sphere_radius = np.max(np.linalg.norm(points - origin, axis=1))


@dataclass
class Leaf:
  portals: list[int]  # A list of portal indices


def get_winding_plane(winding: Winding) -> Plane:
  """
  Plane normal points inside the leaf containing the winding.
  So the "front" face is counter-clockwise bound and the normal
  points away from camera when portal is looked through.
  """
  normal = np.cross(winding[0] - winding[1], winding[2] - winding[1])
  normal /= np.linalg.norm(normal)
  return Plane(normal, winding[0].dot(normal))
