from enum import IntEnum
import numpy as np

from portaltypes import Plane, Point, Winding

ON_EPSILON = 0.1

class Side(IntEnum):
  BACK = 0
  FRONT = 1
  PLANE = 2
  BOTH = 3


# This is a transliteration of the vis.c ClipWinding function
def clip_winding(points: list[Point], plane: Plane):
  out = []
  sides = []
  dists = []
  num_back = num_front = 0

  for p in points:
    d = plane.distance_to(p)
    if d >= -ON_EPSILON and d <= ON_EPSILON:
      sides.append(Side.PLANE)
    if d < -ON_EPSILON:
      sides.append(Side.BACK)
      num_back += 1
    if d > ON_EPSILON:
      sides.append(Side.FRONT)
      num_front += 1
    dists.append(d)

  sides.append(sides[0])
  dists.append(dists[0])

  if num_front == 0:
    return None
  if num_back == 0:
    return points

  for i, p1 in enumerate(points):
    if sides[i] == Side.PLANE:
      out.append(p1)
      continue

    if sides[i] == Side.FRONT:
      out.append(p1)

    if sides[i + 1] == Side.PLANE or sides[i + 1] == sides[i]:
      continue

    p2 = points[(i + 1) % len(points)]

    dot = dists[i] / (dists[i] - dists[i + 1])
    mid = p1 + dot * (p2 - p1)

    # Avoid rounding error for axis-aligned planes
    for j in range(3):
      if plane.normal[j] == 1:
        mid[j] = plane.dist
      elif plane.normal[j] == -1:
        mid[j] = -plane.dist

    out.append(mid)

  if len(out) < 3:
    return None

  return out


def test_if_points_in_front(points: list[Point], plane: Plane) -> bool:
  num_front = 0

  for k, pk in enumerate(points):
    d = plane.distance_to(pk)
    if d >= -ON_EPSILON and d <= ON_EPSILON:
      continue
    if d < -ON_EPSILON:
      return False
    if d > -ON_EPSILON:
      num_front += 1
  
  if num_front == 0:
    return False  # All points were on plane

  return True


def clip_to_separators(
  first_poly: Winding,
  first_plane: Plane,
  second_poly: Winding,
  clipped_poly: Winding,
  otherside=False,
  out_planes=None,
) -> Winding:
  """
  Written to match the approach used in the original ClipToSeparators function.
  See https://github.com/fabiensanglard/Quake--QBSP-and-VIS/blob/e686204812f6464864e2959f9f57c1278409b70b/vis/flow.c#L40
  and https://github.com/ericwa/ericw-tools/blob/a6c7a18cb85cef64948d46780d4fc1bb3d1f575b/vis/flow.cc#L9
  for reference.
  """
  clipped = clipped_poly

  for i in range(len(first_poly)):
    j = (i + 1) % len(first_poly)
    A = first_poly[i]
    B = first_poly[j]
    AB = B - A

    # Try different points C on the second portal
    for k, C in enumerate(second_poly):
      # Test on which side of the first portal point C is
      d = first_plane.distance_to(C)
      if d < -ON_EPSILON:
        # A separating plane must have the second portal on
        # its front side by definition. Here C is behind the
        # first portal, so this will not be the case after
        #   normal = cross(AB, AC)
        # below and we'll have to flip the plane later.
        flip_towards_first = True
      elif d > ON_EPSILON:
        flip_towards_first = False
      else:
        continue  # Point C is on the first polygon's plane

      AC = C - A
      normal = np.cross(AB, AC)
      mag = np.linalg.norm(normal)

      if mag < ON_EPSILON:
        continue  # Portals might share vertices so there's no plane
      normal /= mag

      plane = Plane(normal, normal.dot(C))

      if flip_towards_first:
        plane = -plane

      # Check if the plane is actually a separator
      if not test_if_points_in_front(second_poly, plane):
        continue

      # The 'otherside' flag is set if source and pass portals are swapped.
      # In that case, second_poly == source_poly, so the plane normal
      # points to the source and not the pass portal!
      # We'll flip the plane so that correct side gets clipped below.
      if otherside:
        plane = -plane

      if out_planes is not None:
        out_planes.append((C, plane)) #  Only for debugging

      clipped = clip_winding(clipped, plane)

      if not clipped:
        return None

  return clipped
