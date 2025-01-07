import numpy as np
from portaltypes import Portal, Leaf, Plane, get_winding_plane


def parse_prt_file(text):
  lines = text.strip().split("\n")
  version = 1
  versionstr = lines.pop(0)

  if versionstr == "PRT1":
    version = 1
  elif versionstr == "PRT2":
    version = 2
  else:
    raise ValueError("Invalid PRT format")

  num_leaves = int(lines.pop(0))
  num_clusters = int(lines.pop(0)) if version == 2 else 0
  num_portals = int(lines.pop(0))
  portalleafs_real = num_leaves

  portals = []
  if num_clusters > 0:
    num_leaves = num_clusters
  leaves = [Leaf([]) for _ in range(num_leaves)]

  for i in range(num_portals):
    parts = lines.pop(0).replace("(", "").replace(")", "").split()
    num_points = int(parts.pop(0))
    leaf0 = int(parts.pop(0))
    leaf1 = int(parts.pop(0))

    winding = []
    for i in range(0, len(parts), 3):
      point = np.array([float(parts[i]), float(parts[i + 1]), float(parts[i + 2])])
      winding.append(point)

    assert len(winding) == num_points

    # Get a plane that points to leaf0, the containing leaf of the forward portal.
    plane = get_winding_plane(winding)

    # The "forward portal" leads from leaf0 to leaf1 so its plane is flipped to point to the target leaf.
    forward_portal = Portal(winding, leaf1, Plane(-plane.normal, -plane.dist))
    portals.append(forward_portal)
    leaves[leaf0].portals.append(len(portals) - 1)

    backward_portal = Portal(winding[::-1], leaf0, plane)
    portals.append(backward_portal)
    leaves[leaf1].portals.append(len(portals) - 1)

  clusters = []
  for cluster_id in range(num_clusters):
    parts = lines.pop(0).split(" ")
    assert parts[-1] == "-1"
    leaf_ids = [int(s) for s in parts[:-1]]
    clusters.append(leaf_ids)

  return leaves, portals, clusters, portalleafs_real
