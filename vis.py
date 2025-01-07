import time
import sys
import numpy as np
from argparse import ArgumentParser
from pathlib import Path
from typing import Union

from portaltypes import Plane, Winding, Portal
from prt_format import parse_prt_file
from clipper import clip_winding


parser = ArgumentParser()
parser.add_argument("prt_path", help="Input PRT file")
parser.add_argument("input_bsp", help="Input BSP file")
parser.add_argument("output_bsp", help="Output BSP file")
parser.add_argument("--level", type=int, default=4, help="Test accuracy level from 0 to 4")
parser.add_argument("--fast", action="store_true", help="Only compute coarse visibility")
parser.add_argument("--verbose", action="store_true", help="More logging")
parser.add_argument("--noviz", action="store_true", help="Hide visualization")
parser.add_argument("--pickle", help="Dump the visibility data path to this file")

args = parser.parse_args()
if args.input_bsp is None:
  bsp_path = Path(args.prt_path).with_suffix(".bsp")
else:
  bsp_path = args.input_bsp


verbose = args.verbose
test_level = args.level
show_viz = not args.noviz


def dprint(*args, level=0, **kwargs):
  if verbose:
    print(*args, **kwargs)


# Load the portal file

text = open(args.prt_path, "r").read()
leaves, portals, clusters, portalleafs_real = parse_prt_file(text)
print(f"{len(leaves)} leaves, {len(portals)} portals, {len(clusters)} clusters")

num_leaves = len(leaves)
num_clusters = len(clusters)
num_portals = len(portals)

if num_clusters > 0:
  assert (
    num_leaves == num_clusters
  ), "In PRT2 files, each leaf actually represents a cluster of leaves"

# Allocate an array for the coarse results and hand out rows of it to portals.
# The more accurate vis results are kept in an array allocated later.

portal_mightsee = np.zeros((num_portals, num_leaves), bool)

for pi, Pi in enumerate(portals):
  Pi.mightsee = portal_mightsee[pi, :]

# Coarse or "base" vis


def portal_might_see_other(portal: Portal, other: Portal):
  # Test 1
  # 'other' must have at least one point in front of 'portal'
  for q in other.winding:
    d = portal.plane.distance_to(q)
    if d > 0.001:
      break
  else:
    return False  # no points in front

  # Test 2
  # 'portal' must have at least one point behind 'other'
  for p in portal.winding:
    d = other.plane.distance_to(p)
    if d < -0.001:
      break
  else:
    return False  # no points in back of other

  # Test 3
  # Portals shouldn't face each other
  if portal.plane.normal.dot(other.plane.normal) < -0.99:
    return False

  return True


def base_portal_flow(pi, Pi):
  print(f"Flooding portal {pi+1}/{len(portals)}")

  mightsee = np.zeros(num_leaves, dtype=bool)

  def simple_flood(leafnum, level):
    if mightsee[leafnum]:
      return

    mightsee[leafnum] = True

    leaf = leaves[leafnum]

    for pk in leaf.portals:
      if portal_might_see_other(portals[pi], portals[pk]):
        simple_flood(portals[pk].leaf, level + 1)

  simple_flood(portals[pi].leaf, 0)
  return mightsee


for pi, Pi in enumerate(portals):
  Pi.mightsee[:] = base_portal_flow(pi, Pi)

for Pi in portals:
  Pi.num_mightsee = np.sum(Pi.mightsee)
avg_num_mightsee = np.mean([Pi.num_mightsee for Pi in portals])

print("Base vis flooding done")
print(f"Average 'mightsee' leaves visible: {avg_num_mightsee}")


def print_matrix(A, column_label="", row_label="", idx_start=0):
  print(column_label)
  for pi in range(A.shape[0]):
    row_str = "".join("x " if A[pi, li] else ". " for li in range(A.shape[1]))
    print(f"{(pi+idx_start):2d} {row_str}")
  column_numbers = "".join(f"{i:2d}" for i in range(idx_start, A.shape[1] + idx_start))
  print(f"  {column_numbers} {row_label}")


if verbose:
  print("The portal_mightsee matrix:")
  print_matrix(portal_mightsee, "portal", "leaf")
  print()

# Do recursive portal clipping

from clipper import clip_to_separators

start_time = time.time()

portal_vis = np.zeros((num_portals, num_leaves), dtype=bool)

# Assign each portal a row of the portal->leaf visibility matrix
for pi, Pi in enumerate(portals):
  Pi.vis = portal_vis[pi, :]

# We can accelerate the processing a bit if we know which portals already have
# their final visibility
portal_done = np.zeros(num_portals, dtype=bool)


def portal_flow(ps: int, Ps: Portal):
  def leaf_flow(
    leafnum: int,
    mightsee: np.ndarray,
    src_poly: Winding,
    pass_plane: Plane,
    pass_poly: Union[Winding, None],
  ):
    Ps.vis[leafnum] = True

    # Test every portal leading away from this leaf
    for pt in leaves[leafnum].portals:
      Pt = portals[pt]  # Candidate target portal

      # Can the previous portal possibly see the target leaf?
      if not mightsee[Pt.leaf]:
        continue

      # Use the final visibility array if the portal has been processed
      if portal_done[pt]:
        test = Pt.vis
      else:
        test = Pt.mightsee

      # Filter away any leaves that couldn't be seen by earlier portals
      might = np.bitwise_and(mightsee, test)

      # Skip if we could see only leaves that have already proven visible
      if not any(np.bitwise_and(might, np.bitwise_not(Ps.vis))):
        continue

      # Clip the target portal to source portal's plane
      if not (target_poly := clip_winding(Pt.winding, Ps.plane)):
        continue

      # Immediate neighbors don't need other checks
      if pass_poly is None:
        leaf_flow(Pt.leaf, might, src_poly, Pt.plane, target_poly)
        continue

      # Make sure the target and source portals are in front and behind
      # of the pass portal, respectively

      if not (target_poly := clip_winding(target_poly, pass_plane)):
        continue

      if not (src_clipped := clip_winding(src_poly, -pass_plane)):
        continue

      # Finally clip the target and source polygons with separating planes

      if test_level > 0:
        target_poly = clip_to_separators(
          src_clipped, Ps.plane, pass_poly, target_poly)
        if not target_poly:
          continue

      if test_level > 1:
        target_poly = clip_to_separators(
          pass_poly, pass_plane, src_clipped, target_poly, otherside=True
        )
        if not target_poly:
          continue

      if test_level > 2:
        src_clipped = clip_to_separators(
          target_poly, Pt.plane, pass_poly, src_clipped)
        if not src_clipped:
          continue

      if test_level > 3:
        src_clipped = clip_to_separators(
          pass_poly, pass_plane, target_poly, src_clipped, otherside=True
        )
        if not src_clipped:
          continue

      # If all the checks passed we enter the leaf behind portal 'Pt'.
      # The old target portal becomes the new pass portal. The 'might'
      # list is now filtered more. Both 'src_clipped' and 'target_poly'
      # polygons may have been clipped smaller.

      leaf_flow(Pt.leaf, might, src_clipped, Pt.plane, target_poly)

  leaf_flow(Ps.leaf, Ps.mightsee, Ps.winding, Ps.plane, None)
  portal_done[ps] = True


if args.fast:
  # In fast vis we just copy over the coarse results
  print("Doing fast vis")
  for prt in portals:
    prt.vis[:] = prt.mightsee
else:
  # In full vis the portals are sorted by complexity so that simpler ones are completed first
  print("Doing full vis")
  sorted_portals = sorted(enumerate(portals), key=lambda pair: pair[1].num_mightsee)

  count = 0
  for ps, Ps in (sorted_portals):
    print(f"[{count+1}/{len(sorted_portals)}] portal {ps} with {Ps.num_mightsee} possibly visible leaves")
    portal_flow(ps, Ps)
    count += 1

print(f"Portal flow done in {time.time() - start_time:.3f} s")

if verbose:
  dprint("The portal_vis matrix:")
  print_matrix(portal_vis, "portal", "leaf", idx_start=1)


avg_see = 0
for Pi in portals:
  avg_see += np.sum(Pi.vis)
avg_see /= len(portals)
print(f"Average leaves visible per portal: {avg_see:.1f}")


print("Filling in leaf visibilities")

final_vis = np.zeros((num_leaves, num_leaves), dtype=bool)
for li, leaf in enumerate(leaves):
  for pi in leaf.portals:
    np.bitwise_or(final_vis[li], portal_vis[pi], out=final_vis[li])
  final_vis[li, li] = True

if verbose:
  dprint("The vis matrix (one-based indices):")
  print_matrix(final_vis, "leaf", "leaf", idx_start=1)

# Save the results

if args.pickle:
  import pickle

  print(f"Saving to {args.pickle}")
  with open(args.pickle, "wb") as f:
    pickle.dump(
      {
        "vis": final_vis,
        "clusters": clusters,
        "portalleafs_real": portalleafs_real,
      },
      f,
    )

import bsp_format

bsp_format.update_bsp_leaf_visibility(
  bsp_path, args.output_bsp, final_vis, clusters, portalleafs_real
)

# 3D visualization

if not show_viz:
  print("Done")
  sys.exit(0)

import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


def plot_line(a, b, **kwargs):
  ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], **kwargs)


def plot_text(p, text, **kwargs):
  ax.text(p[0], p[1], p[2], text, **kwargs)


def plot_winding(points, **kwargs):
  if len(points) < 3:
    print("Warning: Trying to plot a degenerate portal")
    return

  x = [p[0] for p in points]
  y = [p[1] for p in points]
  z = [p[2] for p in points]

  mid = np.array([np.sum(x), np.sum(y), np.sum(z)]) / len(x)

  x.append(x[0])
  y.append(y[0])
  z.append(z[0])

  c = ["red", "pink", "olive", "green", "blue", "navy"]
  for i in range(len(x) - 1):
    if "color" in kwargs:
      ax.plot([x[i], x[i + 1]], [y[i], y[i + 1]], [z[i], z[i + 1]], **kwargs)
    else:
      ax.plot(
        [x[i], x[i + 1]],
        [y[i], y[i + 1]],
        [z[i], z[i + 1]],
        color=c[i % len(c)],
        **kwargs,
      )
  return mid


def plot_portal(portal, text=None, show_radius=False, **kwargs):
  def normalize(v):
    return v / np.linalg.norm(v)

  points = portal.winding
  normal = portal.plane.normal
  mid = plot_winding(points, **kwargs)
  midn = mid + 8.0 * normal

  if text:
    mid2 = mid + 9.0 * normal
    ax.text(mid2[0], mid2[1], mid2[2], text)

  if "color" in kwargs:
    plot_line(mid, midn, **kwargs)
  else:
    plot_line(mid, midn, color="grey", **kwargs)

  if show_radius:
    dir = normalize(points[0] - portal.sphere_origin)
    plot_line(
      portal.sphere_origin,
      portal.sphere_origin + portal.sphere_radius * dir,
      color="blue",
    )


if show_viz:
  # Draw all portals
  for idx, port in enumerate(portals):
    if idx % 2 == 0:  #  Draw only forward portals
      plot_portal(port, text=str(idx))
      pass

  for idx, leaf in enumerate(leaves):
    origin = np.array([0.0, 0.0, 0.0])
    for pi in leaf.portals:
      Pi = portals[pi]
      mid = np.mean(Pi.winding, axis=0)
      origin += mid
    origin /= len(leaf.portals)
    plot_text(origin, f"$L_{{{idx}}}$", color="darkgrey")

  ax.set_xlabel("X")
  ax.set_ylabel("Y")
  ax.set_zlabel("Z")
  ax.set_zlim(-100, 100)
  ax.set_aspect("equal")
  fig.tight_layout()
  plt.show()
