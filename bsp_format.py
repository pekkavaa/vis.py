import struct
import logging
import enum
import numpy as np

logger = logging.getLogger(__name__)


class Q1Lump(enum.Enum):
  ENTITIES = 0
  PLANES = 1
  TEXTURES = 2
  VERTEXES = 3
  VISIBILITY = 4
  NODES = 5
  TEXINFO = 6
  FACES = 7
  LIGHTING = 8
  CLIPNODES = 9
  LEAFS = 10
  MARKSURFACES = 11
  EDGES = 12
  SURFEDGES = 13
  MODELS = 14


BSP_LUMPS = 15


# Some Quake BSP utilities adapted from Matthew Earl's pyquake
# https://github.com/matthewearl/pyquake/blob/2f1b0350dfab24b6557a37579cf15c78745e30e1/pyquake/bsp.py


class BspVersion(enum.IntEnum):
  BSP = 29
  _2PSB = struct.unpack("<L", b"2PSB")[0]
  BSP2 = struct.unpack("<L", b"BSP2")[0]

  @property
  def uses_longs(self):
    return self != BspVersion.BSP

  @property
  def leaf_fmt(self):
    if self == BspVersion.BSP:
      return "<llhhhhhhHHBBBB"
    elif self == BspVersion._2PSB:
      return "<llhhhhhhLLBBBB"
    elif self == BspVersion.BSP2:
      return "<llffffffLLBBBB"
    raise ValueError()


def compress_visbits(in_arr):
  dst = bytearray()
  i = 0
  while i < len(in_arr):
    val = in_arr[i]
    dst.append(val)
    if val != 0:
      i += 1
      continue

    rep = 1
    for j in range(i + 1, len(in_arr)):
      repval = in_arr[j]
      if repval != val or rep == 255:
        break
      rep += 1
    dst.append(rep)
    i += rep

  return dst


def load_q1_bsp(file_path: str, lumps_to_load: list[Q1Lump]):
  with open(file_path, "rb") as f:
    version = BspVersion(struct.unpack("<I", f.read(4))[0])

    lumps_info = []
    for i in range(BSP_LUMPS):
      lump_offset, lump_size = struct.unpack("<ii", f.read(8))
      lumps_info.append((Q1Lump(i), lump_offset, lump_size))

    lumps_data = {}

    for lump_id in lumps_to_load:
      _, ofs, size = lumps_info[lump_id.value]
      f.seek(ofs)
      lumps_data[lump_id] = f.read(size)

    return version, lumps_info, lumps_data


def read_leaves(leafdata, version):
  leaf_size = struct.calcsize(version.leaf_fmt)
  leaves = [
    struct.unpack(version.leaf_fmt, leafdata[i : (i + leaf_size)])
    for i in range(0, len(leafdata), leaf_size)
  ]
  return leaves


def write_leaves(leaves, version):
  written = bytearray()
  for leaf in leaves:
    written.extend(struct.pack(version.leaf_fmt, *leaf))
  return written


def cluster_to_bits(mask, num_leaves, clusters):
  cluster_inds = np.nonzero(mask)[0]
  data = np.zeros(num_leaves, dtype=bool)
  for cl in cluster_inds:
    for leaf_idx in clusters[cl]:
      data[leaf_idx] = True
  return bytearray(np.packbits(data, bitorder="little"))


def update_bsp_leaf_visibility(
  old_path: str,
  new_path: str,
  vis: np.ndarray,
  clusters: list[list[int]],
  portalleaves_real: int,
):
  num_clusters = len(clusters)
  version, info, lumps_data = load_q1_bsp(old_path, [Q1Lump.LEAFS])

  leaves = read_leaves(lumps_data[Q1Lump.LEAFS], version)
  num_leaves = len(leaves)
  logger.debug(f"{num_leaves=}")

  if num_clusters > 0:
    assert vis.shape == (num_clusters, num_clusters)

  vis_lump = bytearray()

  patched_leaves = []
  leaf_vis_offsets = {}

  if num_clusters > 0:
    for cluster_idx, cluster in enumerate(clusters):
      vis_offset = len(vis_lump)
      vis_bits = cluster_to_bits(vis[cluster_idx], portalleaves_real, clusters)
      vis_bits_compressed = compress_visbits(vis_bits)
      for vis_leaf_idx in cluster:
        leaf_vis_offsets[vis_leaf_idx + 1] = vis_offset

      vis_lump.extend(vis_bits_compressed)

  else:
    for leaf_idx, leaf in enumerate(leaves):
      logger.debug(f"{leaf_idx}, {leaf}")
      if leaf_idx == 0:
        continue

      if leaf_idx - 1 < vis.shape[0]:
        # Leaf #0 is ignored by visibility calculations, so we map a BSP leaf index to a visibility
        # leaf index by subtracting one.
        vis_full = vis[leaf_idx - 1]
        vis_bits = np.packbits(vis_full, bitorder="little")
        vis_bits_compressed = compress_visbits(bytearray(vis_bits))
        leaf_vis_offsets[leaf_idx] = len(vis_lump)
        vis_lump.extend(vis_bits_compressed)
      else:
        leaf_vis_offsets[leaf_idx] = -1

  for leaf_idx, leaf in enumerate(leaves):
    logger.debug(f"{leaf_idx}, {leaf}")

    old_vis_offset = leaf[1]
    logger.debug(f"  {old_vis_offset=}")
    if leaf_idx == 0:
      patched_leaves.append(leaf)
      continue
    vis_offset = leaf_vis_offsets.get(leaf_idx, -1)
    logger.debug(f"  Set new {vis_offset=}")
    patched_leaves.append((leaf[0], vis_offset, *leaf[2:]))

  leaf_lump = write_leaves(patched_leaves, version)
  assert len(lumps_data[Q1Lump.LEAFS]) == len(leaf_lump)

  to_replace = {Q1Lump.LEAFS: leaf_lump, Q1Lump.VISIBILITY: vis_lump}

  with open(old_path, "rb") as inf:
    with open(new_path, "wb") as outf:
      outf.write(inf.read(4))
      ofs_header = outf.tell()
      for i in range(BSP_LUMPS):
        outf.write(struct.pack("<ii", 0, 0))

      lump_ofs_sizes = {}

      for lump_id, old_ofs, old_size in info:
        new_ofs = outf.tell()
        if lump_id in to_replace:
          new_size = len(to_replace[lump_id])
          outf.write(to_replace[lump_id])
        else:
          new_size = old_size
          inf.seek(old_ofs)
          outf.write(inf.read(new_size))
        lump_ofs_sizes[lump_id] = (new_ofs, new_size)
        while outf.tell() % 4 == 0:
          outf.write(bytes([0]))

      logger.debug("New lumps:")
      outf.seek(ofs_header)
      for i in range(BSP_LUMPS):
        key = Q1Lump(i)
        ofs, size = lump_ofs_sizes[key]
        logger.debug(f"{key.name:<15} {ofs:<8} {size:<8}")
        outf.write(struct.pack("<ii", ofs, size))

  print(f"File {new_path} written")
