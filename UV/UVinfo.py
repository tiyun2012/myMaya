import itertools
import math
from typing import List, Tuple, Set, Dict, Any
# UV Analysis Tool with PySide2
import maya.api.OpenMaya as om
import maya.cmds as cmds
from functools import partial

from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui

import os
from sparx.qc_framework.lib import decorator

import asset.qc.common as qc_common



# === Import your existing functions here ===
# Assuming they are in uv_utils.py (or use from this file directly if in same script)
# f
# Constants
TEXTURE_SIZE = 4096
MIN_SHELL_DISTANCE_TEXTURE_DIS = 32
BORDER_MIN = 16.0
OFFSET_UV = MIN_SHELL_DISTANCE_TEXTURE_DIS / TEXTURE_SIZE
MIN_UV = BORDER_MIN / TEXTURE_SIZE
VALID_BOX = ((MIN_UV, MIN_UV), (1 - MIN_UV, 1 - MIN_UV))
IDENTICAL_PIXEL=4

def uvIdConnectedTriangles(mesh_name: str, uv_id: int) -> List[Tuple[int, int, int]]:
    """
    Find all triangles (as UV index triplets) connected to a given UV index on a mesh.

    Args:
        mesh_name (str): Name of the mesh shape node.
        uv_id (int): The UV index to query.

    Returns:
        List[Tuple[int, int, int]]: List of triangles (each as 3 UV indices) that include the given UV.

    Raises:
        RuntimeError: If the mesh cannot be accessed.
        IndexError: If the UV index is out of range.
    """
    # Initialize mesh
    try:
        sel = om.MSelectionList()
        sel.add(mesh_name)
        dag = sel.getDagPath(0)
        mesh_fn = om.MFnMesh(dag)
    except RuntimeError as e:
        raise RuntimeError(f"Failed to access mesh '{mesh_name}': {e}")

    # Validate UV index
    u_coords, _ = mesh_fn.getUVs()
    num_uvs = len(u_coords)
    if uv_id < 0 or uv_id >= num_uvs:
        raise IndexError(f"UV index {uv_id} out of range 0-{num_uvs-1}")

    connected_tris = []
    poly_iter = om.MItMeshPolygon(dag)

    while not poly_iter.isDone():
        try:
            # Get triangle vertex indices for the current face
            triangle_verts, triangle_indices = poly_iter.getTriangles()
            if not triangle_verts:
                poly_iter.next()
                continue

            # Get UV indices for the face
            face_uvs = [poly_iter.getUVIndex(i) for i in range(
                poly_iter.polygonVertexCount())]

            # Map vertex indices to UV indices for each triangle
            for i in range(0, len(triangle_verts), 3):
                vert_ids = triangle_verts[i:i+3]
                if len(vert_ids) != 3:
                    continue  # Skip if not a triangle

                # Get UV indices for the triangle's vertices
                try:
                    tri_uvs = [face_uvs[triangle_indices[i + j]]
                               for j in range(3)]
                    if uv_id in tri_uvs:
                        connected_tris.append(tuple(tri_uvs))
                except (IndexError, RuntimeError):
                    continue  # Skip invalid UV mappings

        except RuntimeError:
            pass  # Skip faces with invalid data
        poly_iter.next()

    return connected_tris


def triangles_intersect(mesh_fn: om.MFnMesh, uv_ids1: List[int], uv_ids2: List[int], epsilon: float = 1e-6) -> bool:
    """
    Determine if two triangles in UV space intersect.

    Args:
        mesh_fn (om.MFnMesh): The mesh function set.
        uv_ids1 (List[int]): UV indices for the first triangle.
        uv_ids2 (List[int]): UV indices for the second triangle.
        epsilon (float, optional): Tolerance for intersection and AABB checks.

    Returns:
        bool: True if triangles intersect, False otherwise.
    """
    if len(uv_ids1) != 3 or len(uv_ids2) != 3:
        om.MGlobal.displayWarning(
            "Each triangle must have exactly 3 UV indices.")
        return False

    # Helper: Get UV coordinates
    def get_uv_coords(uv_ids):
        coords = []
        for uv_id in uv_ids:
            try:
                u, v = mesh_fn.getUV(uv_id)
                coords.append((u, v))
            except RuntimeError:
                om.MGlobal.displayWarning(f"Invalid UV index: {uv_id}")
                return None
        return coords

    T1 = get_uv_coords(uv_ids1)
    T2 = get_uv_coords(uv_ids2)
    if not T1 or not T2:
        return False

    # AABB test with symmetric epsilon
    def get_aabb(triangle):
        us = [p[0] for p in triangle]
        vs = [p[1] for p in triangle]
        return (min(us), max(us), min(vs), max(vs))

    min_u1, max_u1, min_v1, max_v1 = get_aabb(T1)
    min_u2, max_u2, min_v2, max_v2 = get_aabb(T2)

    if max_u1 < min_u2 - epsilon or min_u1 > max_u2 + epsilon or max_v1 < min_v2 - epsilon or min_v1 > max_v2 + epsilon:
        # om.MGlobal.displayInfo("AABB test: No overlap")
        return False

    # Check for identical triangles
    if set(uv_ids1) == set(uv_ids2):
        om.MGlobal.displayInfo(
            "Triangles have identical UV indices, intersect by definition")
        return True

    # SAT: Get edge perpendiculars
    def get_edge_perpendicular(p1, p2):
        return (p1[1] - p2[1], p2[0] - p1[0])

    def project_vertices(triangle, axis):
        return [p[0] * axis[0] + p[1] * axis[1] for p in triangle]

    axes = []
    for i in range(3):
        p1, p2 = T1[i], T1[(i + 1) % 3]
        normal = get_edge_perpendicular(p1, p2)
        if abs(normal[0]) >= epsilon or abs(normal[1]) >= epsilon:
            axes.append(normal)
        p1, p2 = T2[i], T2[(i + 1) % 3]
        normal = get_edge_perpendicular(p1, p2)
        if abs(normal[0]) >= epsilon or abs(normal[1]) >= epsilon:
            axes.append(normal)

    for axis in axes:
        proj1 = project_vertices(T1, axis)
        proj2 = project_vertices(T2, axis)
        if max(proj1) + epsilon < min(proj2) or max(proj2) + epsilon < min(proj1):
            # om.MGlobal.displayInfo(f"Separating axis found: {axis}")
            return False

    # om.MGlobal.displayInfo("No separating axis found, triangles intersect")
    return True


def triangle_distance(mesh_fn: om.MFnMesh, uv_ids1: List[int], uv_ids2: List[int], uv_offset: float = 0.001) -> Tuple[float, List[int], Dict[str, Any]]:
    """
    Compute the minimum distance between two triangles in UV space.

    Args:
        mesh_fn (om.MFnMesh): The mesh function set.
        uv_ids1 (List[int]): UV indices for the first triangle.
        uv_ids2 (List[int]): UV indices for the second triangle.
        uv_offset (float, optional): Distance threshold for reporting close triangles.

    Returns:
        Tuple[float, List[int], Dict[str, Any]]:
            - Minimum distance between triangles.
            - List of overlapping UV indices (if any).
            - Dictionary describing the closest UV relationship (type and ids).
    """
    overlapIds = []
    closestUV = {'type': None, 'ids': []}
    closestUVInfo = {'type': None, 'ids': [], ' distance': float('inf')}

    if triangles_intersect(mesh_fn, uv_ids1, uv_ids2):
        overlapIds = uv_ids1 + uv_ids2
        return 0.0, overlapIds, closestUV, closestUVInfo

    def get_uv_coords(uv_ids):
        coords = []
        for uv_id in uv_ids:
            try:
                u, v = mesh_fn.getUV(uv_id)
                coords.append((u, v))
            except RuntimeError:
                om.MGlobal.displayWarning(f"Invalid UV index: {uv_id}")
                return None
        return coords

    T1 = get_uv_coords(uv_ids1)
    T2 = get_uv_coords(uv_ids2)
    if not T1 or not T2:
        return float('inf'), overlapIds, closestUV, closestUVInfo

    def point_distance(p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def point_to_edge_distance(p, e1, e2, edge_idx1, edge_idx2):
        ab = (e2[0] - e1[0], e2[1] - e1[1])
        ap = (p[0] - e1[0], p[1] - e1[1])
        ab_len_sq = ab[0]**2 + ab[1]**2
        if ab_len_sq < 1e-10:
            dist1 = point_distance(p, e1)
            dist2 = point_distance(p, e2)
            return min(dist1, dist2), [edge_idx1] if dist1 <= dist2 else [edge_idx2]
        t = max(0, min(1, (ap[0] * ab[0] + ap[1] * ab[1]) / ab_len_sq))
        closest = (e1[0] + t * ab[0], e1[1] + t * ab[1])
        return point_distance(p, closest), [edge_idx1, edge_idx2]

    min_dist = float('inf')
    closest_pair = None
    closest_type = None

    # Vertex-Vertex
    for i, p1 in enumerate(T1):
        for j, p2 in enumerate(T2):
            dist = point_distance(p1, p2)
            if dist < min_dist:
                min_dist = dist
                closest_pair = (uv_ids1[i], uv_ids2[j])
                closest_type = 'vertex-vertex'

    # Vertex-Edge (T1 -> T2)
    for i, p in enumerate(T1):
        for j in range(3):
            e1, e2 = T2[j], T2[(j + 1) % 3]
            dist, edge_ids = point_to_edge_distance(
                p, e1, e2, uv_ids2[j], uv_ids2[(j + 1) % 3])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (uv_ids1[i], edge_ids)
                closest_type = 'vertex-edge'

    # Vertex-Edge (T2 -> T1)
    for i, p in enumerate(T2):
        for j in range(3):
            e1, e2 = T1[j], T1[(j + 1) % 3]
            dist, edge_ids = point_to_edge_distance(
                p, e1, e2, uv_ids1[j], uv_ids1[(j + 1) % 3])
            if dist < min_dist:
                min_dist = dist
                closest_pair = (uv_ids2[i], edge_ids)
                closest_type = 'vertex-edge'

    if min_dist < uv_offset:
        if closest_type == 'vertex-vertex':
            closestUV = {'type': 'vertex-vertex',
                         'ids': [closest_pair[0], closest_pair[1]]}
            closestUVInfo = {'type': 'vertex-vertex',
                             'ids': [closest_pair[0], closest_pair[1]], 'distance': min_dist}
        elif closest_type == 'vertex-edge':
            vertex_id, edge_ids = closest_pair
            closestUV = {'type': 'vertex-edge', 'ids': [vertex_id] + edge_ids}
            closestUVInfo = {'type': 'vertex-edge',
                             'ids': [vertex_id] + edge_ids, 'distance': min_dist}

    # om.MGlobal.displayInfo(f"Calculated minimum distance: {min_dist}")
    return min_dist, overlapIds, closestUV, closestUVInfo


def ShellTriangleInfo(mesh_fn: om.MFnMesh):
    """
    Build a mapping from UV shell ID to all triangles (as lists of 3 UV indices) in that shell.

    Args:
        mesh_fn (om.MFnMesh): The mesh function set.

    Returns:
        Dict[int, List[List[int]]]: Mapping from shell ID to list of triangles (each as 3 UV indices).
    """
    # Get the shell assignment for each UV
    _, shell_ids = mesh_fn.getUvShellsIds()
    shell_triangle_map = {}

    poly_iter = om.MItMeshPolygon(mesh_fn.dagPath())
    while not poly_iter.isDone():
        n_verts = poly_iter.polygonVertexCount()
        try:
            face_uvs = [poly_iter.getUVIndex(i) for i in range(n_verts)]
            if -1 in face_uvs or not face_uvs:
                poly_iter.next()
                continue
        except RuntimeError:
            poly_iter.next()
            continue

        # Use the shell ID of the first UV (or all, if you want to check they are the same)
        shell_id = shell_ids[face_uvs[0]]
        if shell_id not in shell_triangle_map:
            shell_triangle_map[shell_id] = []

        # Fan triangulation: [0,1,2], [0,2,3], [0,3,4] etc for ngons
        for i in range(1, n_verts-1):
            tri = [face_uvs[0], face_uvs[i], face_uvs[i+1]]
            shell_triangle_map[shell_id].append(tri)

        poly_iter.next()

    return shell_triangle_map


def uv_shell_bbox_intersections(
    mesh_fn: om.MFnMesh,
    offset_uv: float = OFFSET_UV,
    valid_box: Tuple[Tuple[float, float], Tuple[float, float]] = VALID_BOX,
    tolerance: float = 1e-6, identical_offset: float = 0.0001
) -> Tuple[Set[int], Set[Tuple[int, ...]], Set[int], List[List[List[int]]], list, list]:
    """
    Analyze UV shells for intersections, proximity, and bounding box validity.

    Args:
        mesh_fn (om.MFnMesh): The mesh function set.
        offset_uv (float, optional): Minimum allowed UV distance between shells.
        valid_box (Tuple[Tuple[float, float], Tuple[float, float]], optional): UV bounding box for validity.
        tolerance (float, optional): Tolerance for floating point comparisons.

    Returns:
        Tuple containing:
            - overlap_ids: the first Pair of overlapping UV (normally 6 uv id).
            - closest_uvs: case not intersecting, but uv ids close under offset_uv.
            - out_valid_border:Set of UV indices outside the valid UV box.
            - valid_overlays:List of shell pairs with identical bounding boxes.
            - shellIdOverlay:List of shell ID pairs with overlaps.
            - shell_ids_list:List of all shell IDs.
    """
    overlap_ids = set()
    closest_uvs = set()
    out_valid_border = set()
    valid_overlays = []
    shellIdOverlay = []
    closestUVInfos = []  # use to replace closest_uvs
    # Use ShellTriangleInfo to get triangles per shell
    shell_to_triangles = ShellTriangleInfo(mesh_fn)
    shell_ids_list = list(shell_to_triangles.keys())
    if len(shell_ids_list) < 2:
        return overlap_ids, closest_uvs, out_valid_border, valid_overlays, shellIdOverlay, shell_ids_list,closestUVInfos
    # Get UV shells
    _, shell_ids = mesh_fn.getUvShellsIds()
    shell_to_uvs = {}
    for uv_id, shell_id in enumerate(shell_ids):
        shell_to_uvs.setdefault(shell_id, []).append(uv_id)

    # Compute shell bounding boxes
    shell_bboxes = {}
    for shell_id, uv_ids in shell_to_uvs.items():
        us, vs = [], []
        for uv_id in uv_ids:
            try:
                u, v = mesh_fn.getUV(uv_id)
                us.append(u)
                vs.append(v)
                min_u, min_v = valid_box[0]
                max_u, max_v = valid_box[1]
                if u < min_u or u > max_u or v < min_v or v > max_v:
                    out_valid_border.add(uv_id)
            except RuntimeError:
                om.MGlobal.displayWarning(f"Invalid UV index: {uv_id}")
        if us and vs:
            shell_bboxes[shell_id] = (min(us), max(us), min(vs), max(vs))

    def aabbs_identical(bbox1, bbox2):
        return all(abs(a - b) < (tolerance+identical_offset) for a, b in zip(bbox1, bbox2))

    def aabb_distance(bbox1, bbox2):
        du = max(0, max(bbox1[0] - bbox2[1], bbox2[0] - bbox1[1]))
        dv = max(0, max(bbox1[2] - bbox2[3], bbox2[2] - bbox1[3]))
        return math.hypot(du, dv)

    def aabb_overlap(bbox1, bbox2, tolerance=1e-6):
        min_u1, max_u1, min_v1, max_v1 = bbox1
        min_u2, max_u2, min_v2, max_v2 = bbox2
        min_u = max(min_u1, min_u2)
        max_u = min(max_u1, max_u2)
        min_v = max(min_v1, min_v2)
        max_v = min(max_v1, max_v2)
        return max_u >= min_u - tolerance and max_v >= min_v - tolerance

    # Compare shell pairs

    is_checked_valid_overlays = set()
    for i, shell1_id in enumerate(shell_ids_list):
        checked_valid_overlays = []
        for shell2_id in shell_ids_list[i + 1:]:
            bbox1, bbox2 = shell_bboxes.get(
                shell1_id), shell_bboxes.get(shell2_id)
            if not bbox1 or not bbox2:
                continue
            if shell2_id in is_checked_valid_overlays:
                continue
            if aabbs_identical(bbox1, bbox2):
                # valid_overlays.append([shell_to_uvs[shell1_id], shell_to_uvs[shell2_id]])
                # valid_overlays.append([shell1_id, shell2_id])
                checked_valid_overlays.append(shell2_id)
                is_checked_valid_overlays.add(shell2_id)
                continue

            if not aabb_overlap(bbox1, bbox2, tolerance):
                distance = aabb_distance(bbox1, bbox2)
                # print(f"Shells {shell1_id} and {shell2_id} do not overlap, distance={distance}")
                if distance > offset_uv:
                    continue

            # print(f"Processing triangle intersections for shells {shell1_id} and {shell2_id}")
            triangles1 = shell_to_triangles.get(shell1_id, [])
            triangles2 = shell_to_triangles.get(shell2_id, [])

            for tri1, tri2 in itertools.product(triangles1, triangles2):
                # print(f"Checking triangle {tri1} and {tri2} for intersection")
                dist, tri_overlap, tri_closest, closestUVInfo = triangle_distance(
                    mesh_fn, tri1, tri2, offset_uv)
                if tri_closest['ids']:
                    closest_uvs.add(tuple(sorted(tri_closest['ids'])))
                    closestUVInfos.append(closestUVInfo)
                if tri_overlap:
                    shellIdOverlay.append([shell1_id, shell2_id])
                    overlap_ids.add(tuple(sorted(tri_overlap)))
                    break
        if checked_valid_overlays:
            checked_valid_overlays.append(shell1_id)
            valid_overlays.append(checked_valid_overlays)

    return overlap_ids, closest_uvs, out_valid_border, valid_overlays, shellIdOverlay, shell_ids_list, closestUVInfos


def select_uvs_by_shell(mesh_fn: om.MFnMesh, shell_ids: List[int]):
    meshName = mesh_fn.name()
    _, uv_shell_ids = mesh_fn.getUvShellsIds()
    uv_indices = [i for i, shell_id in enumerate(
        uv_shell_ids) if shell_id in shell_ids]
    for uv_index in uv_indices:
        cmds.select(f"{meshName}.map[{uv_index}]", add=True)


def select_uvList(mesh_name, uvList=[]):
    cmds.select(cl=1)
    for u in uvList:
        cmds.select(f"{mesh_name}.map[{u}]", add=1)


def list_all_mesh_shapes():
    """
    List all mesh shape nodes in the scene.

    Returns:
        List[str]: List of mesh shape node names.
    """
    meshes = cmds.ls(type="mesh", long=True)
    # Filter out intermediate shapes if needed
    valid_meshes = [m for m in meshes if not cmds.getAttr(
        m + ".intermediateObject")]
    return valid_meshes


# manage UV of multi Mesh
UVMeshData = {}  # Global dict: {mesh_name: uv_analysis_result}
# overlap_ids, closest_uvs, out_valid_border, valid_overlays,shellIdOverlay,shell_ids_list,closestUVInfos=UVMeshData["|mesh|meshShape"]


def build_uv_mesh_data(TEXTURE_SIZE=4096, MIN_SHELL_DISTANCE_TEXTURE_DIS=32, BORDER_MIN=16.0, identical_offset=0.0001):
    """
    Loop all valid meshes in the scene, run uv_shell_bbox_intersections,
    and store results in global UVMeshData.
    Uses parameters provided.
    """
    global UVMeshData
    UVMeshData = {}  # Reset before building

    meshes = list_all_mesh_shapes()
    if not meshes:
        om.MGlobal.displayWarning("No valid mesh shapes found in scene.")
        return

    OFFSET_UV = MIN_SHELL_DISTANCE_TEXTURE_DIS / TEXTURE_SIZE
    MIN_UV = BORDER_MIN / TEXTURE_SIZE
    VALID_BOX = ((MIN_UV, MIN_UV), (1 - MIN_UV, 1 - MIN_UV))

    for mesh_name in meshes:
        try:
            sel = om.MSelectionList()
            sel.add(mesh_name)
            dag = sel.getDagPath(0)
            mesh_fn = om.MFnMesh(dag)

            # Run analysis and store
            UVMeshData[mesh_name] = uv_shell_bbox_intersections(
                mesh_fn, OFFSET_UV, VALID_BOX, identical_offset=identical_offset)
            # UVMeshData[mesh_name] = result

            om.MGlobal.displayInfo(f"UV analysis completed for {mesh_name}.")

        except Exception as e:
            om.MGlobal.displayWarning(f"Failed on {mesh_name}: {e}")

# overlap_ids, closest_uvs, out_valid_border, valid_overlays,shellIdOverlay,shell_ids_list=uv_shell_bbox_intersections(mesh_fn, OFFSET_UV, VALID_BOX, 0.0001)
# len(valid_overlays)

# === Import your existing functions here ===
# Assuming they are in uv_utils.py (or use from this file directly if in same script)
# from uv_utils import uv_shell_bbox_intersections, select_uvs_by_shell, select_uvList

# Global storage for results
# RESULTS = {}


def get_maya_window():
    ptr = omui.MQtUtil.mainWindow()
    return wrapInstance(int(ptr), QtWidgets.QWidget)


class UVTableInfo(QtWidgets.QDialog):
    selectionChangedSignal = QtCore.Signal(list)

    def __init__(self, dataTable: Dict[str, List[Any]], parent=None, TableName="NoNameYet", ColumeName=["noName1", "noName2"], disabled_columns: List[int] = []):
        super(UVTableInfo, self).__init__(parent)
        self.setWindowTitle(TableName)
        self.setMinimumSize(300, 200)
        layout = QtWidgets.QVBoxLayout(self)

        # Table widget
        self.table = QtWidgets.QTableWidget()
        self.table.setColumnCount(len(ColumeName))
        self.table.setHorizontalHeaderLabels([name for name in ColumeName])
        self.table.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.table.setSelectionBehavior(
            QtWidgets.QAbstractItemView.SelectItems)
        self.table.itemSelectionChanged.connect(self.on_item_selection_changed)
        self.disabled_columns = disabled_columns
        layout.addWidget(self.table)

        # Populate with provided data
        if dataTable is None:
            dataTable = []
        self.populate_table(dataTable)

    def populate_table(self, dataTable: Dict[str, List[Any]]):
        # Check if dataTable is valid
        if not dataTable or not isinstance(dataTable, dict):
            return

        # Ensure all columns have same length
        lengths = [len(v) for v in dataTable.values()]
        if not lengths or len(set(lengths)) != 1:
            raise ValueError(
                "All columns in dataTable must have the same number of rows.")

        # Prepare data as list of rows
        columns = list(dataTable.keys())
        rows = []
        num_rows = lengths[0]

        for i in range(num_rows):
            row = [dataTable[col][i] for col in columns]
            rows.append(row)

        # Now rows is List[List[Any]], pass to table

        column_count = len(columns)
        self.table.setColumnCount(column_count)
        self.table.setHorizontalHeaderLabels(columns)
        self.table.setRowCount(num_rows)

        for row_idx, row_data in enumerate(rows):
            for col_idx, value in enumerate(row_data):
                item = QtWidgets.QTableWidgetItem(str(value))
                # Disable editing for this cell
                item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.table.setItem(row_idx, col_idx, item)

    def refresh_table(self, newDataTable: Dict[str, List[Any]]):
        # Clear table
        self.table.clearContents()
        # Repopulate table with new data
        self.populate_table(newDataTable)

    def on_item_selection_changed(self):
        selected_items = self.table.selectedItems()
        selected_values = []
        for item in selected_items:
            if item.column() in self.disabled_columns:
                continue

            print(f"from TableWidget: Row {item.row()}, Col {item.column()}, Value: {item.text()}")

            text = item.text()
            parts = text.split(',')
            for part in parts:
                part = part.strip()
                if part.isdigit():
                    selected_values.append(int(part))
                else:
                    try:
                        value = int(part)
                        selected_values.append(value)
                    except ValueError:
                        pass  # ignore non-integer parts if any
        print(f"Selected values: {selected_values}")
        self.selectionChangedSignal.emit(selected_values)

class UVAnalysisTool(QtWidgets.QDialog):
    def __init__(self, parent=get_maya_window()):
        super(UVAnalysisTool, self).__init__(parent)
        self.setWindowTitle("UV Analysis Tool")
        self.setMinimumWidth(400)
        self.setLayout(QtWidgets.QVBoxLayout())
        self.init_ui()
        self.refresh_mesh_list()
        # Data summary display
        self.data_display = QtWidgets.QTextEdit()
        self.data_display.setReadOnly(True)
        self.data_display.setMinimumHeight(200)
        self.layout().addWidget(self.data_display)
        # cal build_uv_mesh_data
        # identical_offset = float(self.identical_offset_input.text())/ TEXTURE_SIZE
        # build_uv_mesh_data(TEXTURE_SIZE=4096, MIN_SHELL_DISTANCE_TEXTURE_DIS=32,
        #                    BORDER_MIN=16.0, identical_offset=0.0001)
        # Call once to populate on UI load
        self.update_data_display()
        # -----------------------meshInit-----------------------
        self.meshInit()
        self.update_currentData_mesh()

    def init_ui(self):
        # Mesh selection combo box
        mesh_layout = QtWidgets.QHBoxLayout()
        #
        mesh_label = QtWidgets.QLabel("MESH (fullPath):")
        self.mesh_combo = QtWidgets.QComboBox()
        self.mesh_combo.currentIndexChanged.connect(self.on_mesh_combo_changed)
        #
        refresh_mesh_btn = QtWidgets.QPushButton("↻")
        refresh_mesh_btn.setFixedWidth(30)
        refresh_mesh_btn.clicked.connect(self.refresh_mesh_list)
        mesh_layout.addWidget(mesh_label)
        mesh_layout.addWidget(self.mesh_combo)
        mesh_layout.addWidget(refresh_mesh_btn)
        self.layout().addLayout(mesh_layout)

        # Input fields
        self.texture_size_input = self.create_labeled_input(
            "TEXTURE_SIZE", "4096")
        self.min_shell_dist_input = self.create_labeled_input(
            "MIN_SHELL_DISTANCE_BASED_TEXTURE (pixel)", "32")
        self.border_min_input = self.create_labeled_input("BORDER_MIN (pixel)", "16")
        self.identical_offset_input = self.create_labeled_input(
            "IDENTICAL_OFFSET (pixel)", "4")

        # Refresh button
        refresh_btn = QtWidgets.QPushButton("Refresh Analysis")
        refresh_btn.clicked.connect(self.run_analysis)
        self.layout().addWidget(refresh_btn)

        # Action buttons
        # self.add_action_button("All ShellOverlap", self.select_all_shell_overlap)
        # self.add_action_button("All ClosestUV", self.select_all_closest_uv)
        self.add_action_button("UV out valid border",
                               self.select_uv_out_of_border)
        # self.add_action_button("Valid Overlays", self.select_valid_overlays)
        # --------------- Overlap Information -------------------
        # Create group box
        overlap_uvShell_uv_group = QtWidgets.QGroupBox(
            "overlap_uvShell UV Actions")
        overlap_uvShell_uv_layout = QtWidgets.QHBoxLayout()
        # -----------------------------------OVERLAP UV SHELL-----------------------------------
        # "All overlap_uvShellUV" button
        btn_all_overlap_uvShell = QtWidgets.QPushButton(
            "All Overlap_uvShell")
        btn_all_overlap_uvShell.clicked.connect(self.select_all_shell_overlap)
        overlap_uvShell_uv_layout.addWidget(btn_all_overlap_uvShell)
        # --Show Closest UV Table---
        btn_show_overlap_uvShell = QtWidgets.QPushButton("Overlap_uvShellTable")
        btn_show_overlap_uvShell.clicked.connect(
            self.show_overlap_uvShell_table)
        overlap_uvShell_uv_layout.addWidget(btn_show_overlap_uvShell)
        overlap_uvShell_uv_group.setLayout(overlap_uvShell_uv_layout)
        self.layout().addWidget(overlap_uvShell_uv_group)
        # -----------------------------------Valid Overlap---------------------------------------------
        validOverlap_group = QtWidgets.QGroupBox("Valid Overlap UV Actions")
        validOverlap_layout = QtWidgets.QHBoxLayout()
        # "All Valid Overlap shell" button
        btn_all_validOverlap = QtWidgets.QPushButton("All Valid Overlap shell")
        btn_all_validOverlap.clicked.connect(self.select_valid_overlays)
        validOverlap_layout.addWidget(btn_all_validOverlap)
        # --Show Valid Overlap shell Table---
        btn_show_validOverlap = QtWidgets.QPushButton("Show Valid Overlap Table")
        btn_show_validOverlap.clicked.connect(self.show_validOverlap_table)
        validOverlap_layout.addWidget(btn_show_validOverlap)
        validOverlap_group.setLayout(validOverlap_layout)
        self.layout().addWidget(validOverlap_group)
        # -----------------------------------CLOSEST UV---------------------------------------------
        closest_uv_group = QtWidgets.QGroupBox("Closest UV Actions")
        closest_uv_layout = QtWidgets.QHBoxLayout()
        # "All Closest UV" button
        btn_all_closest_uv = QtWidgets.QPushButton("All Closest UV")
        btn_all_closest_uv.clicked.connect(self.select_all_closest_uv)
        closest_uv_layout.addWidget(btn_all_closest_uv)
        # --Show Closest UV Table---
        btn_show_closest_uv = QtWidgets.QPushButton("Show Closest UV Table")
        btn_show_closest_uv.clicked.connect(self.show_closest_uv_table)
        closest_uv_layout.addWidget(btn_show_closest_uv)
        closest_uv_group.setLayout(closest_uv_layout)
        self.layout().addWidget(closest_uv_group)
    def get_updated_ValidOverlap_data(self):
        if not self.valid_overlays:
            return {"Valid Overlays": []}

        dataTable = {"Valid Overlays": []}
        for overlay in self.valid_overlays:
            dataTable["Valid Overlays"].append(','.join(str(i) for i in overlay))
        return dataTable
    def show_validOverlap_table(self):
        dataTable=self.get_updated_ValidOverlap_data()
        self.validOverlapDialog = UVTableInfo(
            dataTable, parent=self, TableName="Valid Overlays UV Shells",
            ColumeName=["Valid Overlays Shells"])
        self.validOverlapDialog.selectionChangedSignal.connect(self.handle_valid_overlap_selection)
        self.validOverlapDialog.show()

    def handle_valid_overlap_selection(self, selected_values):
        # print("Selection changed in ValidOverlapTable:", selected_values)
        cmds.select(cl=True)  # Clear current selection
        select_uvs_by_shell(self.current_mesh_fn, selected_values)
    def show_closest_uv_table(self):  

        dataTable = self.get_updated_closest_uv_data()
        column_names = [n for n in dataTable.keys()]
        self.closest_uvData_dialog = UVTableInfo(
            dataTable, parent=self, TableName="UV Shell closest_uvData", ColumeName=column_names, disabled_columns=[2, 3])
        self.closest_uvData_dialog.selectionChangedSignal.connect(self.handle_closestuv_selection)
        self.closest_uvData_dialog.setWindowModality(
            QtCore.Qt.NonModal)  # Makes it float
        self.closest_uvData_dialog.show()
    def handle_closestuv_selection(self, selected_values):
        print("Selection changed in ClosestUVTable:", selected_values)
        # Here you can call any function or update selection in Maya as needed
        cmds.select(cl=True)
        for value in selected_values:
            cmds.select(f"{self.current_mesh_name}.map[{value}]", add=True)
    def meshInit(self):
        self.current_mesh_fn = None
        self.current_mesh_name = self.mesh_combo.currentText()
        cmds.select(self.current_mesh_name, r=True)  # Clear current selection
        # Handle case when "No meshes found" is selected
        if self.current_mesh_name == "No meshes found":
            self.current_mesh_fn = None
            return

        # Get MFnMesh for the current mesh
        sel = om.MSelectionList()
        try:
            sel.add(self.current_mesh_name)
            dag = sel.getDagPath(0)
            self.current_mesh_fn = om.MFnMesh(dag)
            om.MGlobal.displayInfo(
                f"Updated current mesh to {self.current_mesh_name}")
        except:
            self.current_mesh_fn = None
            om.MGlobal.displayWarning(
                f"Failed to get MFnMesh for {self.current_mesh_name}")

    def on_mesh_combo_changed(self, index):
        self.meshInit()
        # MAKE sure this function is implemented afer meshInit
        self.update_currentData_mesh()

    def update_currentData_mesh(self):
        self.current_mesh_name = self.mesh_combo.currentText()
        if self.current_mesh_name == None:
            return
        # initialize or update the data attributes
        self.overlap_ids = set()
        self.closest_uvs = set()
        self.out_valid_border = set()
        self.valid_overlays = []
        self.shellIdOverlay = []
        self.shell_ids_list = []
        self.closestUVInfos = []
        global UVMeshData
        if self.current_mesh_name in UVMeshData.keys():
            # overlap_ids, closest_uvs, out_valid_border, valid_overlays,shellIdOverlay,shell_ids_list,closestUVInfos
            (self.overlap_ids, self.closest_uvs,
             self.out_valid_border, self.valid_overlays,
             self.shellIdOverlay, self.shell_ids_list,
             self.closestUVInfos) = UVMeshData[self.current_mesh_name]
            om.MGlobal.displayInfo(
                f"Data updated for {self.current_mesh_name}.")
        else:

            om.MGlobal.displayWarning(
                f"No data found for {self.current_mesh_name}. Please run analysis first.")

    def refresh_mesh_list(self):
        meshes = list_all_mesh_shapes()
        self.mesh_combo.clear()
        if meshes:
            self.mesh_combo.addItems(meshes)
        else:
            self.mesh_combo.addItem("No meshes found")

    def run_analysis(self):
        TEXTURE_SIZE = float(self.texture_size_input.text())
        MIN_SHELL_DISTANCE_TEXTURE_DIS = float(
            self.min_shell_dist_input.text())
        BORDER_MIN = float(self.border_min_input.text())
        identical_offset = float(self.identical_offset_input.text())/ TEXTURE_SIZE
        print(f"_______________________{identical_offset}-------------------------------")

        build_uv_mesh_data(
            TEXTURE_SIZE, MIN_SHELL_DISTANCE_TEXTURE_DIS, BORDER_MIN, identical_offset)
        self.update_data_display()
        self.update_currentData_mesh()
        # update dialog if reanalysis is run
        if hasattr(self, 'overlap_Shell_Id_dialog'):
            self.overlap_Shell_Id_dialog.refresh_table(
                self.get_updated_overlap_shell_data())
        if hasattr(self, 'closest_uvData_dialog'):
            self.closest_uvData_dialog.refresh_table(
                self.get_updated_closest_uv_data()) 
        if hasattr(self, 'validOverlapDialog'):
            self.validOverlapDialog.refresh_table(
                self.get_updated_ValidOverlap_data())
        om.MGlobal.displayInfo(
            "UV analysis refreshed and data display updated.")
    def get_updated_overlap_shell_data(self):
        dataTable = {"Shell ID A": [], "Shell ID B": []}
        for a, b in self.shellIdOverlay:
            dataTable["Shell ID A"].append(a)
            dataTable["Shell ID B"].append(b)
        return dataTable

    def get_updated_closest_uv_data(self):
        TEXTURE_SIZE = float(self.texture_size_input.text())
        dataTable = {"uv shell A": [], "uv shell B": [], "distance": [], "type": []}
        for info in self.closestUVInfos:
            ids = info['ids']
            if len(ids) < 2:
                continue
            uv_shell_a = ids[0]
            uv_shell_b = ','.join(str(i) for i in ids[1:])
            dataTable["uv shell A"].append(uv_shell_a)
            dataTable["uv shell B"].append(uv_shell_b)
            dataTable["distance"].append(math.floor(info['distance']* TEXTURE_SIZE))
            dataTable["type"].append(info['type'])
        return dataTable

    def show_overlap_uvShell_table(self):
        

        dataTable = self.get_updated_overlap_shell_data()
        self.overlap_Shell_Id_dialog = UVTableInfo(
            dataTable, parent=self, TableName="UV Shell Overlap", ColumeName=["Shell ID A", "Shell ID B"])
        self.overlap_Shell_Id_dialog.selectionChangedSignal.connect(self.handle_overlap_uv_selection)
        self.overlap_Shell_Id_dialog.setWindowModality(
            QtCore.Qt.NonModal)  # Makes it float
        self.overlap_Shell_Id_dialog.show()

    def handle_overlap_uv_selection(self, selected_values):
        print("Selection changed in ClosestUVTableTest:", selected_values)
        # Here you can call any function or update selection in Maya as needed
        cmds.select(cl=True)
        select_uvs_by_shell(self.current_mesh_fn, selected_values)

    def create_labeled_input(self, label, default):
        layout = QtWidgets.QHBoxLayout()
        lbl = QtWidgets.QLabel(label)
        edit = QtWidgets.QLineEdit(default)
        layout.addWidget(lbl)
        layout.addWidget(edit)
        self.layout().addLayout(layout)
        return edit

    def add_action_button(self, text, func):
        btn = QtWidgets.QPushButton(text)
        btn.clicked.connect(func)
        self.layout().addWidget(btn)

    def select_all_shell_overlap(self):
        mesh_name = self.mesh_combo.currentText()
        if mesh_name == "No meshes found":
            om.MGlobal.displayWarning("No valid mesh selected.")
            return

        global UVMeshData
        if mesh_name not in UVMeshData:
            om.MGlobal.displayWarning(
                f"No data for mesh: {mesh_name}. Please refresh analysis first.")
            return

        _, _, _, _, shellIdOverlay, _, _ = UVMeshData[mesh_name]

        if not shellIdOverlay:
            om.MGlobal.displayInfo(f"No Shell Overlap data for {mesh_name}.")
            return

        # Select UVs by shell for each overlapping shell pair
        sel = om.MSelectionList()
        sel.add(mesh_name)
        dag = sel.getDagPath(0)
        mesh_fn = om.MFnMesh(dag)
        cmds.select(cl=1)
        for pair in shellIdOverlay:
            select_uvs_by_shell(mesh_fn, pair)

        om.MGlobal.displayInfo(f"Selected all Shell Overlaps for {mesh_name}.")

    def select_all_closest_uv(self):
        mesh_name = self.mesh_combo.currentText()
        if mesh_name == "No meshes found":
            om.MGlobal.displayWarning("No valid mesh selected.")
            return

        global UVMeshData
        if mesh_name not in UVMeshData:
            om.MGlobal.displayWarning(
                f"No data for mesh: {mesh_name}. Please refresh analysis first.")
            return

        _, closest_uvs, _, _, _, _, _ = UVMeshData[mesh_name]

        if not closest_uvs:
            om.MGlobal.displayInfo(f"No Closest UV data for {mesh_name}.")
            return

        # Flatten closest_uvs to a list of unique UV indices
        uv_ids = set()
        for uv_pair in closest_uvs:
            uv_ids.update(uv_pair)

        cmds.select(cl=True)
        for uv_id in uv_ids:
            cmds.select(f"{mesh_name}.map[{uv_id}]", add=True)

        om.MGlobal.displayInfo(f"Selected all Closest UVs for {mesh_name}.")

    def select_uv_out_of_border(self):
        mesh_name = self.mesh_combo.currentText()
        if mesh_name == "No meshes found":
            om.MGlobal.displayWarning("No valid mesh selected.")
            return

        global UVMeshData
        if mesh_name not in UVMeshData:
            om.MGlobal.displayWarning(
                f"No data for mesh: {mesh_name}. Please refresh analysis first.")
            return

        _, _, out_valid_border, _, _, _, _ = UVMeshData[mesh_name]

        if not out_valid_border:
            om.MGlobal.displayInfo(
                f"No UVs out of valid border for {mesh_name}.")
            return

        cmds.select(cl=True)
        for uv_id in out_valid_border:
            cmds.select(f"{self.current_mesh_name}.map[{uv_id}]", add=True)

        om.MGlobal.displayInfo(
            f"Selected all UVs out of valid border for {mesh_name}.")

    def select_valid_overlays(self):
        cmds.select(cl=True)
        for idgrp in self.valid_overlays:
            select_uvs_by_shell(self.current_mesh_fn, idgrp)

            
        
        om.MGlobal.displayInfo(f"Selected all valid overlays for {self.current_mesh_name}.")

    def update_data_display(self):
        TEXTURE_SIZE = float(self.texture_size_input.text())
        global UVMeshData
        if not UVMeshData:
            self.data_display.setPlainText("No UV Mesh Data available.")
            return

        lines = []
        lines.append("===== UV Mesh Data Summary =====\n")
        for mesh_name, data in UVMeshData.items():
            (overlap_ids, closest_uvs, out_valid_border,
             valid_overlays, shellIdOverlay, shell_ids_list, closestUVInfos) = data

            lines.append(f"--- Mesh: {mesh_name} ---")
            lines.append(f"Total Shell IDs: {len(shell_ids_list)}")
            lines.append(f"Overlap IDs count: {len(overlap_ids)}")
            lines.append(f"Closest UVs count: {len(closest_uvs)}")
            lines.append(f"UVs out of valid border: {len(out_valid_border)}")
            lines.append(f"Valid overlays count: {len(valid_overlays)}")
            lines.append(f"Shell ID Overlaps count: {len(shellIdOverlay)}\n")
            if closestUVInfos:
                lines.append("Closest UV Info:")
                for info in closestUVInfos:
                    lines.append(
                        f"  Type: {info['type']}, IDs: {info['ids']}, Distance: {float(math.floor(info['distance'] * TEXTURE_SIZE))} pixels")
            else:
                lines.append("No Closest UV Info available.")

        self.data_display.setPlainText('\n'.join(lines))
    



def summarize_UVMeshData_counts():
    """
    Calculate and return total error count and per-mesh summary list.
    
    Returns:
        errorNum (int): Sum of all counts.
        listErrorinfo (list): Per-mesh summary strings.
    """
    global UVMeshData

    total_overlap_ids = 0
    total_closest_uvs = 0
    total_out_valid_border = 0
    total_valid_overlays = 0
    total_shellIdOverlay = 0

    listErrorinfo = []

    for mesh_name, data in UVMeshData.items():
        (overlap_ids, closest_uvs, out_valid_border,
         valid_overlays, shellIdOverlay, shell_ids_list, closestUVInfos) = data

        overlap_count = len(overlap_ids)
        closest_uvs_count = len(closest_uvs)
        out_valid_border_count = len(out_valid_border)
        valid_overlays_count = len(valid_overlays)
        shellIdOverlay_count = len(shellIdOverlay)

        total_overlap_ids += overlap_count
        total_closest_uvs += closest_uvs_count
        total_out_valid_border += out_valid_border_count
        total_valid_overlays += valid_overlays_count
        total_shellIdOverlay += shellIdOverlay_count

        # Append per mesh summary
        mesh_summary = (
            f"Mesh: {mesh_name}\n"
            f"  Overlap IDs count: {overlap_count}\n"
            f"  Closest UVs count: {closest_uvs_count}\n"
            f"  UVs out of valid border: {out_valid_border_count}\n"
            f"  Valid overlays count: {valid_overlays_count}\n"
            f"  Shell ID Overlaps count: {shellIdOverlay_count}\n"
        )
        listErrorinfo.append(mesh_summary)

    # Calculate total error number
    errorNum = (total_overlap_ids + total_closest_uvs +
                total_out_valid_border + total_valid_overlays +
                total_shellIdOverlay)

    # Append total summary at the end
    total_summary = (
        "===== TOTAL UV Data Summary =====\n"
        f"Total Overlap IDs count: {total_overlap_ids}\n"
        f"Total Closest UVs count: {total_closest_uvs}\n"
        f"Total UVs out of valid border: {total_out_valid_border}\n"
        f"Total Valid overlays count: {total_valid_overlays}\n"
        f"Total Shell ID Overlaps count: {total_shellIdOverlay}\n"
        f"Total ErrorNum: {errorNum}"
    )
    listErrorinfo.append(total_summary)

    return errorNum, listErrorinfo




# Show UI
def show_ui():
    global window
    try:
        window.close()
        window.deleteLater()
    except:
        pass
    window = UVAnalysisTool()
    window.show()


# show_ui()

__id__ = "#7040"
__nice__ = "UV Info"
__desc__ = "UV Info"

@decorator.nodefunction
@decorator.log_decorator
def run(*args):
    
    # Get all file nodes in the scene
    identical_offset=IDENTICAL_PIXEL / TEXTURE_SIZE
    build_uv_mesh_data(TEXTURE_SIZE=TEXTURE_SIZE, MIN_SHELL_DISTANCE_TEXTURE_DIS=MIN_SHELL_DISTANCE_TEXTURE_DIS,
                           BORDER_MIN=BORDER_MIN, identical_offset=identical_offset)
 
    err,listErrorinfo=summarize_UVMeshData_counts()
    result = err == 0
    global UVMeshData
    err_list=listErrorinfo
    return result, err_list, ''
def fix(flag, err):
    """
    Show the result of the QC check.
    """
    show_ui()


