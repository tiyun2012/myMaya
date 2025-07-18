# UVToolMain.py

# ===== Imports =====
from PySide2 import QtWidgets, QtCore
from shiboken2 import wrapInstance
import maya.OpenMayaUI as omui
import maya.cmds as cmds
import maya.api.OpenMaya as om
import itertools
import math
import json


def ShellTriangleInfo(mesh_fn: om.MFnMesh):
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
        shell_id = shell_ids[face_uvs[0]]
        shell_triangle_map.setdefault(shell_id, [])
        for i in range(1, n_verts-1):
            tri = [face_uvs[0], face_uvs[i], face_uvs[i+1]]
            shell_triangle_map[shell_id].append(tri)
        poly_iter.next()
    return shell_triangle_map


def uv_shell_bbox_intersections(mesh_fn: om.MFnMesh, offset_uv=0.001):
    # Minimal implementation for demo
    shell_to_triangles = ShellTriangleInfo(mesh_fn)
    return shell_to_triangles


def run_uv_analysis(self):
    mesh_name = self.mesh_combo.currentText()
    if mesh_name == "No meshes found":
        self.result_text.setPlainText("No valid mesh selected.")
        return

    try:
        sel = om.MSelectionList()
        sel.add(mesh_name)
        dag = sel.getDagPath(0)
        mesh_fn = om.MFnMesh(dag)

        results = uv_shell_bbox_intersections(mesh_fn)
        self.result_text.setPlainText(str(results))

    except Exception as e:
        self.result_text.setPlainText(f"Error: {e}")
class UVMainTool(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super(UVMainTool, self).__init__(parent or self.get_maya_window())
        self.setWindowTitle("UV Main Tool")
        self.setMinimumWidth(400)
        self.setLayout(QtWidgets.QVBoxLayout())

        self.init_ui()

    def init_ui(self):
        # Add 'Add New UDim Group' button
        self.add_udim_btn = QtWidgets.QPushButton("Add New UDim Group")
        self.add_udim_btn.clicked.connect(self.add_new_udim_group)
        self.layout().addWidget(self.add_udim_btn)

        # Placeholder: Area to display added UDim groups
        self.udim_group_area = QtWidgets.QVBoxLayout()
        self.layout().addLayout(self.udim_group_area)
        # Save/Load/Clean controls group
        save_group = QtWidgets.QGroupBox("Group Persistence")
        save_layout = QtWidgets.QHBoxLayout()

        self.enable_save_checkbox = QtWidgets.QCheckBox("Enable Save/Load")
        self.enable_save_checkbox.setChecked(False)
        save_layout.addWidget(self.enable_save_checkbox)

        save_btn = QtWidgets.QPushButton("Save Groups")
        save_btn.clicked.connect(self.save_groups)
        save_layout.addWidget(save_btn)

        load_btn = QtWidgets.QPushButton("Load Groups")
        load_btn.clicked.connect(self.load_groups)
        save_layout.addWidget(load_btn)

        clean_btn = QtWidgets.QPushButton("Clean Groups")
        clean_btn.clicked.connect(self.clean_groups)
        save_layout.addWidget(clean_btn)

        save_group.setLayout(save_layout)
        self.layout().addWidget(save_group)


    def add_new_udim_group(self):
        # Called when button is clicked
        group_widget = UDimGroupWidget()
        self.udim_group_area.addWidget(group_widget)
    def save_groups_to_fileinfo(groups_data):
        # Save UDim groups to Maya file info
        cmds.fileInfo("UVToolGroups", json.dumps(groups_data))

    def load_groups_from_fileinfo():
        # Load UDim groups from Maya file info
        if cmds.fileInfo("UVToolGroups", query=True):
            data_str = cmds.fileInfo("UVToolGroups", query=True)[0]
            try:
                return json.loads(data_str)
            except json.JSONDecodeError:
                om.MGlobal.displayWarning("Failed to parse UVToolGroups data.")
        return []

    def clean_groups_from_fileinfo():
        # Remove UDim groups from Maya file info
        cmds.fileInfo("UVToolGroups", remove=True)
    def save_groups(self):
        if not self.enable_save_checkbox.isChecked():
            om.MGlobal.displayInfo("Save/Load is disabled.")
            return

        groups_data = []
        for i in range(self.udim_group_area.count()):
            group_widget = self.udim_group_area.itemAt(i).widget()
            meshes = [group_widget.mesh_list_widget.item(j).text() for j in range(group_widget.mesh_list_widget.count())]
            groups_data.append({
                "group_id": i + 1,
                "meshes": meshes
            })

        save_groups_to_fileinfo(groups_data)
        om.MGlobal.displayInfo("Groups saved to fileInfo.")

    def load_groups(self):
        if not self.enable_save_checkbox.isChecked():
            om.MGlobal.displayInfo("Save/Load is disabled.")
            return

        groups_data = load_groups_from_fileinfo()
        if not groups_data:
            om.MGlobal.displayInfo("No saved group data found.")
            return

        # Clear existing UI groups first if desired
        for i in reversed(range(self.udim_group_area.count())):
            widget = self.udim_group_area.itemAt(i).widget()
            widget.setParent(None)

        # Recreate groups
        for group in groups_data:
            group_widget = UDimGroupWidget()
            for mesh in group["meshes"]:
                if cmds.objExists(mesh):
                    group_widget.mesh_list_widget.addItem(mesh)
            self.udim_group_area.addWidget(group_widget)

        om.MGlobal.displayInfo("Groups loaded from fileInfo.")

    def clean_groups(self):
        clean_groups_from_fileinfo()
        om.MGlobal.displayInfo("Group data cleaned from fileInfo.")

    @staticmethod
    def get_maya_window():
        ptr = omui.MQtUtil.mainWindow()
        return wrapInstance(int(ptr), QtWidgets.QWidget)
class UDimGroupWidget(QtWidgets.QGroupBox):
    def __init__(self, parent=None):
        super(UDimGroupWidget, self).__init__(parent)
        self.setTitle("New UDim Group")
        self.setLayout(QtWidgets.QVBoxLayout())

        # Mesh list display
        self.mesh_list_widget = QtWidgets.QListWidget()
        self.mesh_list_widget.setSelectionMode(QtWidgets.QAbstractItemView.MultiSelection)
        self.layout().addWidget(self.mesh_list_widget)

        # Add / Remove buttons
        btn_layout = QtWidgets.QHBoxLayout()

        add_btn = QtWidgets.QPushButton("Add Meshes")
        add_btn.clicked.connect(self.add_meshes)
        btn_layout.addWidget(add_btn)

        remove_btn = QtWidgets.QPushButton("Remove Selected")
        remove_btn.clicked.connect(self.remove_selected_meshes)
        btn_layout.addWidget(remove_btn)

        self.layout().addLayout(btn_layout)

        # Run Analysis and Clear Results buttons in a horizontal layout
        button_layout = QtWidgets.QHBoxLayout()

        run_btn = QtWidgets.QPushButton("Run UV Analysis")
        run_btn.clicked.connect(self.run_uv_analysis)
        button_layout.addWidget(run_btn)

        clear_btn = QtWidgets.QPushButton("Clear Results")
        clear_btn.clicked.connect(self.clear_results)
        button_layout.addWidget(clear_btn)

        self.layout().addLayout(button_layout)


        # Results display
        self.result_text = QtWidgets.QTextEdit()
        self.result_text.setReadOnly(True)
        self.layout().addWidget(self.result_text)
    def clear_results(self):
        self.result_text.clear()

    def add_meshes(self):
        # Populate mesh list from scene
        meshes = cmds.ls(type="mesh", long=True)
        valid_meshes = [m for m in meshes if not cmds.getAttr(m + ".intermediateObject")]

        for mesh in valid_meshes:
            if not self.mesh_already_in_list(mesh):
                self.mesh_list_widget.addItem(mesh)

    def remove_selected_meshes(self):
        for item in self.mesh_list_widget.selectedItems():
            row = self.mesh_list_widget.row(item)
            self.mesh_list_widget.takeItem(row)

    def mesh_already_in_list(self, mesh_name):
        for index in range(self.mesh_list_widget.count()):
            if self.mesh_list_widget.item(index).text() == mesh_name:
                return True
        return False

    def run_uv_analysis(self):
        selected_meshes = [self.mesh_list_widget.item(i).text() for i in range(self.mesh_list_widget.count())]
        if not selected_meshes:
            self.result_text.setPlainText("No meshes in this UDim group.")
            return

        results = []
        for mesh_name in selected_meshes:
            try:
                sel = om.MSelectionList()
                sel.add(mesh_name)
                dag = sel.getDagPath(0)
                mesh_fn = om.MFnMesh(dag)

                analysis = uv_shell_bbox_intersections(mesh_fn)
                results.append(f"{mesh_name}: {analysis}")

            except Exception as e:
                results.append(f"{mesh_name}: Error: {e}")

        self.result_text.setPlainText("\n".join(results))


def show_ui():
    global window
    try:
        window.close()
        window.deleteLater()
    except:
        pass
    window = UVMainTool()
    window.show()
