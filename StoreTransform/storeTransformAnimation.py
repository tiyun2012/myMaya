from PySide2 import QtWidgets, QtCore
import maya.cmds as cmds
import json
import os

class TransformExporterTool(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(TransformExporterTool, self).__init__(parent)
        self.setWindowTitle("Transform Exporter Tool")
        self.setGeometry(100, 100, 400, 300)
        
        # UI Elements
        layout = QtWidgets.QVBoxLayout()
        
        # Start Frame
        self.start_frame_label = QtWidgets.QLabel("Start Frame:")
        self.start_frame_input = QtWidgets.QLineEdit(str(cmds.playbackOptions(q=True, minTime=True)))
        
        # End Frame
        self.end_frame_label = QtWidgets.QLabel("End Frame:")
        self.end_frame_input = QtWidgets.QLineEdit(str(cmds.playbackOptions(q=True, maxTime=True)))
        
        # JSON File Path
        self.json_path_label = QtWidgets.QLabel("JSON File Path:")
        self.json_path_input = QtWidgets.QLineEdit()
        self.browse_button = QtWidgets.QPushButton("Browse")
        self.browse_button.clicked.connect(self.select_json_file)
        
        # Transform Selection Tree
        self.tree_label = QtWidgets.QLabel("Select Transforms to Save:")
        self.tree_widget = QtWidgets.QTreeWidget()
        self.tree_widget.setHeaderLabels(["Transform Objects"])
        self.populate_tree()
        
        # Execute Button
        self.execute_button = QtWidgets.QPushButton("Execute")
        self.execute_button.clicked.connect(self.export_transform_data)
        
        # Layout
        layout.addWidget(self.start_frame_label)
        layout.addWidget(self.start_frame_input)
        layout.addWidget(self.end_frame_label)
        layout.addWidget(self.end_frame_input)
        layout.addWidget(self.json_path_label)
        layout.addWidget(self.json_path_input)
        layout.addWidget(self.browse_button)
        layout.addWidget(self.tree_label)
        layout.addWidget(self.tree_widget)
        layout.addWidget(self.execute_button)
        
        self.setLayout(layout)
    
    def select_json_file(self):
        file_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select JSON File", "", "JSON Files (*.json)")
        if file_path:
            self.json_path_input.setText(file_path)
    
    def populate_tree(self):
        self.tree_widget.clear()
        transforms = cmds.ls(type='transform')
        for obj in transforms:
            item = QtWidgets.QTreeWidgetItem([obj])
            item.setCheckState(0, QtCore.Qt.Unchecked)
            self.tree_widget.addTopLevelItem(item)
    
    def get_selected_objects(self):
        selected_objects = []
        for i in range(self.tree_widget.topLevelItemCount()):
            item = self.tree_widget.topLevelItem(i)
            if item.checkState(0) == QtCore.Qt.Checked:
                selected_objects.append(item.text(0))
        return selected_objects
    
    def export_transform_data(self):
        start_frame = int(float(self.start_frame_input.text()))
        end_frame = int(float(self.end_frame_input.text()))

        json_path = self.json_path_input.text()
        selected_objects = self.get_selected_objects()
        
        if not selected_objects:
            cmds.warning("No objects selected for export!")
            return
        
        if not json_path:
            cmds.warning("Please specify a JSON file path!")
            return
        
        transform_data = []
        
        for frame in range(start_frame, end_frame + 1):
            cmds.currentTime(frame, edit=True)
            for obj in selected_objects:
                if cmds.objExists(obj):
                    translation = cmds.xform(obj, query=True, translation=True, worldSpace=False)
                    rotation = cmds.xform(obj, query=True, rotation=True, worldSpace=False)
                    scale = cmds.xform(obj, query=True, scale=True, relative=True)
                    transform_data.append({
                        "name": obj,
                        "frame": frame,
                        "translation": translation,
                        "rotation": rotation,
                        "scale": scale
                    })
        
        # Save data to JSON
        with open(json_path, 'w') as json_file:
            json.dump(transform_data, json_file, indent=4)
        
        cmds.confirmDialog(title="Export Complete", message=f"Transform data saved to {json_path}")

# Run the tool
def run():
    global tool_window
    tool_window = TransformExporterTool()
    tool_window.show()

run()
