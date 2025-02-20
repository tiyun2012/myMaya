#!/usr/bin/env mayapy
"""
Usage: mayapy export_single.py <maya_file> <output_folder>

Exports the specified Maya file to FBX.
"""

import os
import sys

if len(sys.argv) < 3:
    sys.stderr.write("Usage: export_single.py <maya_file> <output_folder>\n")
    sys.exit(1)

maya_file = sys.argv[1]
output_folder = sys.argv[2]

# --- Set environment variables and Qt attribute before any Qt UI is created ---
os.environ['MAYA_DISABLE_CLICKTONE'] = '1'
os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"

try:
    from PySide2.QtCore import QCoreApplication, Qt
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
except Exception as e:
    sys.stderr.write("Failed to set Qt attribute: {}\n".format(e))

# --- Initialize Maya Standalone ---
import maya.standalone
maya.standalone.initialize(name='python')

import maya.cmds as cmds
import maya.mel as mel

# --- Helper function to cleanly exit on failure ---
def exit_with_error(msg):
    sys.stderr.write(msg + "\n")
    maya.standalone.uninitialize()
    sys.exit(1)

# --- Optionally unload plugins that cause UI/OpenGL dependencies ---
for plugin in ['mldeformer']:
    if cmds.pluginInfo(plugin, query=True, loaded=True):
        try:
            cmds.unloadPlugin(plugin)
            sys.stdout.write("Unloaded plugin: {}\n".format(plugin))
        except Exception as e:
            sys.stderr.write("Failed to unload plugin {}: {}\n".format(plugin, e))

# --- Load the FBX plugin ---
if not cmds.pluginInfo('fbxmaya', query=True, loaded=True):
    try:
        cmds.loadPlugin('fbxmaya')
    except Exception as e:
        exit_with_error("Error loading fbxmaya plugin: {}".format(e))

# --- Set common FBX export options ---
try:
    mel.eval('FBXExportSmoothingGroups -v true')
    mel.eval('FBXExportSmoothMesh -v true')
    mel.eval('FBXExportTriangulate -v false')
    mel.eval('FBXExportHardEdges -v false')
    mel.eval('FBXExportConvertUnitString cm')
    mel.eval('FBXExportUpAxis y')
except Exception as e:
    exit_with_error("Error setting FBX export options: {}".format(e))

# --- Open the Maya file ---
try:
    cmds.file(maya_file, open=True, force=True)
except Exception as e:
    exit_with_error("Failed to open {}: {}".format(maya_file, e))

# --- Get timeline range and set baking options ---
try:
    start_time = cmds.playbackOptions(q=True, minTime=True)
    end_time = cmds.playbackOptions(q=True, maxTime=True)

    mel.eval('FBXExportBakeComplexAnimation -v true')
    mel.eval('FBXExportBakeComplexStart -v {}'.format(start_time))
    mel.eval('FBXExportBakeComplexEnd -v {}'.format(end_time))
except Exception as e:
    exit_with_error("Error setting baking options for {}: {}".format(maya_file, e))

# --- Select only transform and joint nodes (avoid cameras, lights, etc.) ---
try:
    cmds.select(cmds.ls(type=('transform', 'joint')), replace=True)
except Exception as e:
    exit_with_error("Error selecting objects for export: {}".format(e))

# --- Define output FBX file path ---
base_name = os.path.splitext(os.path.basename(maya_file))[0]
fbx_file = os.path.normpath(os.path.join(output_folder, base_name + ".fbx"))

# --- Attempt FBX export using MEL and native command as fallback ---
export_success = False
try:
    mel.eval('FBXExport -f "{}" -s'.format(fbx_file))
    export_success = True
    sys.stdout.write("MEL FBX export succeeded for: {}\n".format(fbx_file))
except Exception as e:
    sys.stderr.write("MEL export command failed for {}: {}\n".format(fbx_file, e))
    try:
        cmds.file(fbx_file, force=True, exportAll=True, type="FBX export")
        export_success = True
        sys.stdout.write("Native FBX export succeeded for: {}\n".format(fbx_file))
    except Exception as e:
        sys.stderr.write("Native export command failed for {}: {}\n".format(fbx_file, e))
        export_success = False

# --- Verify export result ---
if export_success and os.path.exists(fbx_file):
    sys.stdout.write("FBX file successfully created: {}\n".format(fbx_file))
else:
    sys.stderr.write("FBX file not found after export: {}\n".format(fbx_file))

maya.standalone.uninitialize()
