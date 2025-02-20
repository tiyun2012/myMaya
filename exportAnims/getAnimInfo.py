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


try:
    cmds.file(maya_file, open=True, force=True)
except Exception as e:
    exit_with_error("Failed to open {}: {}".format(maya_file, e))

# --- Get timeline range and set baking options ---
try:
    start_time = cmds.playbackOptions(q=True, minTime=True)
    end_time = cmds.playbackOptions(q=True, maxTime=True)


except Exception as e:
    exit_with_error("Error setting baking options for {}: {}".format(maya_file, e))


print(["startFrame=",start_time,"endFrame=",end_time,"frames=",end_time-start_time])
maya.standalone.uninitialize()
