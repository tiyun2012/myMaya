#!/usr/bin/env mayapy

"""
Usage: mayapy getAnimInfo.py <maya_file> <output_folder>

Prints the animation timeline info for the specified Maya file.
For .mb files, the file is opened in Maya and playbackOptions are queried.
For .ma files, the script reads the ASCII file and extracts timeline info 
from a playbackOptions command.
Errors during reading are logged under the output_folder.
"""

import os
import sys
import re

if len(sys.argv) < 3:
    sys.stderr.write("Usage: getAnimInfo.py <maya_file> <output_folder>\n")
    sys.exit(1)

maya_file = sys.argv[1]
output_folder = sys.argv[2]

def log_error(file_name, error_message):
    log_file = os.path.join(output_folder, "getAnimInfo_error.log")
    with open(log_file, 'a') as lf:
        lf.write("File: {} -- {}\n".format(file_name, error_message))

extension = os.path.splitext(maya_file)[1].lower()

if extension == '.mb':
    # Process .mb file using Maya Standalone
    os.environ['MAYA_DISABLE_CLICKTONE'] = '1'
    import maya.standalone
    maya.standalone.initialize(name='python')
    import maya.cmds as cmds

    try:
        cmds.file(maya_file, open=True, force=True)
    except Exception as e:
        error_msg = "Failed to open {}: {}".format(maya_file, e)
        sys.stderr.write(error_msg + "\n")
        log_error(maya_file, error_msg)
        maya.standalone.uninitialize()
        sys.exit(1)

    try:
        start_time = cmds.playbackOptions(q=True, minTime=True)
        end_time = cmds.playbackOptions(q=True, maxTime=True)
    except Exception as e:
        error_msg = "Error querying timeline options: {}".format(e)
        sys.stderr.write(error_msg + "\n")
        log_error(maya_file, error_msg)
        maya.standalone.uninitialize()
        sys.exit(1)

    print(["startFrame=", start_time, "endFrame=", end_time, "frames=", end_time - start_time])
    maya.standalone.uninitialize()

elif extension == '.ma':
    # Process .ma file by reading it as text and using regex to extract timeline info
    try:
        with open(maya_file, 'r') as f:
            content = f.read()
    except Exception as e:
        error_msg = "Error reading file {}: {}".format(maya_file, e)
        sys.stderr.write(error_msg + "\n")
        log_error(maya_file, error_msg)
        sys.exit(1)

    # Regex pattern to match the playbackOptions command.
    # Expected pattern example:
    #   playbackOptions -min 0 -max 120 -ast 0 -aet 120
    pattern = re.compile(
        r'playbackOptions\s+-min\s+([-+]?[0-9]*\.?[0-9]+)\s+-max\s+([-+]?[0-9]*\.?[0-9]+)\s+-ast\s+([-+]?[0-9]*\.?[0-9]+)\s+-aet\s+([-+]?[0-9]*\.?[0-9]+)'
    )
    match = pattern.search(content)
    if match:
        start_time = float(match.group(1))
        end_time = float(match.group(2))
        print(["startFrame=", start_time, "endFrame=", end_time, "frames=", end_time - start_time])
    else:
        error_msg = "No valid playbackOptions command found in {} file.".format(maya_file)
        sys.stderr.write(error_msg + "\n")
        log_error(maya_file, error_msg)
        sys.exit(1)
else:
    error_msg = "Unsupported file extension: {}".format(extension)
    sys.stderr.write(error_msg + "\n")
    log_error(maya_file, error_msg)
    sys.exit(1)
