# RUDRA in Nuke's Nodes toolbar, with the FXTD RUDRA mark (26 Nov 2026).
#
# The OFX plug-in puts "RUDRA SDR to HDR" in a RUDRA menu by itself; this file only gives that menu and the
# node the RUDRA icon. tmp\BUILD_OFX.ps1 -Install copies this folder to ~/.nuke/RUDRA and adds it to
# ~/.nuke/init.py; by hand: copy the folder there and add  nuke.pluginAddPath("./RUDRA")  to init.py.
import os

import nuke

_HERE = os.path.dirname(os.path.abspath(__file__))
_ICON = os.path.join(_HERE, "RUDRA.png").replace("\\", "/")
_NODE = "OFXcom.fxtdstudios.rudra.SdrToHdr_v1"   # Nuke's class for an OFX effect: OFX<identifier>_v<major>

try:
    menu = nuke.menu("Nodes").addMenu("RUDRA", icon=_ICON)
    menu.addCommand("RUDRA SDR to HDR", "nuke.createNode('%s')" % _NODE, icon=_ICON)
except Exception as e:  # never stop Nuke's start for an icon
    nuke.tprint("RUDRA menu: %s" % e)
