# test.py: Verwende es später für das Laden des Modells und das Testen mit neuen Bildern.


"""
from roboflow import Roboflow
rf = Roboflow(api_key="i69F4gJn4h97e9d0s4XC")
project = rf.workspace("bildererkennung-im-bild").project("ui-control-c")
version = project.version(1)
dataset = version.download("yolov8")
"""

# fremd model mit 4 tausend bildern!
from roboflow import Roboflow
rf = Roboflow(api_key="i69F4gJn4h97e9d0s4XC")
project = rf.workspace("cingoz8").project("cingoz8")
version = project.version(2)
dataset = version.download("yolov8")
                