import json


mesh_to_name = {
"Mesh": "noseSneerRight", 
"Mesh1": "noseSneerLeft",
"Mesh2": "mouthUpperUpLeft",
"Mesh3": "mouthUpperUpRight",
"Mesh4": "mouthStretchRight",
"Mesh5": "mouthSmileLeft",
"Mesh6": "mouthStretchLeft",
"Mesh7": "mouthSmileRight",
"Mesh8": "mouthStretchRight",
"Mesh9": "mouthShrugLower",
"Mesh10": "mouthSmileRight",
"Mesh11": "mouthRollLower",
"Mesh12": "mouthRight",
"Mesh13": "mouthPucker",
"Mesh14": "mouthPressRight",
"Mesh15": "mouthPressLeft",
"Mesh16": "mouthLowerDownRight",
"Mesh17": "mouthLowerDownLeft",
"Mesh18": "mouthLeft",
"Mesh19": "mouthFunnel",
"Mesh20": "mouthFrownRight",
"Mesh21": "mouthFrownLeft",
"Mesh22": "mouthDimpleRight",
"Mesh23": "mouthDimpleLeft",
"Mesh24": "mouthClose",
"Mesh25": "jawRight",
"Mesh26": "jawOpen",
"Mesh27": "jawLeft",
"Mesh28": "jawForward",
"Mesh29": "eyeWideRight",
"Mesh30": "eyeWideLeft",
"Mesh31": "eyeSquintRight",
"Mesh32": "eyeSquintLeft",
"Mesh33": "eyeLookUpRight",
"Mesh34": "eyeLookUpLeft",
"Mesh35": "eyeLookOutRight",
"Mesh36": "mouthRollLower",
"Mesh37": "eyeLookOutLeft",
"Mesh38": "eyeLookInLeft",
"Mesh39": "eyeLookDownRight",
"Mesh40": "eyeLookDownLeft", 
"Mesh41": "eyeBlinkRight",
"Mesh42": "eyeBlinkLeft",
"Mesh43": "cheekSquintRight",
"Mesh44": "cheekSquintLeft",
"Mesh45": "cheekPuff",
"Mesh46": "browOuterUpRight",
"Mesh47": "browOuterUpLeft",
"Mesh48": "browInnerUp",
"Mesh49": "browDownRight",
"Mesh50": "browDownLeft"
}
new_name_to_mesh = {}
new_mesh_to_name = {}
for key in mesh_to_name:
    new_mesh_to_name[key] = mesh_to_name[key].lower()
    new_name_to_mesh[mesh_to_name[key].lower()] = key
with open('C:/Users/evan1/Documents/staggered_face/models/joonho_mesh_number_to_AU_name.json', 'w') as fp:
    json.dump(new_mesh_to_name, fp)
with open('C:/Users/evan1/Documents/staggered_face/models/joonho_AU_name_to_mesh_number.json', 'w') as fp:
    json.dump(new_name_to_mesh, fp)