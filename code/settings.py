"""

GLOBAL SETTINGS FOR THE PROJECT

"""

# All of this will just be imported with *

outputfolder = "../output/"


def filename2d(experiment):
    return f"{outputfolder}/2d_{experiment['name']}.h5"



def filename3d(experiment):
    return f"{outputfolder}/3d_{experiment['name']}.h5"


def experiment(**kwargs):
    ex = {}
    ex["name"] = "control"
    ex["n"] = 3
    ex["A"] = 1e-25
    ex["gmag"] = 9.81  # grav accel
    ex["alpha"] = 0.003  # angle of inclined plane  (600m/200km ~3permil)
    ex["rho"] = 917
    ex["domain_size"] = (400e3, 50e3, 1500)
    ex["resolution"] = (30, 20, 8)
    ex["model_half"] = True  # exploit y-symmetry.
    ex["icestream_width"] = 20e3
    ex["shearmargin_enhancement"] = 0
    ex["shearmargin_enhancement_pos"] = 0
    ex.update(kwargs)
    return ex
