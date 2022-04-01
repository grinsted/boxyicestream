"""

GLOBAL SETTINGS FOR THE PROJECT

"""

# All of this will just be imported with *

import pprint

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
    ex["domain_l"] = 400e3
    ex["domain_w"] = 100e3
    ex["domain_h"] = 1000
    ex["resolution_l"] = 30
    ex["resolution_w"] = 40
    ex["resolution_h"] = 8
    ex["model_half"] = True  # exploit y-symmetry.
    ex["icestream_width"] = 20e3
    ex["shearmargin_enhancement"] = 0
    ex["shearmargin_enhancement_pos"] = 0
    ex["weertman_beta"] = 100e3 / (100 / (365 * 24 * 3600))
    ex.update(kwargs)
    return ex


def print_experiment_highlights(exp1):
    set_control = set(experiment().items())
    set_exp1 = set(exp1.items())
    pprint.pprint(dict(set_exp1 - set_control))
