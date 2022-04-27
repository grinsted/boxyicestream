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
    ex["domain_w"] = 150e3
    ex["domain_h"] = 2000
    ex["resolution_l"] = 50
    ex["resolution_w"] = 50
    ex["resolution_h"] = 15
    ex["model_half"] = True  # exploit y-symmetry.
    ex["icestream_width"] = 50e3
    ex["shearmargin_enhancement"] = 4
    ex["shearmargin_enhancement_pos"] = 24000
    ex["shearmargin_enhancement_sigma"] = 2e3
    ex["weertman_beta2"] = 100e3 / (100 / (365 * 24 * 3600))
    ex["icestream_Exx"] = 1  # only used in 3d experiment
    ex["anisotropic"] = False  # only used in 3d experiment
    ex.update(kwargs)
    return ex


def print_experiment_highlights(exp1):
    set_control = set(experiment().items())
    set_exp1 = set(exp1.items())
    pprint.pprint(dict(set_exp1 - set_control))


experiments = []
experiments += [experiment()]
experiments += [experiment(name="wider_slip", icestream_width=50100)]
experiments += [experiment(name="softer_margin", shearmargin_enhancement=4.2)]
#experiments += [experiment(name="soft_margin_pos", shearmargin_enhancement_pos=25100)]
#experiments += [experiment(name="wider_soft_margin", shearmargin_enhancement_sigma=2.1e3)]
experiments += [experiment(name="thicker", domain_h=2010)]
