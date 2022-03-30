"""

GLOBAL SETTINGS FOR THE PROJECT

"""

# All of this will just be imported with *

outputfolder = "../output/" 


control_experiment = {}
control_experiment["name"] = "control"
control_experiment["n"] = 3
control_experiment["A"] = 1e-25
control_experiment["gmag"] = 9.81  # grav accel
control_experiment["alpha"] = 0.003  # angle of inclined plane  (600m/200km ~3permil)
control_experiment["rho"] = 917
control_experiment["domain_size"] = (400e3, 50e3, 1500)
control_experiment["resolution"] = (30, 20, 8)
control_experiment["model_half"] = True  # exploit y-symmetry.
control_experiment["icestream_width"] = 20e3
control_experiment["shearmargin_enhancement"] = 0
