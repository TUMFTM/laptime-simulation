# Introduction
This repository contains a quasi-steady-state lap time simulation implemented in Python. It can be used to evaluate the
effect of various vehicle parameters on lap time and energy consumption. To generate a proper raceline for a given race
track, it furthermore contains our minimum curvature optimization.

Contact person: [Alexander Heilmeier](mailto:alexander.heilmeier@tum.de).

# List of components
* `laptimesim`: This python module is used to simulate the lap time of a specified race car on a given race track as
accurate as possible. It can furthermore be used to evaluate the effects of parameter changes, e.g. the lap time
sensitivity against mass. The `input` folder contains the racelines and track parameters as well as the vehicle
parameters. Please see the paper linked below for further information.
* `opt_raceline`: This python module is used to determine a proper raceline for a given race track. The used approach is
based on a minimization of the summed curvature. It is extracted from our main repository 
https://github.com/TUMFTM/global_racetrajectory_optimization. Please see the paper linked below for further information.

# Dependencies
Use the provided `requirements.txt` in the root directory of this repo, in order to install all required modules.\
`pip3 install -r /path/to/requirements.txt`

The code is tested with Python 3.8.3 on Windows 10 and 3.6.8 on Ubuntu 18.04.

### Solutions for possible installation problems (Windows)
`cvxpy`, `cython` or any other package requires a `Visual C++ compiler` -> Download the build tools for Visual Studio
2019 (https://visualstudio.microsoft.com/de/downloads/ -> tools for Visual Studio 2019 -> build tools), install them and
chose the `C++ build tools` option to install the required C++ compiler and its dependencies

### Solutions for possible installation problems (Ubuntu)
1. `matplotlib` requires `tkinter` -> can be solved by `sudo apt install python3-tk`
2. `Python.h` required `quadprog` -> can be solved by `sudo apt install python3-dev`

# Intended workflow
The intended workflow is as follows:
* `opt_raceline`: Calculate a proper raceline for the race track.
* `laptimesim`: Use the determined raceline in the lap time simulation to calculate the velocity profile, lap time,
energy consumption and so on. Sensitivity analysis can be performed to determine further parameters, e.g. the lap time
mass sensitivity.

# Running the raceline optimization
If the requirements are installed on the system, follow these steps:

* `Step 1`: (optional) The race track can be supplied in two formats: `.csv` and `.geojson`. The former includes not
only the centerline but also the track widths `[x, y, w_tr_right, w_tr_left]`. The latter contains only the centerline.
Add your own files to the according folder, either `/opt_raceline/input/centerlines/geojson` or 
`/opt_raceline/input/tracks/csv`. A `.geojson` file can be extracted from map services such as OpenStreetMap, for
example (see separate instructions below). Additionally to this step, a track map should be copied to
`/opt_raceline/input/maps` to be able to check the track data during the import. Such a track map can be obtained from the
FIA, e.g. on https://www.fia.com/events/fia-formula-one-world-championship/season-2017/eventtiming-information
* `Step 2`: Check the user input section in the upper part of `main_opt_raceline.py`. It might be necessary to test a
little bit to find a working parameter set for the individual race track.
* `Step 3`: Execute `main_opt_raceline.py` to start the raceline optimization process. During the import of the track data
file you will see a plot of the track on its corresponding track map (if it was provided in the first step). In case of
a GeoJSON file you must select which of the sections should be used. By clicking on the legend entries you can activate
or deactivate the corresponding lines in the plot to obtain a closed but unique 
centerline. Additionally, you can enter the ID of the section containing the start finish line into the text field. As
soon as you close the plot the final status will be used for the further processing steps. If there is only one line for
the whole race track, this step seems unncessary (e.g. Budapest). However, many exported GeoJSON data files will contain
a lot of different lines (e.g. Shanghai).
* `Step 4`: If the optimization was finished successfully, you will see a plot of the optimized raceline as well as
its curvature profile (if using the standard plotting options). For a later usage in the lap time simulation it is 
of great importance that this curvature profile is smooth because this heavily influences simulation result. Enter the 
presented length of the raceline into the `track_pars.ini` file within the `laptimesim` folder and copy
the exported raceline from the output folder to the according input folder of the lap time simulation. Furthermore, the
smoothed centerline was saved in the output folder and can be used if required, e.g. for plotting purposes.

![Resulting raceline for the Berlin FE track](opt_raceline/opt_raceline_berlin.png)

### Acknowledgement for the available race tracks
The currently available tracks in the input folder were created by Andressa de Paula Suiti during her semester thesis.

### Detailed description of the curvature minimization used during the raceline optimization
Please refer to our paper for further information:\
Heilmeier, Wischnewski, Hermansdorfer, Betz, Lienkamp, Lohmann\
Minimum Curvature Trajectory Planning and Control for an Autonomous Racecar\
DOI: 10.1080/00423114.2019.1631455

### Extracting the centerline of a race track from OpenStreetMap
* `Step 1`: Open https://overpass-turbo.eu/ This is a tool to extract map informations from OpenStreetMap.
* `Step 2`: Navigate to the desired race track, e.g. the Red Bull Ring in Austria.
* `Step 3`: Paste the following search into the text field and execute the search to highlight everything which is tagged
as a raceway.
<!-- language: lang-none -->
    [out:json][timeout:25];
    (
      node["highway"="raceway"]({{bbox}});
      way["highway"="raceway"]({{bbox}});
      relation["highway"="raceway"]({{bbox}});
    );
    out body;
    >;
    out skel qt;
* `Step 4`: Click export and save it as a GeoJSON. Be aware that the export might include a lot of unnecessary points
which must be excluded either in a separate step or during the import.

# Running the lap time simulation
If the requirements are installed on the system, follow these steps:

* `Step 1`: (optional) Adjust a given or create a new vehicle parameter file (.ini) for the simulation. The files are 
located in `/laptimesim/input/vehicles`.
 * `Step 2`: (optional) Adjust a given or create a new track. Every track consists of some parameters (e.g. length)
  as well as a raceline. Therefore, you have to make sure the parameters are given in 
  `/laptimesim/input/tracks/track_pars.ini` and the raceline is available in `/laptimesim/input/tracks/racelines`.
  Additionally, you can place a .png track map in `/laptimesim/input/tracks/maps`.
* `Step 3`: Check the user input section in our main file `main_laptimesim.py`.
* `Step 4`: Run `main_laptimesim.py`.

![Lap time simulation result for the Monza racetrack](laptimesim/laptimesim_monza.png)

### Detailed description of the lap time simulation
Please refer to our paper for further information:
```
@inproceedings{Heilmeier2019,
doi = {10.1109/ever.2019.8813646},
url = {https://doi.org/10.1109/ever.2019.8813646},
year = {2019},
month = may,
publisher = {{IEEE}},
author = {Alexander Heilmeier and Maximilian Geisslinger and Johannes Betz},
title = {A Quasi-Steady-State Lap Time Simulation for Electrified Race Cars},
booktitle = {2019 Fourteenth International Conference on Ecological Vehicles and Renewable Energies ({EVER})}}
```

# Related open-source repositories
* Lap-discrete race simulation: https://github.com/TUMFTM/race-simulation
* Time-discrete race simulator: https://github.com/heilmeiera/time-discrete-race-simulator
* Race track database: https://github.com/TUMFTM/racetrack-database
* Formula 1 timing database: https://github.com/TUMFTM/f1-timing-database
