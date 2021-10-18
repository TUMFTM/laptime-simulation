import laptimesim
import time
import datetime
import os
import numpy as np
import pkg_resources
import toml

from definitions import *  # FIXME enumerate imports
from datastore import (DataStore)

from race_sim import RaceSim

"""
author:
Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

date:
23.12.2018

.. description::
The file contains the script to run the lap time simulation starting with the import of various parameters and ending
with the visualization of the calculated data.

.. hints:
Input tracks must be unclosed, i.e. last point != first point!
"""


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def main(track_opts: dict,
         solver_opts: dict,
         driver_opts: dict,
         sa_opts: dict,
         debug_opts: dict,
         car_config: dict,
         veh_pars: dict,
         track_pars: dict,) -> laptimesim.src.lap.Lap:

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK PYTHON DEPENDENCIES ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get repo path
    repo_path = os.path.dirname(os.path.abspath(__file__))

    # read dependencies from requirements.txt
    requirements_path = os.path.join(repo_path, 'requirements.txt')
    dependencies = []

    with open(requirements_path, 'r') as fh:
        line = fh.readline()

        while line:
            dependencies.append(line.rstrip())
            line = fh.readline()

    # check dependencies
    pkg_resources.require(dependencies)

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION ---------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    output_path = os.path.join(repo_path, "laptimesim", "output")

    output_path_velprofile = os.path.join(output_path, "velprofile")
    os.makedirs(output_path_velprofile, exist_ok=True)

    output_path_testobjects = os.path.join(output_path, "testobjects")
    os.makedirs(output_path_testobjects, exist_ok=True)

    if debug_opts["use_plot_comparison_tph"]:
        output_path_veh_dyn_info = os.path.join(output_path, "veh_dyn_info")
        os.makedirs(output_path_veh_dyn_info, exist_ok=True)
    

    date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
    resultsfile = os.path.join(repo_path, "laptimesim", "output", "results-{}.csv".format(date))
    datastore = DataStore(results_file_name=resultsfile,
                            track_pars=track_pars,
                            car_name=veh_pars[VEHICLE_TAG])

    datastore.parse_car_config(car_config)

    datastore.generate_unique_sa_combinations()

    first_iter_veh_pars = datastore.single_iteration_data[0].race_car_model.get_car_parameters_for_laptimesim()

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE TRACK INSTANCE --------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    tmp_track_file_path = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines",
                                    track_opts["trackname"])
    trackfilepath = os.path.join(tmp_track_file_path + ".csv")

    vel_lim_glob = np.inf

    # create instance
    track = laptimesim.src.track.Track(track_opts=track_opts,
                                       track_pars=track_pars,
                                       trackfilepath=trackfilepath,
                                       vel_lim_glob=vel_lim_glob,
                                       yellow_s1=driver_opts["yellow_s1"],
                                       yellow_s2=driver_opts["yellow_s2"],
                                       yellow_s3=driver_opts["yellow_s3"])

    # debug plot
    if debug_opts["use_debug_plots"]:
        # check if track map exists and set path accordingly
        mapfolderpath = os.path.join(repo_path, "laptimesim", "input", "tracks", "maps")
        mapfilepath = ""

        for mapfile in os.listdir(mapfolderpath):
            if track_opts["trackname"] in mapfile:
                mapfilepath = os.path.join(mapfolderpath, mapfile)
                break

        # plot trackmap
        track.plot_trackmap(mapfilepath=mapfilepath)

        # plot curvature
        track.plot_curvature()

        if track_opts["use_elevation"]:
            track.plot_elevation()
            track.plot_elevation_3d()

        # recalculate raceline based on curvature
        track.check_track()

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE CAR INSTANCE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    car = laptimesim.src.car_electric.CarElectric(pars=first_iter_veh_pars)

    # debug plot
    if debug_opts["use_debug_plots"]:
        # plot tire force potential characteristics
        car.plot_tire_characteristics()

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE DRIVER INSTANCE -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create instance
    driver = laptimesim.src.driver.Driver(carobj=car,
                                          pars_driver=driver_opts,
                                          trackobj=track,
                                          stepsize=track.stepsize)

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE LAP INSTANCE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    lap = laptimesim.src.lap.Lap(driverobj=driver,
                                 trackobj=track,
                                 pars_solver=solver_opts,
                                 debug_opts=debug_opts)
    # ------------------------------------------------------------------------------------------------------------------
    # CALL SOLVER ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # save start time
    t_start = time.perf_counter()

    # call simulation
    if not sa_opts["use_sa"]:
        # normal simulation --------------------------------------------------------------------------------------------
        lap.simulate_lap()

        # debug plot
        if debug_opts["use_debug_plots"]:
            # plot torques
            lap.plot_torques()

            # plot lateral acceleration profile
            lap.plot_lat_acc()

            # plot tire load profile
            lap.plot_tire_loads()

            # plot aero forces
            lap.plot_aero_forces()

            #plot power
            lap.plot_power()
            lap.plot_throttle()
            # plot engine speed and gear selection
            lap.plot_enginespeed_gears()
    if not sa_opts["use_sa"]:
        if debug_opts["use_plot"]:
            lap.plot_overview()
            lap.plot_revs_gears()

    else:

        # sensitivity analysis -----------------------------------------------------------------------------------------

        for i, single_simulation_data in datastore.single_iteration_data.items():
            print("SA: Starting solver run (%i)" % (i + 1))

            race_car_object = single_simulation_data.race_car_model

            veh_pars = race_car_object.get_car_parameters_for_laptimesim()
            car = laptimesim.src.car_electric.CarElectric(pars=veh_pars)

            # change properties of vehicle in the lap simulation
            lap.driverobj.carobj = car


            # simulate lap and save lap time
            lap.simulate_lap()

            total_pit_time = race_car_object.general_parameters.pit_time + track_pars[PIT_DRIVE_THROUGH_PENALTY_TIME]

            race_sim = RaceSim(pit_time=total_pit_time,
                                gwc_times=datastore.track_pars[GWC_TIMES_TAG],
                                lap_time=lap.t_cl[-1],
                                energy_per_lap=lap.e_cons_cl[-1],
                                battery_capacity=race_car_object.battery_parameters.size)
            race_sim.calculate()

            total_pits = 0
            energy_remaining = 0
            for day in race_sim.race_days:
                total_pits += day.number_of_pits
                energy_remaining += day.energy_remaining*100  # for percentage conversion

            is_winning_car_configuration = race_sim.total_laps > track_pars["winning_laps"]

            datastore.set_single_iteration_results(iteration=i,
                                                   lap_time=lap.t_cl[-1],
                                                   total_laps=race_sim.total_laps,
                                                   lap_energy=lap.e_cons_cl[-1]/1000, # 1000 factor fo J -> kJ
                                                   total_pits=total_pits,
                                                   energy_remaining=energy_remaining,
                                                   is_winning_car_configuration=is_winning_car_configuration) 

            lap.reset_lap()

            print("Solver run {}. Winning car?: {}, total laps: {}".format(i,
                                                                          is_winning_car_configuration,
                                                                          race_sim.total_laps ))
        best_results, multiple_optimum_results = datastore.get_best_result()

        print("Best result was iteration: {}".format(best_results[ITER_TAG]))
        print("{}: {}".format(WINNING_ELECTRIC_CAR_TAG, best_results[WINNING_ELECTRIC_CAR_TAG]))
        print("{}: {}".format(TOTAL_LAPS_TAG, best_results[TOTAL_LAPS_TAG]))
        print("{}: {}".format(WINNING_GAS_CAR_LAPS, track_pars[WINNING_GAS_CAR_LAPS]))
        print("{}: {}".format(TOTAL_PITS_TAG, best_results[TOTAL_PITS_TAG]))
        print("{}: {}".format(LAP_ENERGY_TAG, best_results[LAP_ENERGY_TAG]))
        print("{}: {}".format(ENERGY_REMAINING_TAG, best_results[ENERGY_REMAINING_TAG]))
        print("{}: {}".format(BATTERY_SIZE_TAG, best_results[BATTERY_SIZE_TAG]))
        print("Are there multiple optimum results?: {}".format(multiple_optimum_results))
        print("total simulation time: {}"
            .format(time.perf_counter() - t_start))

# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # Importing config from sim_config.toml

    # get repo path
    repo_path = os.path.dirname(os.path.abspath(__file__))

    config = toml.load(os.path.join(repo_path, "sim_config.toml"))
    car_name = config["car_opts_"]["car"]
    car_name = "{}.toml".format(car_name)
    car_config = toml.load(os.path.join(repo_path, "laptimesim", "input", "vehicles", car_name))
    track_config = toml.load(os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.toml"))
 
    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # See sim_config.toml for variable descriptions
    track_opts_ = config["track_opts_"]
    solver_opts_ = config["solver_opts_"]
    driver_opts_ = config["driver_opts_"]
    sa_opts_ = config["sa_opts_"]
    debug_opts_ = config["debug_opts_"]

    # get track parameters
    track_pars_ = track_config[track_opts_["trackname"]]
    track_pars_[WINNING_GAS_CAR_LAPS] = track_pars_["winning_laps"]
    track_pars_[PIT_DRIVE_THROUGH_PENALTY_TIME] = track_pars_["pit_penalty"]
    track_pars_[GWC_TIMES_TAG] = track_pars_["gwc_times"]

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(track_opts=track_opts_,
         solver_opts=solver_opts_,
         driver_opts=driver_opts_,
         sa_opts=sa_opts_,
         debug_opts=debug_opts_,
         car_config=car_config,
         track_pars=track_pars_)
