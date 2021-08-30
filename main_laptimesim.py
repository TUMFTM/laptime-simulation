import laptimesim
import time
import datetime
import os
import numpy as np
import pkg_resources
import toml
import argparse

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
         race_characteristics: dict,
         car_properties: dict,
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

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE TRACK INSTANCE --------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines",
                                    track_opts["trackname"] + ".csv")

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

        # recalculate raceline based on curvature
        track.check_track()

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE CAR INSTANCE ----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    car = laptimesim.src.car_electric.CarElectric(pars=veh_pars)

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

            # plot engine speed and gear selection
            lap.plot_enginespeed_gears()

    else:

        # output file 
        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        resultsfile = os.path.join(repo_path, "laptimesim", "output", "results-{}.csv".format(date))
        datastore = DataStore(results_file_name=resultsfile)

        # sensitivity analysis -----------------------------------------------------------------------------------------

        # turn debug messages off
        lap.pars_solver["print_debug"] = False

        datastore.parse_car_properties(car_properties)

        datastore.generate_unique_sa_combinations()

        for i, single_simulation_data in datastore.single_iteration_data.items():
            print("SA: Starting solver run (%i)" % (i + 1))

            current_lap_car_variables = single_simulation_data.race_car_model.get_vehicle_properties()

            # change mass of vehicle
            lap.driverobj.carobj.pars_general["m"] = current_lap_car_variables[TOTAL_VEHICLE_MASS_TAG]
            lap.driverobj.carobj.pars_general["c_w_a"] = current_lap_car_variables[C_W_A_TAG]
            lap.driverobj.carobj.pars_engine["torque_e_motor_max"] = current_lap_car_variables[MOTOR_MAX_TORQUE_TAG]
            lap.driverobj.carobj.pars_general["f_roll"] = current_lap_car_variables[ROLLING_RESISTANCE_TAG]
            lap.driverobj.carobj.pars_engine["pow_e_motor_max"] = current_lap_car_variables[MOTOR_MAX_TORQUE_TAG]

            # simulate lap and save lap time
            lap.simulate_lap()

            race_sim = RaceSim(pit_time=current_lap_car_variables[PIT_TIME_TAG],
                                gwc_times=race_characteristics["gwc_times"],
                                lap_time=lap.t_cl[-1],
                                energy_per_lap=lap.e_cons_cl[-1],
                                battery_capacity=current_lap_car_variables[BATTERY_SIZE_TAG])
            race_sim.calculate()

            total_pits = 0
            energy_remaining = 0
            for day in race_sim.race_days:
                total_pits += day.number_of_pits
                energy_remaining += day.energy_remaining
            
            datastore.set_single_iteration_results(iteration=i,
                                                   lap_time=lap.t_cl[-1],
                                                   total_laps=race_sim.total_laps,
                                                   lap_energy=lap.e_cons_cl[-1]/1000, # 1000 factor fo J -> kJ
                                                   total_pits=total_pits,
                                                   gwc_times=race_characteristics["gwc_times"],
                                                   energy_remaining=energy_remaining) 

            lap.reset_lap()

            print("SA: Finished solver run (%i)" % (i + 1))
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
    race_characteristics_ = config["race_characteristics_"]

    # see the car_config for variable details
    car_properties_ = car_config["car_properties_"]
    veh_pars_ = car_config["veh_pars_"]

    # Remap characteristics to the tag constants that are used throughout
    # the simulation
    car_properties = {}
    car_properties[BATTERY_SIZE_TAG] = car_properties_["independent_variables"]["battery_size"]
    car_properties[MOTOR_MAX_TORQUE_TAG] = car_properties_["independent_variables"]["motor_max_torque"]
    car_properties[GROSS_VEHICLE_WEIGHT_TAG] = car_properties_["independent_variables"]["gross_vehicle_weight"]
    car_properties[WEIGHT_REDUCTION_TAG] = car_properties_["independent_variables"]["weight_reduction"]
    car_properties[COEFFICIENT_OF_DRAG_TAG] = car_properties_["independent_variables"]["coefficient_of_drag"]

    car_properties[BATTERY_ENERGY_DENSITY_TAG] = car_properties_["relationship_variables"]["battery_energy_density"]
    car_properties[BATTERY_MASS_PIT_FACTOR_TAG] = car_properties_["relationship_variables"]["battery_mass_pit_factor"]
    car_properties[BATTERY_POWER_OUTPUT_FACT0R_TAG] = car_properties_["relationship_variables"]["battery_power_output_factor"]
    car_properties[MOTOR_CONSTANT_TAG] = car_properties_["relationship_variables"]["motor_constant"]
    car_properties[MOTOR_TORQUE_DENSITY_TAG] = car_properties_["relationship_variables"]["motor_torque_density"]
    car_properties[MAX_VEHICLE_WEIGHT_RATIO_TAG] = car_properties_["relationship_variables"]["max_vehicle_weight_ratio"]
    car_properties[CAR_DENSITY_TAG] = car_properties_["relationship_variables"]["car_density"]
    car_properties[CHASSIS_BATTERY_MASS_FACTOR_TAG] = car_properties_["relationship_variables"]["chassis_battery_mass_factor"]
    car_properties[CHASSIS_MOTOR_MASS_FACTOR_TAG] = car_properties_["relationship_variables"]["chassis_motor_mass_factor"]
    car_properties[ROLLING_RESISTANCE_MASS_FACTOR_TAG] = car_properties_["relationship_variables"]["rolling_resistance_mass_factor"]

    # get track parameters
    track_pars_ = track_config[track_opts_["trackname"]]
    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(track_opts=track_opts_,
         solver_opts=solver_opts_,
         driver_opts=driver_opts_,
         sa_opts=sa_opts_,
         debug_opts=debug_opts_,
         race_characteristics=race_characteristics_,
         car_properties=car_properties,
         veh_pars=veh_pars_,
         track_pars=track_pars_)
