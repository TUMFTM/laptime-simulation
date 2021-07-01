import laptimesim
import time
import datetime
import os
import numpy as np
import matplotlib.pyplot as plt
import pkg_resources
import pickle
import csv
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import toml

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
         debug_opts: dict) -> laptimesim.src.lap.Lap:

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

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "track_pars.ini")

    if not track_opts["use_pit"]:  # normal case
        trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines",
                                     track_opts["trackname"] + ".csv")

    else:  # case pit
        trackfilepath = os.path.join(repo_path, "laptimesim", "input", "tracks", "racelines",
                                     track_opts["trackname"] + "_pit.csv")

    # set velocity limit
    if driver_opts["vel_lim_glob"] is not None:
        vel_lim_glob = driver_opts["vel_lim_glob"]
    elif solver_opts["series"] == "FE":
        vel_lim_glob = 225.0 / 3.6
    else:
        vel_lim_glob = np.inf

    # create instance
    track = laptimesim.src.track.Track(pars_track=track_opts,
                                       parfilepath=parfilepath,
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

    parfilepath = os.path.join(repo_path, "laptimesim", "input", "vehicles", solver_opts["vehicle"])

    # create instance
    if solver_opts["series"] == "F1":
        car = laptimesim.src.car_hybrid.CarHybrid(parfilepath=parfilepath)
    elif solver_opts["series"] == "FE":
        car = laptimesim.src.car_electric.CarElectric(parfilepath=parfilepath)
    else:
        raise IOError("Unknown racing series!")

    # debug plot
    if debug_opts["use_debug_plots"]:
        # plot tire force potential characteristics
        car.plot_tire_characteristics()

        # plot engine power characteristics
        if car.powertrain_type == "combustion":
            car.plot_power_engine()

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
        print("in else")
        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        resultsfile = os.path.join(repo_path, "laptimesim", "output", "results-{}.csv".format(date))

        # sensitivity analysis -----------------------------------------------------------------------------------------

        if debug_opts["use_print"]:
            print("INFO: Performing sensitivity analysis!")

        # turn debug messages off
        lap.pars_solver["print_debug"] = False

        # create parameter ranges
        sa_range_1 = np.linspace(sa_opts["range_1"][0], sa_opts["range_1"][1], sa_opts["range_1"][2])
        if sa_opts["range_2"] is not None:
             sa_range_2 = np.linspace(sa_opts["range_2"][0], sa_opts["range_2"][1], sa_opts["range_2"][2])

        # perform analysis
        if sa_opts["sa_type"] == "mass":
            sa_t_lap = np.zeros(sa_opts["range_1"][2])
            sa_fuel_cons = np.zeros(sa_opts["range_1"][2])

            for i, cur_mass in enumerate(sa_range_1):
                print("SA: Starting solver run (%i)" % (i + 1))

                # change mass of vehicle
                lap.driverobj.carobj.pars_general["m"] = cur_mass

                # simulate lap and save lap time
                lap.simulate_lap()
                sa_t_lap[i] = lap.t_cl[-1]
                sa_fuel_cons[i] = lap.fuel_cons_cl[-1]
                if solver_opts["series"] == "FE":
                    sa_fuel_cons[i] = lap.e_cons_cl[-1] # RMH

                # reset lap
                lap.reset_lap()

                print("SA: Finished solver run (%i)" % (i + 1))

        # perform eLemons analysis
        elif sa_opts["sa_type"] == "elemons_mass":
 
            # initialize this pass variables that collect results
            #len_results = sa_opts["range_1"][2] * sa_opts["range_2"][2]
            len_results = sa_opts["range_1"][2] 
            sa_t_lap = np.zeros(len_results)
            sa_fuel_cons = np.zeros(len_results)
            sa_iter = np.zeros(len_results)
            sa_mass = np.zeros(len_results)
            sa_c_d = np.zeros(len_results)

            for i, cur_mass in enumerate(sa_range_1):
                print("SA: Starting solver run (%i)" % (i + 1))

                # change mass of vehicle
                lap.driverobj.carobj.pars_general["m"] = cur_mass

                # simulate lap and save lap time
                lap.simulate_lap()
                
                sa_fuel_cons[i] = lap.fuel_cons_cl[-1]
                sa_t_lap[i] = lap.t_cl[-1]
                sa_iter[i] = i
                sa_mass[i] = cur_mass
                sa_c_d[i] = lap.driverobj.carobj.pars_general["c_w_a"] 
                sa_t_lap[i] = lap.t_cl[-1]
                sa_fuel_cons[i] = lap.fuel_cons_cl[-1]
                if solver_opts["series"] == "FE":
                    # RMH for formula E (electric) vehicles overide gas fuel and publish electrical energy'
                    sa_fuel_cons[i] = lap.e_cons_cl[-1] 

                # reset lap
                lap.reset_lap()

                print("SA: Finished solver run (%i)" % (i + 1))

        # perform eLemons analysis
        elif sa_opts["sa_type"] == "elemons_mass_cd":

            # initialize this pass variables that collect results
            len_results = sa_opts["range_1"][2] * sa_opts["range_2"][2]
            print("len_results: {}".format(len_results))
            sa_t_lap = np.zeros(len_results)
            sa_fuel_cons = np.zeros(len_results)
            sa_iter = np.zeros(len_results)
            sa_mass = np.zeros(len_results)
            sa_c_d = np.zeros(len_results)

            iter = 0
            for j, cur_cd in enumerate(sa_range_2):
                # change coeff of drag of vehicle
                lap.driverobj.carobj.pars_general["c_w_a"] = cur_cd

                for i, cur_mass in enumerate(sa_range_1):
                    print("\nSA: Starting solver run (%i)" % ((j*sa_opts["range_1"][2]) + (i + 1)))

                    # change mass of vehicle
                    lap.driverobj.carobj.pars_general["m"] = cur_mass

                    # simulate lap and save lap time
                    lap.simulate_lap()

                    sa_fuel_cons[iter] = lap.fuel_cons_cl[-1]
                    sa_t_lap[iter] = lap.t_cl[-1]
                    sa_iter[iter] = iter + 1
                    sa_mass[iter] = cur_mass
                    sa_c_d[iter] = cur_cd
                    if solver_opts["series"] == "FE":
                        # RMH for formula E (electric) vehicles overide gas fuel and publish electrical energy'
                        sa_fuel_cons[iter] = lap.e_cons_cl[-1] 

                    iter += 1

                    lap.reset_lap()
    
    # ------------------------------------------------------------------------------------------------------------------
    # EXPORT -----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # print to command window
    if not sa_opts["use_sa"] and debug_opts["use_print_result"]:
        print("-" * 50)
        print("Forward/Backward Plus Solver")
        print("Solver runtime: %.2f s" % (time.perf_counter() - t_start))
        print("-" * 50)
        print("Lap time: %.3f s" % lap.t_cl[-1])
        print("S1: %.3f s  |  S2: %.3f s  |  S3: %.3f s" %
              (lap.t_cl[track.zone_inds["s12"]],
               lap.t_cl[track.zone_inds["s23"]] - lap.t_cl[track.zone_inds["s12"]],
               lap.t_cl[-1] - lap.t_cl[track.zone_inds["s23"]]))
        print("-" * 50)
        v_tmp = lap.vel_cl[0] * 3.6
        print("Start velocity: %.1f km/h" % v_tmp)
        v_tmp = lap.vel_cl[-1] * 3.6
        print("Final velocity: %.1f km/h" % v_tmp)
        v_tmp = (lap.vel_cl[0] - lap.vel_cl[-1]) * 3.6
        print("Delta: %.1f km/h" % v_tmp)
        print("-" * 50)
        print("Consumption: %.2f kg/lap | %.2f kJ/lap" % (lap.fuel_cons_cl[-1], lap.e_cons_cl[-1] / 1000.0))
        # [J] -> [kJ]
        print("-" * 50)

    elif debug_opts["use_print_result"]:
        print("-" * 50)
        print("Forward/Backward Plus Solver")
        print("Runtime for sensitivity analysis: %.2f s" % (time.perf_counter() - t_start))
        print("-" * 50)

        if sa_opts["sa_type"] == "mass":
            m_diff = sa_range_1[-1] - sa_range_1[0]
            t_lap_diff = sa_t_lap[-1] - sa_t_lap[0]
            fuel_cons_diff = sa_fuel_cons[-1] - sa_fuel_cons[0]

            print("Average sensitivity of lap time to mass: %.3f s/kg" % (t_lap_diff / m_diff))
            print("Average sensitivity of fuel consumption to mass: %.5f kg/kg" % (fuel_cons_diff / m_diff))
            print("-" * 50)

        else:
            pass
            # TODO: implementation of COG and aero variation missing
    if debug_opts["use_elemons_result"]:
        iter_tag = "iteration"
        vehicle_tag = "vehicle"
        mass_tag = "mass (kg)"
        c_d_tag = "Cd(c_w_a)"
        laptime_tag = "laptime (s)"
        energy_tag = "energy (kJ)"
        header_row = [iter_tag, vehicle_tag, mass_tag, c_d_tag, laptime_tag, energy_tag]

        with open(resultsfile, 'a') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(header_row)
            for i in range(len(sa_t_lap)):
                csvwriter.writerow([sa_iter[i]] +
                                    [solver_opts["vehicle"]] + 
                                    [("%.1f" %  sa_mass[i])] +          
                                    [("%.3f" %  sa_c_d[i])] +          
                                    [("%.3f" %  sa_t_lap[i])] +
                                    [("%.2f" %  (sa_fuel_cons[i] / 1000.0))])

    output_path = os.path.join(output_path_velprofile, "velprofile_" + 
                               track_opts["trackname"].lower() + "_" +
                               solver_opts["vehicle"].lower() + ".csv")

    tmp_data = np.column_stack((lap.trackobj.dists_cl[:-1], lap.vel_cl[:-1]))

    with open(output_path, "wb") as fh:
        np.savetxt(fh, tmp_data, fmt="%.5f,%.5f", header="distance_m,vel_mps")

    # ------------------------------------------------------------------------------------------------------------------
    # PLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if not sa_opts["use_sa"]:
        if debug_opts["use_plot"]:
            lap.plot_overview()
            # lap.plot_revs_gears()
    else:
        if sa_opts["sa_type"] == "mass":
            # lap time
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sa_range_1, sa_t_lap, "x")
            ax.set_xlim(sa_range_1[0], sa_range_1[-1])
            ax.set_ylim(sa_t_lap[0], sa_t_lap[-1])
            ax.set_title("SA of lap time to mass")
            ax.set_xlabel("mass m in kg")
            ax.set_ylabel("lap time t in s")
            plt.grid()
            plt.show()

            # fuel consumption
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sa_range_1, sa_fuel_cons, "x")
            ax.set_xlim(sa_range_1[0], sa_range_1[-1])
            ax.set_ylim(sa_fuel_cons[0], sa_fuel_cons[-1])
            ax.set_title("SA of fuel consumption to mass")
            ax.set_xlabel("mass m in kg")
            ax.set_ylabel("fuel consumption in kg/lap")
            plt.grid()
            plt.show()

        elif sa_opts["sa_type"] == "elemons_mass":
            # lap time
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sa_range_1, sa_t_lap, "x")
            ax.set_xlim(sa_range_1[0], sa_range_1[-1])
            ax.set_ylim(sa_t_lap[0], sa_t_lap[-1])
            ax.set_title("SA of lap time to mass")
            ax.set_xlabel("mass m in kg")
            ax.set_ylabel("lap time t in s")
            ax.set_title('Lap Times\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
            plt.grid()
            plt.show()

            # fuel (energy) consumption
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(sa_range_1, sa_fuel_cons, "x")
            ax.set_xlim(sa_range_1[0], sa_range_1[-1])
            ax.set_ylim(sa_fuel_cons[0], sa_fuel_cons[-1])
            ax.set_title("SA of energy consumption to mass")
            ax.set_xlabel("mass m in kg")
            ax.set_ylabel("energy consumption in kJ/lap")
            ax.set_title('Energy Consumption\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
            plt.grid()
            plt.show()

        elif sa_opts["sa_type"] == "elemons_mass_cd":
            # Good oold data mainpulation to get it graphing            
            Laptime_dataframe = pd.DataFrame({mass_tag: sa_mass[:], c_d_tag: sa_c_d[:], laptime_tag: sa_t_lap[:]})
            Energy_dataframe = pd.DataFrame({mass_tag: sa_mass[:], c_d_tag: sa_c_d[:], energy_tag: sa_fuel_cons[:]})

            Energy_array = Energy_dataframe.pivot_table(index=mass_tag, columns=c_d_tag, values=energy_tag).T.values
            Laptime_array = Laptime_dataframe.pivot_table(index=mass_tag, columns=c_d_tag, values=laptime_tag).T.values

            mass_unique = np.sort(np.unique(sa_mass))
            c_d_unique = np.sort(np.unique(sa_c_d))

            mass_array, c_d_array = np.meshgrid(mass_unique, c_d_unique)

            fig = plt.figure()
            fig2 = plt.figure()
            ax1 = fig.add_subplot(111,projection='3d')
            ax1.set_xlabel("Mass (kg)")
            ax1.set_ylabel("Coeff of Drag - Cd")
            ax1.set_zlabel("Energy per lap (kJ) * ")
            ax1.set_title('Energy Per Lap\nvehicle: ' +  solver_opts["vehicle"] + '\ntrack: ' + track_opts["trackname"])
            ax1.plot_surface(mass_array, c_d_array, Energy_array)

            ax2 = fig2.add_subplot(111,projection='3d')
            ax2.set_xlabel('Mass (kg)')
            ax2.set_ylabel('Coeff of Drag - Cd')
            ax2.set_zlabel('Lap Time (sec)')
            ax2.set_title('Lap Times\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
            ax2.plot_surface(mass_array, c_d_array, Laptime_array)
            plt.show()
    # ------------------------------------------------------------------------------------------------------------------
    # CI TESTING -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # pickle lap object for possible CI testing
    result_objects_file_path = os.path.join(output_path_testobjects,
                                            "testobj_laptimesim_" + track_opts["trackname"] + ".pkl")
    with open(result_objects_file_path, 'wb') as fh:
        pickle.dump(lap, fh)

    return lap  # return required in case of CI testing


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':
    # Importing config from sim_config.toml
    config = toml.load("sim_config.toml")
 
    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------
    # See sim_config.toml for variable descriptions
    track_opts_ = config["track_opts_"]

    solver_opts_ = config["solver_opts_"]
    driver_opts_ = config["driver_opts_"]
    
    # These are because of this bug: https://github.com/uiri/toml/issues/270
    config["sa_opts_"]["range_1"][2] = int(config["sa_opts_"]["range_1"][2])
    config["sa_opts_"]["range_2"][2] = int(config["sa_opts_"]["range_2"][2])
    sa_opts_ = config["sa_opts_"]

    debug_opts_ = config["debug_opts_"]

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(track_opts=track_opts_,
         solver_opts=solver_opts_,
         driver_opts=driver_opts_,
         sa_opts=sa_opts_,
         debug_opts=debug_opts_)
