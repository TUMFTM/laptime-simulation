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
        date = datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p")
        resultsfile = os.path.join(repo_path, "laptimesim", "output", "results-{}".format(date))

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
        elif sa_opts["sa_type"] == "elemons-mass":
            # initialize this pass variables
            sa_t_lap = np.zeros(sa_opts["range_1"][2])
            sa_fuel_cons = np.zeros(sa_opts["range_1"][2])

        
            with open(resultsfile, 'w') as csvfile:
                spamwriter = csv.writer(csvfile )
                spamwriter.writerow( ['vehicle', 'recuperation', 'mass (kg)', 'Cd(c_w_a)','lapttime (s)', 'energy (kJ)'])

            for i, cur_mass in enumerate(sa_range_1):
                print("SA: Starting solver run (%i)" % (i + 1))

                # change mass of vehicle
                lap.driverobj.carobj.pars_general["m"] = cur_mass

                # simulate lap and save lap time
                lap.simulate_lap()
                sa_t_lap[i] = lap.t_cl[-1]
                sa_fuel_cons[i] = lap.fuel_cons_cl[-1]
                if solver_opts["series"] == "FE":
                    # RMH for formula E (electric) vehicles overide gas fuel and publish electrical energy'
                    sa_fuel_cons[i] = lap.e_cons_cl[-1] 

                # record the lap data
                with open(resultsfile, 'a') as csvfile:
                    spamwriter = csv.writer(csvfile )
                    spamwriter.writerow( [solver_opts["vehicle"]] +
                                        ["{}".format( lap.driverobj.pars_driver["use_recuperation"])] +
                                        [("%.1f" %  lap.driverobj.carobj.pars_general["m"])] +          
                                        [("%.3f" %  lap.driverobj.carobj.pars_general["c_w_a"])] +          
                                        [("%.3f" %  lap.t_cl[-1])] +
                                        [("%.2f" %  (lap.e_cons_cl[-1] / 1000.0))])
                # reset lap
                lap.reset_lap()

                print("SA: Finished solver run (%i)" % (i + 1))

        # perform eLemons analysis
        elif sa_opts["sa_type"] == "elemons_mass_cd":
            results_header = ['iteration', 'vehicle', 'mass (kg)', 'Cd(c_w_a)','laptime (s)', 'energy (kJ)']
            # initialize this pass variables that collect results
            sa_t_lap = np.zeros(sa_opts["range_1"][2])
            sa_fuel_cons = np.zeros(sa_opts["range_1"][2])

            # open a fresh file to accumulate results
           
            with open(resultsfile, 'w') as csvfile:
                spamwriter = csv.writer(csvfile)
                spamwriter.writerow(results_header)
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
                    sa_t_lap[i] = lap.t_cl[-1]
                    sa_fuel_cons[i] = lap.fuel_cons_cl[-1]
                    if solver_opts["series"] == "FE":
                        # RMH for formula E (electric) vehicles overide gas fuel and publish electrical energy'
                        sa_fuel_cons[i] = lap.e_cons_cl[-1] 
                    iter += 1
                    with open(resultsfile, 'a') as csvfile:
                        spamwriter = csv.writer(csvfile )
                        spamwriter.writerow([iter] +
                                            [solver_opts["vehicle"]] + 
                                            [("%.1f" %  lap.driverobj.carobj.pars_general["m"])] +          
                                            [("%.3f" %  lap.driverobj.carobj.pars_general["c_w_a"])] +          
                                            [("%.3f" %  lap.t_cl[-1])] +
                                            [("%.2f" %  (lap.e_cons_cl[-1] / 1000.0))])
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
        #print("elemons results")
        #print("#VehicleName, mass_kg, laptime_s, perLapEnergyConsumption_kj")
        #print("VehicleName:  {} ".format( solver_opts["vehicle"]))
        #print("use_recuperation:  {} ".format( lap.driverobj.pars_driver["use_recuperation"]))
        #print("mass:  %.1f kg" %  lap.driverobj.carobj.pars_general["m"])
        #print("Lap time: %.3f s, Consumption: %.2f kJ/lap" %( lap.t_cl[-1], lap.e_cons_cl[-1] / 1000.0))
    # write velocity profile output
    #output_path = os.path.join(output_path_velprofile, "velprofile_" + track_opts["trackname"].lower() + ".csv")
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
        elif sa_opts["sa_type"] == "elemons_mass_cd":
            # get axes from csv lists
            contour_data = pd.read_csv(resultsfile)

            Energy_array = contour_data.pivot_table(index='mass (kg)', columns='Cd(c_w_a)', values = 'energy (kJ)').T.values
            Laptime_array = contour_data.pivot_table(index='mass (kg)', columns='Cd(c_w_a)', values='laptime (s)').T.values

            mass_unique = np.sort(contour_data['mass (kg)'].unique())
            c_d_unique = np.sort(contour_data['Cd(c_w_a)'].unique())

            mass_array, c_d_array = np.meshgrid(mass_unique, c_d_unique)

            fig = plt.figure()
            fig2 = plt.figure()
            ax1 = fig.add_subplot(111,projection='3d')
            ax2 = fig2.add_subplot(111,projection='3d')

            ax1.plot_surface(mass_array, c_d_array, Energy_array)
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

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # F1 qualifying mode:   DRS activated, EM strategy FCFB
    # F1 race mode:         DRS as desired, EM strategy LBP
    # FE qualifying mode:   DRS deactivated, EM strategy FCFB
    # FE race mode:         DRS deactivated, EM strategy FCFB + lift&coast
    # tracks must be unclosed, i.e. last point != first point!

    # track options ----------------------------------------------------------------------------------------------------
    # trackname:            track name of desired race track (file must be available in the input folder)
    # flip_track:           switch to flip track direction if required
    # mu_weather:           [-] factor to consider wet track, e.g. by mu_weather = 0.6
    # interp_stepsize_des:  [m], desired stepsize after interpolation of the input raceline points
    # curv_filt_width:      [m] window width of moving average filter -> set None for deactivation
    # use_drs1:             DRS zone 1 switch
    # use_drs2:             DRS zone 2 switch
    # use_pit:              activate pit stop (requires _pit track file!)

    track_opts_ = {"trackname": "HighPlainsFullTrack",
                   "flip_track": False,
                   "mu_weather": 1.0,
                   "interp_stepsize_des": 5.0,
                   "curv_filt_width": 10.0,
                   "use_drs1": True,
                   "use_drs2": True,
                   "use_pit": False}

    # solver options ---------------------------------------------------------------------------------------------------
    # vehicle:                  vehicle parameter file
    # series:                   F1, FE
    # limit_braking_weak_side:  can be None, 'FA', 'RA', 'all' -> set if brake force potential should be determined
    #                           based on the weak (i.e. inner) side of the car, e.g. when braking into a corner
    # v_start:                  [m/s] velocity at start
    # find_v_start:             determine the real velocity at start
    # max_no_em_iters:          maximum number of iterations for EM recalculation
    # es_diff_max:              [J] stop criterion -> maximum difference between two solver runs

    solver_opts_ = {"vehicle": "FE_Berlin.ini",
                    "series": "FE",
                    "limit_braking_weak_side": 'FA',
                    "v_start": 100.0 / 3.6,
                    "find_v_start": True,
                    "max_no_em_iters": 5,
                    "es_diff_max": 1.0}

    # driver options ---------------------------------------------------------------------------------------------------
    # vel_subtr_corner: [m/s] velocity subtracted from max. cornering vel. since drivers will not hit the maximum
    #                   perfectly
    # vel_lim_glob:     [m/s] velocity limit, set None if unused
    # yellow_s1:        yellow flag in sector 1
    # yellow_s2:        yellow flag in sector 2
    # yellow_s3:        yellow flag in sector 3
    # yellow_throttle:  throttle position in a yellow flag sector
    # initial_energy:   [J] initial energy (F1: max. 4 MJ/lap, FE Berlin: 4.58 MJ/lap)
    # em_strategy:      FCFB, LBP, LS, NONE -> FCFB = First Come First Boost, LBP = Longest (time) to Breakpoint,
    #                   LS = Lowest Speed, FE requires FCFB as it only drives in electric mode!
    # use_recuperation: set if recuperation by e-motor and electric turbocharger is allowed or not (lift&coast is
    #                   currently only considered with FCFB)
    # use_lift_coast:   switch to turn lift and coast on/off
    # lift_coast_dist:  [m] lift and coast before braking point

    driver_opts_ = {"vel_subtr_corner": 0.5,
                    "vel_lim_glob": None,
                    "yellow_s1": False,
                    "yellow_s2": False,
                    "yellow_s3": False,
                    "yellow_throttle": 0.3,
                    "initial_energy": 4.58e6,
                    "em_strategy": "FCFB",
                    "use_recuperation": False,
                    "use_lift_coast": False,
                    "lift_coast_dist": 10.0}

    # sensitivity analysis options -------------------------------------------------------------------------------------
    # use_sa:   switch to deactivate sensitivity analysis
    # sa_type:  'mass', 'aero', 'cog' 'elemons-mass'
    # range_1:  range of parameter variation [start, end, number of steps]
    # range_2:  range of parameter variation [start, end, number of steps] -> CURRENTLY NOT IMPLEMENTED
    # RMH Note:
    #  sa_type          Range 1 variable     Range 2 variable
    # 'mass'            vehicle mass         not used - set to 'None' without quotes
    # 'areo'            feature not implement 
    # 'cog'             feature not implement 
    # 'elemons-mass'    vehicle mass         not used - set to 'None' without quotes 
    # 'elemons-mass-cd' vehicle mass         Cd c_w_a (coefficient of drag)

    '''
    # Original TUM settings 
    sa_opts_ = {"use_sa": False,
                "sa_type": "mass",
                "range_1": [733.0, 833.0, 5],
                "range_2": None}
    '''
    # eLemons modifications to allow iteration over ranges of our interest
    sa_opts_ = {"use_sa": True,
                "sa_type": "elemons_mass_cd",
                "range_1": [700.0, 1200.0, 10],
                "range_2": [1.10, 1.50, 5]}

    # debug options ----------------------------------------------------------------------------------------------------
    # use_plot:                 plot results
    # use_debug_plots:          plot additional plots for debugging
    # use_plot_comparison_tph:  calculate velocity profile with TPH FB solver and plot a comparison
    # use_print:                set if prints to console should be used or not (does not suppress hints/warnings)
    # use_print_result:         set if result should be printed to console or not
    # use_elemons_result:       set if eLemons result should be printed (added) to csv file or not

    debug_opts_ = {"use_plot": False,
                   "use_debug_plots": False,
                   "use_plot_comparison_tph": True,
                   "use_print": True,
                   "use_print_result": True,
                   "use_elemons_result": True}

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(track_opts=track_opts_,
         solver_opts=solver_opts_,
         driver_opts=driver_opts_,
         sa_opts=sa_opts_,
         debug_opts=debug_opts_)
