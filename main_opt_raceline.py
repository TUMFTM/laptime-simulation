import opt_raceline
import numpy as np
import time
import os
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph
import pkg_resources

"""
author:
Alexander Heilmeier

date:
08.02.2019

.. description::
This script has to be executed to generate a minimum curvature raceline on the basis of a given reference track.
"""


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION --------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

def main(track_pars: dict,
         plot_opts: dict,
         imp_opts: dict,
         reg_smooth_opts: dict,
         stepsize_opts: dict,
         optim_opts_mincurv: dict) -> None:

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
    # CHECK USER INPUT -------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if imp_opts["mode"] == "track" and track_pars["track_width"] is not None:
        print("WARNING: The track_width set within the user input section is ignored since the track widths are"
              " supplied within the .csv file!")

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION STUFF ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create output folders (if not existing)
    outfolderpath_cl = os.path.join(repo_path, 'opt_raceline', "output", "centerlines")
    outfolderpath_rl = os.path.join(repo_path, 'opt_raceline', "output", "racelines")

    os.makedirs(outfolderpath_cl, exist_ok=True)
    os.makedirs(outfolderpath_rl, exist_ok=True)

    # set paths
    if imp_opts["mode"] == "centerline":
        trackfilepath = os.path.join(repo_path, 'opt_raceline', "input", "centerlines", "geojson",
                                     track_pars["location"] + ".geojson")
    elif imp_opts["mode"] == "track":
        trackfilepath = os.path.join(repo_path, 'opt_raceline', "input", "tracks", "csv",
                                     track_pars["location"] + ".csv")
    else:
        raise ValueError("Unknown mode!")

    mapfilepath = os.path.join(repo_path, 'opt_raceline', "input", "maps", track_pars["location"] + "_2017.png")
    outfilepath_cl = os.path.join(outfolderpath_cl, track_pars["location"] + ".csv")
    outfilepath_rl = os.path.join(outfolderpath_rl, track_pars["location"] + ".csv")

    if not os.path.isfile(mapfilepath):
        mapfilepath = ""

    # ------------------------------------------------------------------------------------------------------------------
    # IMPORT TRACK -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # import geojson file with centerline
    if trackfilepath.endswith(".geojson"):
        # load centerline
        refline_imp = opt_raceline.src.import_geojson_gps_centerline.\
            import_geojson_gps_centerline(trackfilepath=trackfilepath,
                                          mapfilepath=mapfilepath)

        # set artificial track widths in case of centerline
        track_imp = np.column_stack((refline_imp, np.ones((refline_imp.shape[0], 2)) * track_pars["track_width"] / 2))

    elif trackfilepath.endswith(".csv"):
        # load track
        track_imp = opt_raceline.src.import_csv_track.import_csv_track(trackfilepath=trackfilepath)

    else:
        raise ValueError("Unknown file type!")

    # check if imported track should be flipped, i.e. reverse direction
    if imp_opts["flip_imp_track"]:
        track_imp = np.flipud(track_imp)

    # check if imported track should be reordered for a new starting point
    if imp_opts["set_new_start"]:
        ind_start = np.argmin(np.hypot(track_imp[:, 0] - imp_opts["new_start"][0],
                                       track_imp[:, 1] - imp_opts["new_start"][1]))
        track_imp = np.roll(track_imp, track_imp.shape[0] - ind_start, axis=0)

    # ------------------------------------------------------------------------------------------------------------------
    # SCALE TRACK TO CORRECT LENGTH ------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate current track length
    centerline_cl = np.vstack((track_imp[:, :2], track_imp[0, :2]))
    length_centerline = np.sum(np.sqrt(np.sum(np.power(np.diff(centerline_cl, axis=0), 2), axis=1)))

    # scale centerline (not track widths)
    if track_pars["track_length"] is not None:
        scale_factor = track_pars["track_length"] / length_centerline
    else:
        scale_factor = 1.0
    track_imp[:, :2] *= scale_factor

    # check for too large deviation of lengths
    if np.abs(1.0 - scale_factor) > 0.02:
        print("WARNING: Large deviation (>2%) between calculated and stored length of the race track!")

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE TRACK ----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # use spline approximation to prepare centerline input
    track_interp = tph.spline_approximation.\
        spline_approximation(track=track_imp,
                             stepsize_prep=stepsize_opts["stepsize_prep"],
                             stepsize_reg=stepsize_opts["stepsize_reg"],
                             k_reg=reg_smooth_opts["k_reg"],
                             s_reg=reg_smooth_opts["s_reg"],
                             debug=True)

    # export centerline as [x, y] for other tools
    with open(outfilepath_cl, "wb") as fh:
        np.savetxt(fh, track_interp[:, :2], fmt='%.6f, %.6f', header="x_m, y_m")

    # calculate splines
    refpath_interp_cl = np.vstack((track_interp[:, :2], track_interp[0, :2]))

    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = tph.calc_splines.\
        calc_splines(path=refpath_interp_cl,
                     use_dist_scaling=True)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT TRACK INCLUDING TRACK BOUNDARIES ----------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # plot imported as well as smoothed track if option was set
    if imp_opts["plot_track"]:
        opt_raceline.src.plot_track.plot_track(track_imp=track_imp,
                                               track_interp=track_interp,
                                               mapfilepath=mapfilepath)

    # ------------------------------------------------------------------------------------------------------------------
    # CHECK SPLINE NORMALS FOR CROSSING POINTS -------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    normals_crossing = tph.check_normals_crossing.check_normals_crossing(track=track_interp,
                                                                         normvec_normalized=normvec_normalized_interp,
                                                                         horizon=10)

    if normals_crossing:
        raise IOError("At least two spline normals are crossed, check input or increase smoothing factor!")

    # ------------------------------------------------------------------------------------------------------------------
    # CALL OPTIMIZATION ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    alpha_opt, track_interp, normvec_normalized_interp = tph.iqp_handler.\
        iqp_handler(reftrack=track_interp,
                    normvectors=normvec_normalized_interp,
                    A=a_interp,
                    kappa_bound=optim_opts_mincurv["curvlim"],
                    w_veh=optim_opts_mincurv["width_opt"],
                    print_debug=True,
                    plot_debug=False,
                    stepsize_interp=stepsize_opts["stepsize_reg"],
                    iters_min=optim_opts_mincurv["iqp_iters_min"],
                    curv_error_allowed=optim_opts_mincurv["iqp_curverror_allowed"])

    # ------------------------------------------------------------------------------------------------------------------
    # INTERPOLATE SPLINES TO SMALL DISTANCES BETWEEN RACELINE POINTS ---------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    raceline_opt_interp, a_opt, coeffs_x_opt, coeffs_y_opt, spline_inds_opt_interp, t_vals_opt_interp,\
        s_points_opt_interp, spline_lengths_opt, el_lengths_opt_interp_cl = tph.create_raceline.\
        create_raceline(refline=track_interp[:, :2],
                        normvectors=normvec_normalized_interp,
                        alpha=alpha_opt,
                        stepsize_interp=stepsize_opts["stepsize_interp_after_opt"])

    # ------------------------------------------------------------------------------------------------------------------
    # CALCULATE HEADING AND CURVATURE ----------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate heading and curvature (analytically)
    psi_vel_opt, kappa_opt = tph.calc_head_curv_an.\
        calc_head_curv_an(coeffs_x=coeffs_x_opt,
                          coeffs_y=coeffs_y_opt,
                          ind_spls=spline_inds_opt_interp,
                          t_spls=t_vals_opt_interp)

    # ------------------------------------------------------------------------------------------------------------------
    # EXPORT -----------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # export raceline
    with open(outfilepath_rl, "wb") as fh:
        np.savetxt(fh, raceline_opt_interp, fmt='%.6f, %.6f', header="x_m, y_m")

    print("RESULT: Length of minimum curvature path is %.1fm. Include this into the track parameters of the lap time"
          " simulation!" % np.sum(spline_lengths_opt))
    print("INFO: Finished creation of trajectory:", time.strftime("%H:%M:%S"))

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT RESULTS -----------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    if plot_opts["raceline"]:
        # calc required data
        bound1 = track_interp[:, :2] + normvec_normalized_interp * np.expand_dims(track_interp[:, 2], 1)
        bound2 = track_interp[:, :2] - normvec_normalized_interp * np.expand_dims(track_interp[:, 3], 1)

        normvec_normalized_opt = tph.calc_normal_vectors.calc_normal_vectors(psi_vel_opt)

        veh_bound1_real = raceline_opt_interp + normvec_normalized_opt * optim_opts_mincurv["width_opt"] / 2
        veh_bound2_real = raceline_opt_interp - normvec_normalized_opt * optim_opts_mincurv["width_opt"] / 2

        point1_arrow = track_interp[0, :2]
        point2_arrow = track_interp[3, :2]
        vec_arrow = point2_arrow - point1_arrow

        # plot track including optimized path
        plt.figure()
        plt.plot(track_interp[:, 0], track_interp[:, 1], "k--", linewidth=0.7)
        plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
        plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
        plt.plot(bound1[:, 0], bound1[:, 1], "k-", linewidth=0.7)
        plt.plot(bound2[:, 0], bound2[:, 1], "k-", linewidth=0.7)
        plt.plot(raceline_opt_interp[:, 0], raceline_opt_interp[:, 1], "r-", linewidth=0.7)
        plt.grid()
        ax = plt.gca()
        ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1], head_width=7.0, head_length=7.0,
                 fc='g', ec='g')
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.show()

    if plot_opts["raceline_curv"]:
        plt.figure()
        plt.plot(s_points_opt_interp, kappa_opt, linewidth=0.7)
        plt.grid()
        plt.xlabel("distance in m")
        plt.ylabel("curvature in radpm")
        plt.legend(["kappa after opt"])
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set file containing the track centerline (.geojson) or the track centerline and track widths (.csv) --------------
    # if track widths are supplied by the file, the track_width setting here will be ignored! --------------------------

    track_pars_ = {"location": "Budapest",
                   "track_length": 4381.0,
                   "track_width": 12.0}

    # track_pars_ = {"location": "Montreal",
    #                "track_length": 4361.0,
    #                "track_width": 10.0}

    # track_pars_ = {"location": "Shanghai",
    #                "track_length": 5451.0,
    #                "track_width": 14.0}

    # track_pars_ = {"location": "Silverstone",
    #                "track_length": 5891.0,
    #                "track_width": 13.0}

    # track_pars_ = {"location": "Spielberg",
    #                "track_length": 4318.0,
    #                "track_width": 12.0}

    # track_pars_ = {"location": "YasMarina",
    #                "track_length": 5554.0,
    #                "track_width": 15.0}

    # set plot options -------------------------------------------------------------------------------------------------
    # raceline:         plot optimized path on the race track
    # raceline_curv:    plot curvature profile of optimized path

    plot_opts_ = {"raceline": True,
                  "raceline_curv": True}

    # set import options -----------------------------------------------------------------------------------------------
    # mode:             "track" or "centerline" -> track is supplied as .csv and contains [x, y, w_tr_right, w_tr_left],
    #                   centerline is supplied as .geojson
    # flip_imp_track:   flip imported track to reverse direction
    # set_new_start:    set new starting point (changes order, not coordinates)
    # new_start:        [x_m, y_m] coordinates of new starting point
    # plot_track:       plot imported as well as smoothed track

    imp_opts_ = {"mode": "centerline",
                 "flip_imp_track": False,
                 "set_new_start": False,
                 "new_start": [125.0, -244.0],
                 "plot_track": True}

    # spline regression smoothing options ------------------------------------------------------------------------------
    # k_reg:    [-] order of BSplines -> standard: 3
    # s_reg:    [-] smoothing factor -> range [1.0, 100.0] (play a little bit)

    reg_smooth_opts_ = {"k_reg": 3,
                        "s_reg": 20}

    # set stepsizes used during optimization ---------------------------------------------------------------------------
    # stepsize_prep:                [m] used for linear interpolation before spline approximation
    # stepsize_reg:                 [m] used for spline interpolation after spline approximation (stepsize during opt.)
    # stepsize_interp_after_opt:    [m] used for spline interpolation after optimization
    # FSS: stepsizes 2.0m
    stepsize_opts_ = {"stepsize_prep": 2.0,
                      "stepsize_reg": 5.0,
                      "stepsize_interp_after_opt": 5.0}

    # optimization problem options -------------------------------------------------------------------------------------
    # width_opt:                [m] vehicle width for optimization incl. safety distance
    # curvlim:                  [rad/m] curvature limit for optimization
    # iqp_iters_min:            [-] minimum number of iterations for the IQP
    # iqp_curverror_allowed:    [rad/m] maximum allowed curvature error for the IQP
    # FSS: curv_lim 0.2rad/m
    optim_opts_mincurv_ = {"width_opt": 1.5,
                           "curvlim": 0.12,
                           "iqp_iters_min": 3,
                           "iqp_curverror_allowed": 0.01}

    # ------------------------------------------------------------------------------------------------------------------
    # SIMULATION CALL --------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    main(track_pars=track_pars_,
         plot_opts=plot_opts_,
         imp_opts=imp_opts_,
         reg_smooth_opts=reg_smooth_opts_,
         stepsize_opts=stepsize_opts_,
         optim_opts_mincurv=optim_opts_mincurv_)
