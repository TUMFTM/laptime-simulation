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

    elif imp_opts["mode"] == "centerline" and track_pars["track_width"] is None:
        raise RuntimeError("The track_width must be set within the user input section since it is not supplied within"
                           " the .geojson file!")

    # ------------------------------------------------------------------------------------------------------------------
    # INITIALIZATION STUFF ---------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create output folders (if not existing)
    outfolderpath_cl = os.path.join(repo_path, 'opt_raceline', "output", "centerlines_smoothed")
    outfolderpath_tr = os.path.join(repo_path, 'opt_raceline', "output", "tracks_smoothed")
    outfolderpath_rl = os.path.join(repo_path, 'opt_raceline', "output", "racelines")

    os.makedirs(outfolderpath_cl, exist_ok=True)
    os.makedirs(outfolderpath_tr, exist_ok=True)
    os.makedirs(outfolderpath_rl, exist_ok=True)

    # set paths
    if imp_opts["mode"] == "centerline":
        trackfilepath = os.path.join(repo_path, 'opt_raceline', "input", "centerlines",
                                     track_pars["location"] + ".geojson")
    elif imp_opts["mode"] == "track":
        trackfilepath = os.path.join(repo_path, 'opt_raceline', "input", "tracks", track_pars["location"] + ".csv")
    else:
        raise RuntimeError("Unknown mode!")

    mapfolderpath = os.path.join(repo_path, 'opt_raceline', "input", "maps")
    mapfilepath = ""

    for mapfile in os.listdir(mapfolderpath):
        if track_pars["location"] in mapfile:
            mapfilepath = os.path.join(mapfolderpath, mapfile)
            break

    outfilepath_cl = os.path.join(outfolderpath_cl, track_pars["location"] + ".csv")
    outfilepath_tr = os.path.join(outfolderpath_tr, track_pars["location"] + ".csv")
    outfilepath_rl = os.path.join(outfolderpath_rl, track_pars["location"] + ".csv")

    # paths for saving the plots
    outfilepath_tr_plot = os.path.join(outfolderpath_tr, track_pars["location"] + ".png")
    outfilepath_rl_plot1 = os.path.join(outfolderpath_rl, track_pars["location"] + "_raceline.png")
    outfilepath_rl_plot2 = os.path.join(outfolderpath_rl, track_pars["location"] + "_curvature.png")

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
        raise RuntimeError("Unknown file type!")

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
        print("WARNING: Large deviation (>2%%) between calculated (%.0fm) and stored length (%.0fm) of the race track!"
              % (length_centerline, track_pars["track_length"]))

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

    # check if imported track should be flipped, i.e. reverse direction
    if imp_opts["flip_imp_track"]:
        track_interp = np.flipud(track_interp)

    # check if imported track should be reordered for a new starting point
    if imp_opts["set_new_start"]:
        ind_start = np.argmin(np.hypot(track_interp[:, 0] - imp_opts["new_start"][0],
                                       track_interp[:, 1] - imp_opts["new_start"][1]))
        track_interp = np.roll(track_interp, track_interp.shape[0] - ind_start, axis=0)

    # export centerline (smoothed) as [x, y] and track (smoothed) as [x, y, w_tr_right, w_tr_left] for other tools
    with open(outfilepath_cl, "wb") as fh:
        np.savetxt(fh, track_interp[:, :2], fmt='%.6f,%.6f', header="x_m,y_m")

    with open(outfilepath_tr, "wb") as fh:
        np.savetxt(fh, track_interp, fmt='%.6f,%.6f,%.3f,%.3f', header="x_m,y_m,w_tr_right_m,w_tr_left_m")

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
                                               mapfilepath=mapfilepath,
                                               filepath_tr_plot=outfilepath_tr_plot)

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
        np.savetxt(fh, raceline_opt_interp, fmt='%.6f,%.6f', header="x_m,y_m")

    print("RESULT: Length of minimum curvature path is %.1fm" % np.sum(spline_lengths_opt))
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
        point2_arrow = track_interp[10, :2]
        vec_arrow = point2_arrow - point1_arrow

        # plot track including optimized path
        plt.figure(figsize=(12.0, 8.0))
        plt.plot(track_interp[:, 0], track_interp[:, 1], "k--", linewidth=0.7)
        plt.plot(veh_bound1_real[:, 0], veh_bound1_real[:, 1], "c", linewidth=0.5)
        plt.plot(veh_bound2_real[:, 0], veh_bound2_real[:, 1], "c", linewidth=0.5)
        plt.plot(bound1[:, 0], bound1[:, 1], "k-", linewidth=0.7)
        plt.plot(bound2[:, 0], bound2[:, 1], "k-", linewidth=0.7)
        plt.plot(raceline_opt_interp[:, 0], raceline_opt_interp[:, 1], "r-", linewidth=0.7)
        plt.grid()
        ax = plt.gca()
        ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1], head_width=50.0, head_length=50.0,
                 fc='g', ec='g', width=25.0)
        ax.set_aspect("equal", "datalim")
        plt.xlabel("east in m")
        plt.ylabel("north in m")
        plt.savefig(outfilepath_rl_plot1, dpi=250)
        plt.show()

    if plot_opts["raceline_curv"]:
        plt.figure(figsize=(12.0, 8.0))
        plt.plot(s_points_opt_interp, kappa_opt)
        plt.grid()
        plt.xlabel("distance in m")
        plt.ylabel("curvature in rad/m")
        plt.savefig(outfilepath_rl_plot2, dpi=250)
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# MAIN FUNCTION CALL ---------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == '__main__':

    # ------------------------------------------------------------------------------------------------------------------
    # USER INPUT -------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # set file containing the track centerline (.geojson) or the track centerline and track widths (.csv)
    #   -> file type is chosen by the "mode" option in the import options below
    #   -> if the track widths are supplied in the file (.csv) the track_width option here is ignored

    # F1 ---------------------------------------------------------------------------------------------------------------

    track_pars_ = {"location": "Austin",
                   "track_length": 5513.0,
                   "track_width": None}

    # track_pars_ = {"location": "Budapest",
    #                "track_length": 4381.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Catalunya",
    #                "track_length": 4655.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Hockenheim",
    #                "track_length": 4574.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Melbourne",
    #                "track_length": 5303.0,
    #                "track_width": None}

    # track_pars_ = {"location": "IMS",
    #                "track_length": 4023.0,
    #                "track_width": 15.3}

    # track_pars_ = {"location": "MexicoCity",
    #                "track_length": 4304.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Montreal",
    #                "track_length": 4361.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Monza",
    #                "track_length": 5793.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Sakhir",
    #                "track_length": 5412.0,
    #                "track_width": None}

    # track_pars_ = {"location": "SaoPaulo",
    #                "track_length": 4309.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Sepang",
    #                "track_length": 5543.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Shanghai",
    #                "track_length": 5451.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Silverstone",
    #                "track_length": 5891.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Sochi",
    #                "track_length": 5848.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Spa",
    #                "track_length": 7004.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Spielberg",
    #                "track_length": 4318.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Suzuka",
    #                "track_length": 5807.0,
    #                "track_width": None}

    # track_pars_ = {"location": "YasMarina",
    #                "track_length": 5554.0,
    #                "track_width": None}

    # DTM --------------------------------------------------------------------------------------------------------------

    # track_pars_ = {"location": "BrandsHatch",
    #                "track_length": 3908.0,
    #                "track_width": None}

    # track_pars_ = {"location": "MoscowRaceway",
    #                "track_length": 4070.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Norisring",
    #                "track_length": 2300.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Nuerburgring",
    #                "track_length": 5148.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Oschersleben",
    #                "track_length": 3696.0,
    #                "track_width": None}

    # track_pars_ = {"location": "Zandvoort",
    #                "track_length": 4320.0,
    #                "track_width": None}

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

    imp_opts_ = {"mode": "track",
                 "flip_imp_track": False,
                 "set_new_start": True,
                 "new_start": [0.0, 0.0],
                 "plot_track": True}

    # spline regression smoothing options ------------------------------------------------------------------------------
    # k_reg:    [-] order of BSplines -> standard: 3
    # s_reg:    [-] smoothing factor -> range [1.0, 100.0] (play a little bit)

    reg_smooth_opts_ = {"k_reg": 3,
                        "s_reg": 40.0}

    # set stepsizes used during optimization ---------------------------------------------------------------------------
    # stepsize_prep:                [m] used for linear interpolation before spline approximation
    # stepsize_reg:                 [m] used for spline interpolation after spline approximation (stepsize during opt.)
    # stepsize_interp_after_opt:    [m] used for spline interpolation after optimization

    stepsize_opts_ = {"stepsize_prep": 2.0,
                      "stepsize_reg": 5.0,
                      "stepsize_interp_after_opt": 5.0}

    # optimization problem options -------------------------------------------------------------------------------------
    # width_opt:                [m] vehicle width for optimization incl. safety distance
    # curvlim:                  [rad/m] curvature limit for optimization
    # iqp_iters_min:            [-] minimum number of iterations for the IQP
    # iqp_curverror_allowed:    [rad/m] maximum allowed curvature error for the IQP

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
