import numpy as np
import math
import matplotlib.pyplot as plt
import json
import trajectory_planning_helpers as tph
import configparser


class Track(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The class provides functions related to the track, e.g. curvature and distance calculations.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__zone_inds",
                 "__pars_track",
                 "__raceline",
                 "__kappa",
                 "__mu",
                 "__vel_lim",
                 "__drs",
                 "__stepsize",
                 "__no_points",
                 "__no_points_cl",
                 "__dists_cl")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, track_opts: dict, track_pars: dict, trackfilepath: str, elevationfilepath: str, vel_lim_glob: float = np.inf,
                 yellow_s1: bool = False, yellow_s2: bool = False, yellow_s3: bool = False):

        # save given track parameters, load track parameters and append the relevant ones to pars_track
        # MH: sorry this naming is very confusing
        # all of the track options get put together into a dictionary anyway
        # so I'm leaving the confusing names and just changing in the import section
        # of this object
        self.pars_track = track_opts

        # reassign
        for key in track_pars:
            self.pars_track[key] = track_pars[key]

        # load raceline
        self.raceline = np.loadtxt(trackfilepath, comments='#', delimiter=',')

        # load elevation profile
        self.elevationprofile = np.loadtxt(elevationfilepath, delimiter=',')

        # set friction values artificially as long as no real friction values available and limit them to a valid range
        self.mu = np.ones(self.raceline.shape[0]) * self.pars_track["mu_mean"] * self.pars_track["mu_weather"]

        if np.any(self.mu < 0.5) or np.any(self.mu > 1.3):
            print("WARNING: Friction values seem invalid, friction values are limited to 0.5 <= mu <= 1.3!")
            self.mu[self.mu < 0.5] = 0.5
            self.mu[self.mu > 1.3] = 1.3

        # flip track if required
        if self.pars_track["flip_track"]:
            self.raceline = np.flipud(self.raceline)
            self.mu = np.flipud(self.mu)

        # prepare raceline (interpolation, distance and curvature calculation)
        self.__prep_raceline()

        # set sector boundaries
        self.zone_inds = {}  # initialize zone_inds
        self.__get_zone_bounds()

        # set speed limits
        self.vel_lim = np.full(self.no_points, vel_lim_glob)  # [m/s] contains speed limit for whole track
        self.__set_pitspeed_limit()

        # set DRS
        self.drs = np.full(self.no_points, False)  # bool array contains the points where DRS is activated
        self.__set_drs()

        # adjust DRS zones to possible yellow flags
        self.__adj_drs_yellow_flag(yellow_s1=yellow_s1,
                                   yellow_s2=yellow_s2,
                                   yellow_s3=yellow_s3)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_zone_inds(self) -> dict: return self.__zone_inds
    def __set_zone_inds(self, x: dict) -> None: self.__zone_inds = x
    zone_inds = property(__get_zone_inds, __set_zone_inds)

    def __get_pars_track(self) -> dict: return self.__pars_track
    def __set_pars_track(self, x: dict) -> None: self.__pars_track = x
    pars_track = property(__get_pars_track, __set_pars_track)

    def __get_mu(self) -> np.ndarray: return self.__mu
    def __set_mu(self, x: np.ndarray) -> None: self.__mu = x
    mu = property(__get_mu, __set_mu)

    def __get_raceline(self) -> np.ndarray: return self.__raceline
    def __set_raceline(self, x: np.ndarray) -> None: self.__raceline = x
    raceline = property(__get_raceline, __set_raceline)

    def __get_kappa(self) -> np.ndarray: return self.__kappa
    def __set_kappa(self, x: np.ndarray) -> None: self.__kappa = x
    kappa = property(__get_kappa, __set_kappa)

    def __get_vel_lim(self) -> np.ndarray: return self.__vel_lim
    def __set_vel_lim(self, x: np.ndarray) -> None: self.__vel_lim = x
    vel_lim = property(__get_vel_lim, __set_vel_lim)

    def __get_drs(self) -> np.ndarray: return self.__drs
    def __set_drs(self, x: np.ndarray) -> None: self.__drs = x
    drs = property(__get_drs, __set_drs)

    def __get_stepsize(self) -> float: return self.__stepsize
    def __set_stepsize(self, x: float) -> None: self.__stepsize = x
    stepsize = property(__get_stepsize, __set_stepsize)

    def __get_no_points(self) -> int: return self.__no_points
    def __set_no_points(self, x: int) -> None: self.__no_points = x
    no_points = property(__get_no_points, __set_no_points)

    def __get_no_points_cl(self) -> int: return self.__no_points_cl
    def __set_no_points_cl(self, x: int) -> None: self.__no_points_cl = x
    no_points_cl = property(__get_no_points_cl, __set_no_points_cl)

    def __get_dists_cl(self) -> np.ndarray: return self.__dists_cl
    def __set_dists_cl(self, x: np.ndarray) -> None: self.__dists_cl = x
    dists_cl = property(__get_dists_cl, __set_dists_cl)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    """The following functions contain the code required for linear interpolation and numerical curvature calculation.
    Due to the better approximation by splines they were replaced and are therefore not used anymore."""

    # def __calc_dists_cl(self) -> np.ndarray:
    #     """Raceline is an array containing unclosed x and y coords: [x, y]. Output is in m. The distances are returned
    #     for a closed track."""
    #
    #     raceline_cl = np.vstack((self.raceline, self.raceline[0, :]))
    #     dists_cl_tmp = np.cumsum(np.sqrt(np.power(np.diff(raceline_cl[:, 0]), 2)
    #                                      + np.power(np.diff(raceline_cl[:, 1]), 2)))
    #     dists_cl_tmp = np.insert(dists_cl_tmp, 0, 0.0)
    #
    #     return dists_cl_tmp
    #
    # def __interp_raceline(self) -> None:
    #     """
    #     Raceline is an unclosed array containing x and y coords: [x, y], stepsize_des is the desired stepsize after
    #     interpolation. Returns the unclosed raceline, the closed distances to every point, the stepsize after
    #     interpolation and the number of points (unclosed) after interpolation.
    #     """
    #
    #     dists_cl_preinterp = self.__calc_dists_cl()
    #     no_points = int(np.round(dists_cl_preinterp[-1] / self.pars_track["interp_stepsize_des"]))
    #     self.stepsize = dists_cl_preinterp[-1] / no_points
    #     dists_cl = np.arange(0, no_points + 1) * self.stepsize
    #
    #     raceline_preinterp_cl = np.vstack((self.raceline, self.raceline[0, :]))
    #     x_interp_cl = np.interp(dists_cl, dists_cl_preinterp, raceline_preinterp_cl[:, 0])
    #     y_interp_cl = np.interp(dists_cl, dists_cl_preinterp, raceline_preinterp_cl[:, 1])
    #     self.raceline = np.column_stack((x_interp_cl[:-1], y_interp_cl[:-1]))  # unclosed
    #
    #     mu_preinterp_cl = np.append(self.mu, self.mu[0])
    #     self.mu = np.interp(dists_cl, dists_cl_preinterp, mu_preinterp_cl)[:-1]  # unclosed
    #
    # def __calc_curvature(self) -> None:
    #     """Raceline is an array containing x and y coords: [x, y]. Output is in rad/m."""
    #
    #     # create temporary path including 3 points before and after original path for more accurate gradients
    #     raceline_tmp = np.vstack((self.raceline[-3:, :], self.raceline, self.raceline[:3, :]))
    #
    #     # calculate curvature using np.gradient
    #     dx = np.gradient(raceline_tmp[:, 0])
    #     ddx = np.gradient(dx)
    #     dy = np.gradient(raceline_tmp[:, 1])
    #     ddy = np.gradient(dy)
    #
    #     num = dx * ddy - ddx * dy
    #     denom = np.power(np.sqrt(np.power(dx, 2) + np.power(dy, 2)), 3)
    #
    #     kappa = num / denom
    #
    #     # remove temporary points
    #     self.kappa = kappa[3:-3]
    #
    # def __smooth_curvature(self) -> None:
    #     self.kappa = np.convolve(self.kappa, np.ones(self.pars_track["curv_filt_window"])
    #                              / self.pars_track["curv_filt_window"], mode="same")

    def __prep_raceline(self) -> None:
        """This function prepares the inserted raceline in several steps: interpolation, distance calculation,
        curvature calculation and curvature smoothing. The workflow is based on a spline representation of the
        raceline. Raceline is an unclosed array containing x and y coords: [x, y]. stepsize_des is the desired stepsize
        after interpolation. Curvature is in rad/m."""

        # get spline coefficients (use inserted raceline as basis for the splines)
        raceline_cl = np.vstack((self.raceline, self.raceline[0]))

        coeffs_x_cl, coeffs_y_cl = tph.calc_splines.calc_splines(path=raceline_cl,
                                                                 use_dist_scaling=True)[:2]

        # calculate spline lengths and distances to points before interpolation
        spline_lenghts_cl = tph.calc_spline_lengths.calc_spline_lengths(coeffs_x=coeffs_x_cl,
                                                                        coeffs_y=coeffs_y_cl,
                                                                        quickndirty=False)

        dists_cl_preinterp = np.insert(np.cumsum(spline_lenghts_cl), 0, 0.0)

        # interpolate splines to the desired (equal) stepsize (now unclosed raceline)
        self.raceline, ind_spls, t_spls, dists_interp = tph.interp_splines.\
            interp_splines(spline_lengths=spline_lenghts_cl,
                           coeffs_x=coeffs_x_cl,
                           coeffs_y=coeffs_y_cl,
                           incl_last_point=False,
                           stepsize_approx=self.pars_track["interp_stepsize_des"])

        # save stepsize
        self.stepsize = dists_interp[1] - dists_interp[0]

        # save distances to every point
        self.dists_cl = np.append(dists_interp, dists_cl_preinterp[-1])

        # save number of points
        self.no_points = self.raceline.shape[0]
        self.no_points_cl = self.no_points + 1

        # (linear) interpolation of friction values that matched the originally inserted raceline
        mu_preinterp_cl = np.append(self.mu, self.mu[0])
        self.mu = np.interp(self.dists_cl[:-1], dists_cl_preinterp, mu_preinterp_cl)  # unclosed

        # calculate curvature profile (unclosed)
        self.kappa = tph.calc_head_curv_an.calc_head_curv_an(coeffs_x=coeffs_x_cl,
                                                             coeffs_y=coeffs_y_cl,
                                                             ind_spls=ind_spls,
                                                             t_spls=t_spls)[1]

        # smooth curvature profile if desired
        if self.pars_track["curv_filt_width"] is not None and self.pars_track["curv_filt_width"] > self.stepsize:

            # calculate window size of convolution filter based on desired filter width (+1 to include middle point)
            window_size = int(round(self.pars_track["curv_filt_width"] / self.stepsize)) + 1

            # handle case that window_size is not odd
            if window_size % 2 == 0:
                print("WARNING: Convolution filter window size for kappa is set to %i instead of %i (must be odd)"
                      % (window_size + 1, window_size))
                window_size += 1

            # apply filter
            self.kappa = tph.conv_filt.conv_filt(signal=self.kappa,
                                                 filt_window=window_size,
                                                 closed=True)

    def __get_zone_bounds(self) -> None:
        # sectors ------------------------------------------------------------------------------------------------------
        # set indices
        self.zone_inds["s12"] = np.argmin(np.abs(self.pars_track["s12"] - self.dists_cl))
        self.zone_inds["s23"] = np.argmin(np.abs(self.pars_track["s23"] - self.dists_cl))

        # pit ----------------------------------------------------------------------------------------------------------
        # initialize pit zone indices
        self.zone_inds["pit_in"] = 0
        self.zone_inds["pit_out"] = 0

        # set indices
        if self.pars_track["use_pit"]:
            self.zone_inds["pit_in"] = np.argmin(np.abs(self.pars_track["pit_in"] - self.dists_cl))
            self.zone_inds["pit_out"] = np.argmin(np.abs(self.pars_track["pit_out"] - self.dists_cl))

        # drs ----------------------------------------------------------------------------------------------------------
        # initialize drs zone indices
        self.zone_inds["drs1_a"] = 0
        self.zone_inds["drs1_d"] = 0
        self.zone_inds["drs2_a"] = 0
        self.zone_inds["drs2_d"] = 0

        # check if drs zones are set properly
        if self.pars_track["use_drs1"] and (math.isclose(self.pars_track["drs1_act"], 0.0)
                                            or math.isclose(self.pars_track["drs1_deact"], 0.0)):
            print("WARNING: DRS zone 1 is not set properly. Therefore, DRS is deactivated in zone 1!")
            self.pars_track["use_drs1"] = False

        if self.pars_track["use_drs2"] and (math.isclose(self.pars_track["drs2_act"], 0.0)
                                            or math.isclose(self.pars_track["drs2_deact"], 0.0)):
            print("WARNING: DRS zone 2 is not set properly. Therefore, DRS is deactivated in zone 2!")
            self.pars_track["use_drs2"] = False

        # set indices
        if self.pars_track["use_drs1"]:
            self.zone_inds["drs1_a"] = np.argmin(np.abs(self.pars_track["drs1_act"] - self.dists_cl))
            self.zone_inds["drs1_d"] = np.argmin(np.abs(self.pars_track["drs1_deact"] - self.dists_cl))

        if self.pars_track["use_drs2"]:
            self.zone_inds["drs2_a"] = np.argmin(np.abs(self.pars_track["drs2_act"] - self.dists_cl))
            self.zone_inds["drs2_d"] = np.argmin(np.abs(self.pars_track["drs2_deact"] - self.dists_cl))

    def __set_pitspeed_limit(self) -> None:
        if self.pars_track["use_pit"]:
            self.vel_lim[0:self.zone_inds["pit_out"]] = self.pars_track["pitspeed"]
            self.vel_lim[self.zone_inds["pit_in"]:] = self.pars_track["pitspeed"]

    def __set_drs(self) -> None:
        # check if pit stop causes DRS deactivation in zone 1
        if self.pars_track["use_pit"]:
            print("WARNING: DRS zone 1 gets deactivated due to pit stop!")
            self.pars_track["use_drs1"] = False

        # DRS zone 1
        if self.pars_track["use_drs1"]:
            if self.zone_inds["drs1_a"] < self.zone_inds["drs1_d"]:
                # common case
                self.drs[self.zone_inds["drs1_a"]:self.zone_inds["drs1_d"]] = True
            else:
                # DRS zone is split by start/finish line
                self.drs[self.zone_inds["drs1_a"]:] = True
                self.drs[:self.zone_inds["drs1_d"]] = True

        # DRS zone 2
        if self.pars_track["use_drs2"]:
            if self.zone_inds["drs2_a"] < self.zone_inds["drs2_d"]:
                # common case
                self.drs[self.zone_inds["drs2_a"]:self.zone_inds["drs2_d"]] = True
            else:
                # DRS zone is split by start/finish line
                self.drs[self.zone_inds["drs2_a"]:] = True
                self.drs[:self.zone_inds["drs2_d"]] = True

    def __adj_drs_yellow_flag(self, yellow_s1: bool, yellow_s2: bool, yellow_s3: bool):
        """Adjust DRS zones to yellow flags -> deactivate DRS if yellow flags are active in the according sectors."""

        # DRS zone 1 ---------------------------------------------------------------------------------------------------
        if self.pars_track["use_drs1"]:
            if yellow_s1 and (0 <= self.zone_inds["drs1_a"] < self.zone_inds["s12"]
                              or 0 <= self.zone_inds["drs1_d"] < self.zone_inds["s12"]) \
                or yellow_s2 and (self.zone_inds["s12"] <= self.zone_inds["drs1_a"] < self.zone_inds["s23"]
                                  or self.zone_inds["s12"] <= self.zone_inds["drs1_d"] < self.zone_inds["s23"]) \
                or yellow_s3 and (self.zone_inds["s23"] <= self.zone_inds["drs1_a"]
                                  or self.zone_inds["s23"] <= self.zone_inds["drs1_d"]):

                print("WARNING: DRS zone 1 gets deactivated due to yellow flag!")
                self.pars_track["use_drs1"] = False

        # DRS zone 2 ---------------------------------------------------------------------------------------------------
        if self.pars_track["use_drs2"]:
            if yellow_s1 and (0 <= self.zone_inds["drs2_a"] < self.zone_inds["s12"]
                              or 0 <= self.zone_inds["drs2_d"] < self.zone_inds["s12"]) \
                    or yellow_s2 and (self.zone_inds["s12"] <= self.zone_inds["drs2_a"] < self.zone_inds["s23"]
                                      or self.zone_inds["s12"] <= self.zone_inds["drs2_d"] < self.zone_inds["s23"]) \
                    or yellow_s3 and (self.zone_inds["s23"] <= self.zone_inds["drs2_a"]
                                      or self.zone_inds["s23"] <= self.zone_inds["drs2_d"]):

                print("WARNING: DRS zone 2 gets deactivated due to yellow flag!")
                self.pars_track["use_drs2"] = False

    def check_track(self) -> None:
        """Recalculate raceline based on curvature. Raceline is an array containing x and y coords: [x, y].
        kappa contains the curvature in rad/m. stepsize is the stepsize after interpolation in m. heading_start is in
        rad."""

        # create required arrays
        raceline_re = np.zeros((self.no_points, 2))
        phi_re = np.zeros(self.no_points)

        # calculate start heading of original track
        dx = self.raceline[1, 0] - self.raceline[0, 0]
        dy = self.raceline[1, 1] - self.raceline[0, 1]
        heading_start = math.atan2(dy, dx) - 0.5 * math.pi

        # set initial heading such that origin is "north" along the y-axis and start point is equal to original track
        phi_re[0] = 0.5 * np.pi + heading_start
        raceline_re[0] = self.raceline[0]

        # calculate raceline points based on curvature according to Velenis 2005 DOI: 10.1109/ACC.2005.1470288
        for i in range(self.no_points - 1):
            phi_re[i + 1] = phi_re[i] + (self.kappa[i] + self.kappa[i + 1]) / 2 * self.stepsize  # heading
            raceline_re[i + 1, 0] = raceline_re[i, 0] + math.cos((phi_re[i + 1] + phi_re[i]) / 2) * self.stepsize  # x
            raceline_re[i + 1, 1] = raceline_re[i, 1] + math.sin((phi_re[i + 1] + phi_re[i]) / 2) * self.stepsize  # y

        # plot results
        plt.figure()
        plt.plot(self.raceline[:, 0], self.raceline[:, 1])
        plt.plot(raceline_re[:, 0], raceline_re[:, 1])

        plt.axis("equal")
        plt.title("Raceline comparison")
        plt.xlabel("x in m")
        plt.ylabel("y in m")
        plt.legend(["Original raceline", "Recalculated raceline"])

        plt.show()

    def plot_curvature(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)

        ax.plot(self.dists_cl[:-1], self.kappa)

        ax.set_title("Track curvature")
        ax.set_xlabel("s in m")
        ax.set_ylabel("kappa in rad/m")

    def plot_trackmap(self, mapfilepath: str = "") -> None:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # plot raceline
        ax1.plot(self.raceline[:, 0], self.raceline[:, 1], "k-")

        # plot DRS zones
        if self.pars_track["use_drs1"]:
            if self.zone_inds["drs1_a"] < self.zone_inds["drs1_d"]:
                # common case
                ax1.plot(self.raceline[self.zone_inds["drs1_a"]:self.zone_inds["drs1_d"], 0],
                         self.raceline[self.zone_inds["drs1_a"]:self.zone_inds["drs1_d"], 1],
                         "g--", linewidth=3.0)
            else:
                # DRS zone is split by start/finish line
                ax1.plot(self.raceline[self.zone_inds["drs1_a"]:, 0],
                         self.raceline[self.zone_inds["drs1_a"]:, 1],
                         "g--", linewidth=3.0)
                ax1.plot(self.raceline[:self.zone_inds["drs1_d"], 0],
                         self.raceline[:self.zone_inds["drs1_d"], 1],
                         "g--", linewidth=3.0)

        if self.pars_track["use_drs2"]:
            if self.zone_inds["drs2_a"] < self.zone_inds["drs2_d"]:
                # common case
                ax1.plot(self.raceline[self.zone_inds["drs2_a"]:self.zone_inds["drs2_d"], 0],
                         self.raceline[self.zone_inds["drs2_a"]:self.zone_inds["drs2_d"], 1],
                         "g--", linewidth=3.0)
            else:
                # DRS zone is split by start/finish line
                ax1.plot(self.raceline[self.zone_inds["drs2_a"]:, 0],
                         self.raceline[self.zone_inds["drs2_a"]:, 1],
                         "g--", linewidth=3.0)
                ax1.plot(self.raceline[:self.zone_inds["drs2_d"], 0],
                         self.raceline[:self.zone_inds["drs2_d"], 1],
                         "g--", linewidth=3.0)

        # plot pit
        if self.pars_track["use_pit"]:
            ax1.plot(self.raceline[:self.zone_inds["pit_out"], 0],
                     self.raceline[:self.zone_inds["pit_out"], 1],
                     "r--", linewidth=3.0)
            ax1.plot(self.raceline[self.zone_inds["pit_in"]:, 0],
                     self.raceline[self.zone_inds["pit_in"]:, 1],
                     "r--", linewidth=3.0)

        # plot arrow showing the driving direction
        ax1.arrow(self.raceline[0, 0], self.raceline[0, 1],
                  self.raceline[10, 0] - self.raceline[0, 0],
                  self.raceline[10, 1] - self.raceline[0, 1],
                  head_width=30.0, width=10.0)

        # plot dots at start/finish and at the sector boundaries
        ax1.plot(self.raceline[0, 0], self.raceline[0, 1], "k.", markersize=13.0)
        ax1.plot(self.raceline[self.zone_inds["s12"], 0],
                 self.raceline[self.zone_inds["s12"], 1], "k.", markersize=13.0)
        ax1.plot(self.raceline[self.zone_inds["s23"], 0],
                 self.raceline[self.zone_inds["s23"], 1], "k.", markersize=13.0)

        ax1.set_aspect("equal", "datalim")
        ax1.set_title("track map: " + self.pars_track["trackname"])
        ax1.set_xlabel("x in m")
        ax1.set_ylabel("y in m")

        # create empty handles for text and point to be able to click within plot
        txt_handle = ax1.text(0.05, 0.95, "Track distance of selected point: ", transform=plt.gcf().transFigure)
        pt_handle = ax1.plot([], [], "r.", markersize=13.0)[0]

        # set track picture as background
        if mapfilepath:
            x_min = np.amin(self.raceline[:, 0])
            x_max = np.amax(self.raceline[:, 0])
            y_min = np.amin(self.raceline[:, 1])
            y_max = np.amax(self.raceline[:, 1])

            img = plt.imread(mapfilepath)
            ax1.imshow(img, zorder=0, extent=[x_min, x_max, y_min, y_max])  # [left, right, bottom, top]

        # connect to canvas to be able to click within plot
        fig.canvas.mpl_connect('button_press_event', lambda event: self.__onpick(event=event,
                                                                                 pt_handle=pt_handle,
                                                                                 txt_handle=txt_handle,
                                                                                 fig_handle=fig))

        plt.show()

    def __onpick(self, event, pt_handle, txt_handle, fig_handle):
        # get position of click event
        pos_click = [event.xdata, event.ydata]

        # determine nearest point on track
        dists = math.hypot(self.raceline[:, 0] - pos_click[0], self.raceline[:, 1] - pos_click[1])
        ind = np.argpartition(dists, 1)[0]
        cur_node = self.raceline[ind]
        cur_dist = self.dists_cl[ind]

        # update position of text and point handles
        pt_handle.set_data(cur_node[0], cur_node[1])
        txt_handle.set_text("Track distance of selected point: %.0fm" % cur_dist)

        # re-draw figure
        fig_handle.canvas.draw()


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
