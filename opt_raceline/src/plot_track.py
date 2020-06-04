import numpy as np
import matplotlib.pyplot as plt
import trajectory_planning_helpers as tph


def plot_track(track_imp: np.ndarray,
               track_interp: np.ndarray,
               mapfilepath: str = "",
               filepath_tr_plot: str = "") -> None:
    """
    author:
    Alexander Heilmeier

    date:
    08.01.2020

    .. description::
    This function is used to plot the imported as well as the smoothed track.

    .. inputs::
    :param track_imp:           imported track containing four columns [x_m, y_m, w_tr_right_m, w_tr_left_m]
    :type track_imp:            np.ndarray
    :param track_interp:        interpolated/smoothed track with four columns [x_m, y_m, w_tr_right_m, w_tr_left_m]
    :type track_interp:         np.ndarray
    :param mapfilepath:         path to track map file (optional)
    :type mapfilepath:          str
    :param filepath_tr_plot:    path to save plot (*.png)
    :type filepath_tr_plot:     str
    """

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE IMPORTED TRACK -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate normal vectors
    track_imp_cl = np.vstack((track_imp, track_imp[0]))
    el_lengths_imp_cl = np.sqrt(np.sum(np.power(np.diff(track_imp_cl[:, :2], axis=0), 2), axis=1))

    normvecs_normalized_imp = tph.calc_splines.calc_splines(path=track_imp_cl[:, :2],
                                                            el_lengths=el_lengths_imp_cl,
                                                            use_dist_scaling=True)[3]
    normvecs_normalized_imp_cl = np.vstack((normvecs_normalized_imp, normvecs_normalized_imp[0]))

    # calculate boundaries
    bound_right_imp_cl = track_imp_cl[:, :2] + normvecs_normalized_imp_cl * np.expand_dims(track_imp_cl[:, 2], 1)
    bound_left_imp_cl = track_imp_cl[:, :2] - normvecs_normalized_imp_cl * np.expand_dims(track_imp_cl[:, 3], 1)

    # ------------------------------------------------------------------------------------------------------------------
    # PREPARE SMOOTHED TRACK -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # calculate normal vectors
    track_interp_cl = np.vstack((track_interp, track_interp[0]))
    el_lengths_interp_cl = np.sqrt(np.sum(np.power(np.diff(track_interp_cl[:, :2], axis=0), 2), axis=1))

    normvecs_normalized_interp = tph.calc_splines.calc_splines(path=track_interp_cl[:, :2],
                                                               el_lengths=el_lengths_interp_cl,
                                                               use_dist_scaling=True)[3]
    normvecs_normalized_interp_cl = np.vstack((normvecs_normalized_interp, normvecs_normalized_interp[0]))

    # calculate boundaries
    track_interp_cl = np.vstack((track_interp, track_interp[0]))
    bound_right_interp_cl = (track_interp_cl[:, :2]
                             + normvecs_normalized_interp_cl * np.expand_dims(track_interp_cl[:, 2], 1))
    bound_left_interp_cl = (track_interp_cl[:, :2]
                            - normvecs_normalized_interp_cl * np.expand_dims(track_interp_cl[:, 3], 1))

    # ------------------------------------------------------------------------------------------------------------------
    # CREATE PLOT ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # plot everything
    fig = plt.figure(figsize=(12.0, 8.0))
    ax = fig.add_subplot(111)

    ax.plot(track_imp_cl[:, 0], track_imp_cl[:, 1], 'r--', label="centerline (imported)")
    ax.plot(bound_right_imp_cl[:, 0], bound_right_imp_cl[:, 1], 'r-', label="right boundary (imported)")
    ax.plot(bound_left_imp_cl[:, 0], bound_left_imp_cl[:, 1], 'r-', label="left boundary (imported)")

    ax.plot(track_interp_cl[:, 0], track_interp_cl[:, 1], 'k--', label="centerline (smoothed)")
    ax.plot(bound_right_interp_cl[:, 0], bound_right_interp_cl[:, 1], 'k-', label="right boundary (smoothed)")
    ax.plot(bound_left_interp_cl[:, 0], bound_left_interp_cl[:, 1], 'k-', label="left boundary (smoothed)")

    point1_arrow = track_interp_cl[0, :2]
    point2_arrow = track_interp_cl[10, :2]
    vec_arrow = point2_arrow - point1_arrow
    ax.arrow(point1_arrow[0], point1_arrow[1], vec_arrow[0], vec_arrow[1], head_width=50.0, head_length=50.0,
             fc='g', ec='g', width=25.0)

    ax.legend()
    ax.set_aspect("equal", "datalim")
    ax.set_xlabel("east in m")
    ax.set_ylabel("north in m")
    plt.title("Comparison between imported and smoothed track")
    plt.grid()

    # set track map as background
    if mapfilepath:
        x_min = np.amin(track_imp_cl[:, 0])
        x_max = np.amax(track_imp_cl[:, 0])
        y_min = np.amin(track_imp_cl[:, 1])
        y_max = np.amax(track_imp_cl[:, 1])

        img = plt.imread(mapfilepath)
        ax.imshow(img, zorder=0, extent=[x_min, x_max, y_min, y_max])  # [left, right, bottom, top]

    if filepath_tr_plot:
        plt.savefig(filepath_tr_plot, dpi=250)
    plt.show()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
