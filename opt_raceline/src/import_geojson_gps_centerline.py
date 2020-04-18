import json
import utm
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox
from collections import OrderedDict


def import_geojson_gps_centerline(trackfilepath: str,
                                  mapfilepath: str = "") -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    date:
    08.02.2019

    .. description::
    This function loads the GeoJSON data obtained from OpenStreetMap and provides some functionality to chose only the
    desired data parts. How to obtain the GeoJSON data is explained in the readme.

    .. inputs::
    :param trackfilepath:   path to GeoJSON centerline file
    :type trackfilepath:    str
    :param mapfilepath:     path to track map file
    :type mapfilepath:      str

    .. outputs::
    :return centerline:     centerline of track [x_m, y_m]
    :rtype centerline:      np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD GEOJSON DATA ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # load data from json file
    with open(trackfilepath, "r") as fh:
        gps_data = json.load(fh)

    # convert GPS data to xy coordinates
    gps_data_xy = []

    for cur_sec in gps_data["features"]:
        # get angle data out of dict
        tmp_gps = cur_sec["geometry"]["coordinates"]

        # create numpy array and convert angle data to xy coordinates
        tmp_no_points = len(tmp_gps)
        tmp_xy = np.zeros((tmp_no_points, 2))

        for i in range(tmp_no_points):
            tmp_xy[i] = utm.from_latlon(tmp_gps[i][1], tmp_gps[i][0])[:2]

        # save current section into list
        gps_data_xy.append(tmp_xy)

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT ALL THE SECTIONS --------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    fig = plt.figure()
    ax = fig.add_subplot(111)

    # plot all the sections and save handles into list
    plt_handles = []

    for i, cur_sec in enumerate(gps_data_xy):
        plt_handles.append(ax.plot(cur_sec[:, 0], cur_sec[:, 1], label="section ID " + str(i))[0])

    # create legend and save handle
    leg = ax.legend(loc="upper left")

    # couple legend entries with real lines
    elements = OrderedDict()

    for leg_element, line_element in zip(leg.get_lines(), plt_handles):
        leg_element.set_picker(10)  # 5 pts tolerance
        elements[leg_element] = line_element

    ax.set_aspect("equal", "datalim")
    ax.set_title("Choose sections to use for centerline creation!")
    ax.set_xlabel("x in m")
    ax.set_ylabel("y in m")
    plt.grid()

    # set track map as background
    if mapfilepath:
        x_min = None
        x_max = None
        y_min = None
        y_max = None

        for cur_sec in gps_data_xy:
            tmp_x_min = np.amin(cur_sec[:, 0])
            tmp_x_max = np.amax(cur_sec[:, 0])
            tmp_y_min = np.amin(cur_sec[:, 1])
            tmp_y_max = np.amax(cur_sec[:, 1])

            if x_min is None:
                x_min = tmp_x_min
                x_max = tmp_x_max
                y_min = tmp_y_min
                y_max = tmp_y_max

            else:
                if tmp_x_min < x_min:
                    x_min = tmp_x_min
                if tmp_x_max > x_max:
                    x_max = tmp_x_max
                if tmp_y_min < y_min:
                    y_min = tmp_y_min
                if tmp_y_max > y_max:
                    y_max = tmp_y_max

        img = plt.imread(mapfilepath)
        ax.imshow(img, zorder=0, extent=[x_min, x_max, y_min, y_max])  # [left, right, bottom, top]

    # create textbox to be able to get ID of start element from the user
    ax_textbox = plt.axes([0.84, 0.82, 0.05, 0.05])
    text_box = TextBox(ax_textbox, 'Set start ID to:', initial='0')

    # connect to canvas to be able to pick within plot
    fig.canvas.mpl_connect('pick_event', lambda event: __onpick(event=event,
                                                                elements=elements,
                                                                fig_handle=fig))

    plt.show()

    # ------------------------------------------------------------------------------------------------------------------
    # PLOT IS NOW CLOSED -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # get start ID from textbox
    id_start = int(text_box.text)

    # create bool array containing which sections are active after plot was closed
    act_secs = np.full(len(gps_data_xy), False)

    for i, (key, value) in enumerate(elements.items()):
        if value.get_visible():
            act_secs[i] = True

    # get relevant sections
    use_secs = np.arange(0, len(gps_data_xy))[act_secs]

    # ------------------------------------------------------------------------------------------------------------------
    # GET CORRECT ORDER OF RELEVANT SECTIONS ---------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # create list containing the used sections (gps_data_xy_used) and list for the sections in correct order
    # (centerline)
    gps_data_xy_used = []
    centerline = []

    for cur_sec_id in use_secs:
        if cur_sec_id == id_start:
            centerline.append(gps_data_xy[cur_sec_id])
        else:
            gps_data_xy_used.append(gps_data_xy[cur_sec_id])

    # start from start element and search next element until all used sections are processed
    while gps_data_xy_used:
        # loop through all remaining sections and find the closest to the current endpoint
        cur_end_pt = centerline[-1][-1]
        d_min = None
        id_min = None
        inverted_min = None  # handle the case that some sections were recorded in the other direction

        for i, cur_sec in enumerate(gps_data_xy_used):
            tmp_dist_1 = np.hypot(cur_sec[0, 0] - cur_end_pt[0], cur_sec[0, 1] - cur_end_pt[1])     # current direction
            tmp_dist_2 = np.hypot(cur_sec[-1, 0] - cur_end_pt[0], cur_sec[-1, 1] - cur_end_pt[1])   # other direction
            tmp_d_min = min(tmp_dist_1, tmp_dist_2)

            if d_min is None or tmp_d_min < d_min:
                d_min = tmp_d_min
                id_min = i

                if tmp_dist_1 < tmp_dist_2:
                    inverted_min = False
                else:
                    inverted_min = True

            if math.isclose(d_min, 0.0):
                break

        # print warning if found next section does not fit perfectly
        if not math.isclose(d_min, 0.0):
            print("WARNING: Did not find a perfectly matching section, minimum distance is %.2fm!" % d_min)

        # append closest section to centerline list in correct direction
        if inverted_min:
            centerline.append(np.flipud(gps_data_xy_used.pop(id_min)))
        else:
            centerline.append(gps_data_xy_used.pop(id_min))

    # unite list of numpy arrays into one array
    centerline = np.vstack(centerline)

    # use unique to remove equal points (start and end points of the sections)
    ids_sort = np.unique(centerline, axis=0, return_index=True)[1]
    centerline = centerline[np.sort(ids_sort)]

    # set new coordinate center
    centerline = centerline - centerline[0, :]

    return centerline


def __onpick(event, elements, fig_handle):
    # on the pick event, find the orig line corresponding to the legend entry and toggle the visibility
    leg_element = event.artist
    line_element = elements[leg_element]
    vis = not line_element.get_visible()
    line_element.set_visible(vis)

    # change alpha of the line in the legend so we can see which lines have been toggled
    if vis:
        leg_element.set_alpha(1.0)
    else:
        leg_element.set_alpha(0.2)

    # update plot
    fig_handle.canvas.draw()


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
