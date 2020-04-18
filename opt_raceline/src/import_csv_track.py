import numpy as np


def import_csv_track(trackfilepath: str) -> np.ndarray:
    """
    author:
    Alexander Heilmeier

    date:
    23.12.2019

    .. description::
    This function loads the track data [x, y, w_tr_right, w_tr_left] from the given .csv file.

    .. inputs::
    :param trackfilepath:   path to csv track file
    :type trackfilepath:    str

    .. outputs::
    :return track:          track data [x, y, w_tr_right, w_tr_left]
    :rtype track:           np.ndarray
    """

    # ------------------------------------------------------------------------------------------------------------------
    # LOAD TRACK DATA FROM FILE ----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    track = np.loadtxt(trackfilepath, comments='#', delimiter=',')  # [x, y, w_tr_right, w_tr_left]

    if not track.shape[1] == 4:
        print("WARNING: .csv track file must supply four columns: [x, y, w_tr_right, w_tr_left]. Continuing under the"
              " assumption that the third column contains the total track width!")
        track = np.column_stack((track[:, :2], track[:, 2] / 2, track[:, 2] / 2))

    return track


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    pass
