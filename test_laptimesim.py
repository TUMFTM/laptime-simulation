import main_laptimesim
import os
import pickle
import numpy as np


def test_laptimesim():
    # user input
    track_opts_ = {"trackname": "Shanghai",
                   "flip_track": False,
                   "mu_weather": 1.0,
                   "interp_stepsize_des": 5.0,
                   "curv_filt_width": 10.0,
                   "use_drs1": True,
                   "use_drs2": True,
                   "use_pit": False}
    solver_opts_ = {"vehicle": "F1_Shanghai.ini",
                    "series": "F1",
                    "limit_braking_weak_side": 'FA',
                    "v_start": 100.0 / 3.6,
                    "find_v_start": True,
                    "max_no_em_iters": 5,
                    "es_diff_max": 1.0}
    driver_opts_ = {"vel_subtr_corner": 0.5,
                    "vel_lim_glob": None,
                    "yellow_s1": False,
                    "yellow_s2": False,
                    "yellow_s3": False,
                    "yellow_throttle": 0.3,
                    "initial_energy": 4.58e6,
                    "em_strategy": "FCFB",
                    "use_recuperation": True,
                    "use_lift_coast": False,
                    "lift_coast_dist": 10.0}
    sa_opts_ = {"use_sa": False,
                "sa_type": "mass",
                "range_1": [733.0, 833.0, 5],
                "range_2": None}
    debug_opts_ = {"use_plot": False,
                   "use_debug_plots": False,
                   "use_plot_comparison_tph": False,
                   "use_print": False,
                   "use_print_result": False}

    # simulation call
    lap = main_laptimesim.main(track_opts=track_opts_,
                               solver_opts=solver_opts_,
                               driver_opts=driver_opts_,
                               sa_opts=sa_opts_,
                               debug_opts=debug_opts_)

    # testing
    repo_path_ = os.path.dirname(os.path.abspath(__file__))
    target_lap_path_ = os.path.join(repo_path_, ".github", "testobjects", "testobj_laptimesim_Shanghai.pkl")

    with open(target_lap_path_, 'rb') as fh:
        target_lap = pickle.load(fh)

    assert np.allclose(target_lap.vel_cl, lap.vel_cl)


# testing --------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    test_laptimesim()
