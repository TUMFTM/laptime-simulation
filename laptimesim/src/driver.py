import numpy as np
import math
from laptimesim.src.car_hybrid import CarHybrid
from laptimesim.src.car_electric import CarElectric
from laptimesim.src.track import Track


class Driver(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    25.12.2018

    .. description::
    The file provides functions related to the energy management strategy. Therefore, it determines when the hybrid
    system is used during a lap.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__carobj",
                 "__pars_driver",
                 "__em_boost_use",
                 "__throttle_pos",
                 "__no_points_lac")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, carobj: CarHybrid or CarElectric, pars_driver: dict, trackobj: Track,
                 stepsize: float = 5.0):
        """stepsize must only be supplied for lift and coast strategy."""

        # save car object and parameters
        self.carobj = carobj
        self.pars_driver = pars_driver

        # --------------------------------------------------------------------------------------------------------------
        # ENERGY MANAGEMENT --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # set initial energy management strategy -> em_boost_use contains where e_motor boost can be applied
        if self.pars_driver["em_strategy"] == "FCFB":
            self.em_boost_use = np.full(trackobj.no_points, True)
        elif self.pars_driver["em_strategy"] in ["LBP", "LS", "NONE"]:
            self.em_boost_use = np.full(trackobj.no_points, False)
        else:
            raise IOError("Unknown energy management strategy!")

        # calculate number of points in front of a braking point without throttle (lac = lift and coast)
        if self.pars_driver["use_lift_coast"]:
            self.no_points_lac = max(int(round(self.pars_driver["lift_coast_dist"] / stepsize)), 1)
        else:
            self.no_points_lac = 0

        # --------------------------------------------------------------------------------------------------------------
        # THROTTLE POSITION --------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initialize array containing the throttle actuation for the consideration of yellow flags. Furthermore, it is
        # used for "lift and coast" consideration later on.
        self.throttle_pos = np.ones(trackobj.no_points)

        # set reduced throttle for yellow flags
        if any((self.pars_driver["yellow_s1"], self.pars_driver["yellow_s2"], self.pars_driver["yellow_s3"])):
            self.__set_yellow_throttle(trackobj=trackobj)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_carobj(self) -> CarHybrid or CarElectric: return self.__carobj
    def __set_carobj(self, x: CarHybrid or CarElectric) -> None: self.__carobj = x
    carobj = property(__get_carobj, __set_carobj)

    def __get_pars_driver(self) -> dict: return self.__pars_driver
    def __set_pars_driver(self, x: dict) -> None: self.__pars_driver = x
    pars_driver = property(__get_pars_driver, __set_pars_driver)

    def __get_em_boost_use(self) -> np.ndarray: return self.__em_boost_use
    def __set_em_boost_use(self, x: np.ndarray) -> None: self.__em_boost_use = x
    em_boost_use = property(__get_em_boost_use, __set_em_boost_use)

    def __get_throttle_pos(self) -> np.ndarray: return self.__throttle_pos
    def __set_throttle_pos(self, x: np.ndarray) -> None: self.__throttle_pos = x
    throttle_pos = property(__get_throttle_pos, __set_throttle_pos)

    def __get_no_points_lac(self) -> int: return self.__no_points_lac
    def __set_no_points_lac(self, x: int) -> None: self.__no_points_lac = x
    no_points_lac = property(__get_no_points_lac, __set_no_points_lac)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset_driver(self, trackobj: Track):
        """Deployed into a function to be able to also reset the driver object during the simulations."""

        if self.pars_driver["em_strategy"] == "FCFB":
            self.em_boost_use = np.full(trackobj.no_points, True)
        elif self.pars_driver["em_strategy"] in ["LBP", "LS", "NONE"]:
            self.em_boost_use = np.full(trackobj.no_points, False)
        else:
            raise IOError("Unknown energy management strategy!")

        self.throttle_pos = np.ones(trackobj.no_points)

        if any((self.pars_driver["yellow_s1"], self.pars_driver["yellow_s2"], self.pars_driver["yellow_s3"])):
            self.__set_yellow_throttle(trackobj=trackobj)

    def __set_yellow_throttle(self, trackobj: Track):
        if self.pars_driver["yellow_s1"]:
            self.throttle_pos[:trackobj.zone_inds["s12"]] = self.pars_driver["yellow_throttle"]

        if self.pars_driver["yellow_s2"]:
            self.throttle_pos[trackobj.zone_inds["s12"]:trackobj.zone_inds["s23"]] = self.pars_driver["yellow_throttle"]

        if self.pars_driver["yellow_s3"]:
            self.throttle_pos[trackobj.zone_inds["s23"]:] = self.pars_driver["yellow_throttle"]

    def calc_em_boost_use(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                          es_final: float):
        if self.pars_driver["em_strategy"] == "LBP":
            self.__strategy_lbp(t_cl=t_cl,
                                vel_cl=vel_cl,
                                n_cl=n_cl,
                                m_requ=m_requ,
                                es_final=es_final)

        elif self.pars_driver["em_strategy"] == "LS":
            self.__strategy_ls(t_cl=t_cl,
                               vel_cl=vel_cl,
                               n_cl=n_cl,
                               m_requ=m_requ,
                               es_final=es_final)

        elif self.pars_driver["em_strategy"] == "FCFB" and self.pars_driver["use_lift_coast"]:
            # set array where throttle is 0.0 when driving in lift and coast condition
            self.__lift_coast(vel_cl=vel_cl,
                              n_lac=self.no_points_lac)

        else:
            raise IOError("EM strategy not considered!")

    def __strategy_lbp(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                       es_final: float):
        """lbp = longest time to (next) brakepoint. The approximation of the ES state during this calcultion should be
        on the conservative side as the recalculated velocity profil will be faster and therefore the times of
        appliance will get shorter."""

        # input check: energy store
        if es_final < 0.0:
            print("WARNING: ES charge state already negative when entering EM strategy calculation!")

        # find indices of brake points
        inds_brake = np.squeeze(np.argwhere(np.diff(vel_cl) < 0.0))

        # calculate time until next brake point for every point (0.0 for brake points themself)
        no_points = t_cl.size - 1  # - 1 to get number of points for unclosed lap
        t_until_brake = np.zeros(no_points)

        for i in range(no_points):
            if i <= inds_brake[-1]:
                ind_brake_rel = inds_brake[np.searchsorted(inds_brake, i)]
                t_until_brake[i] = t_cl[ind_brake_rel] - t_cl[i]
            else:
                ind_brake_rel = inds_brake[0]
                t_until_brake[i] = t_cl[-1] - t_cl[i] + t_cl[ind_brake_rel]

        # sort t_until_brake and get indices (minus sign to sort in a descending order)
        inds_sorted = list(np.argsort(-t_until_brake))

        while es_final > 0.0 and len(inds_sorted) > 0:
            # get current index and remove it from indices list
            ind_cur = inds_sorted.pop(0)

            # check if a brake point would be used (case when too much energy available) -> if so break
            if math.isclose(t_until_brake[ind_cur], 0.0):
                self.em_boost_use = np.full(no_points, True)
                break

            # apply boost if boost was not applied here so far
            if not self.em_boost_use[ind_cur]:
                # apply boost here
                self.em_boost_use[ind_cur] = True

                # calculate torque distribution within the hybrid system
                m_e_motor = self.carobj.calc_torque_distr(n=n_cl[ind_cur],
                                                          m_requ=m_requ[ind_cur],
                                                          throttle_pos=self.throttle_pos[ind_cur],
                                                          es=np.inf,
                                                          em_boost_use=True,
                                                          vel=vel_cl[ind_cur])[1]

                # update energy store status (approximation because velocity profile is influenced obviously)
                es_final -= (self.carobj.power_demand_e_motor_drive(n=n_cl[ind_cur],
                                                                    m_e_motor=np.array(m_e_motor))
                             * (t_cl[ind_cur + 1] - t_cl[ind_cur]))

    def __strategy_ls(self, t_cl: np.ndarray, vel_cl: np.ndarray, n_cl: np.ndarray, m_requ: np.ndarray,
                      es_final: float):
        """ls = lowest speed. The approximation of the ES state during this calcultion should be on the conservative
        side as the recalculated velocity profil will be faster and therefore the times of appliance will get
        shorter."""

        # input check: energy store
        if es_final < 0.0:
            print("WARNING: ES charge state already negative when entering EM strategy calculation!")

        # sort vel and get indices
        inds_sorted = list(np.argsort(vel_cl[:-1]))

        while es_final > 0.0 and len(inds_sorted) > 0:
            # get current index and remove it from indices list
            ind_cur = inds_sorted.pop(0)

            # apply boost if boost was not applied here so far
            if not self.em_boost_use[ind_cur]:
                # apply boost here
                self.em_boost_use[ind_cur] = True

                # calculate torque distribution within the hybrid system
                m_e_motor = self.carobj.calc_torque_distr(n=n_cl[ind_cur],
                                                          m_requ=m_requ[ind_cur],
                                                          throttle_pos=self.throttle_pos[ind_cur],
                                                          es=np.inf,
                                                          em_boost_use=True,
                                                          vel=vel_cl[ind_cur])[1]

                # update energy store status (approximation because velocity profile is influenced obviously)
                es_final -= (self.carobj.power_demand_e_motor_drive(n=n_cl[ind_cur],
                                                                    m_e_motor=np.array(m_e_motor))
                             * (t_cl[ind_cur + 1] - t_cl[ind_cur]))

    def __lift_coast(self, vel_cl: np.ndarray, n_lac: int):
        """Velocity input in m/s, n_lac is the number of points without throttle in front of a brake point."""

        no_points = vel_cl.size - 1
        vel_diffs = np.diff(vel_cl)
        inds_neg_vel_diff = np.squeeze(np.argwhere(vel_diffs < 0.0))

        for i in inds_neg_vel_diff:
            # catch case that lift&coast starts in front of start/finish line
            if i - n_lac < 0:
                self.throttle_pos[no_points + i - n_lac:] = 0.0
                # current point is not included to allow acceleration directly after last brakepoint
                self.throttle_pos[0:i] = 0.0
            else:
                # current point is not included to allow acceleration directly after last brakepoint
                self.throttle_pos[i - n_lac:i] = 0.0


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
