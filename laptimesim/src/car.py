import numpy as np
import math
import matplotlib.pyplot as plt


class Car(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The file provides functions related to the vehicle, e.g. power and torque calculations.
    Vehicle coordinate system: x - front, y - left, z - up.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__powertrain_type",
                 "__pars_general",
                 "__pars_engine",
                 "__pars_gearbox",
                 "__pars_tires",
                 "__f_z_calc_stat")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, powertrain_type: str, pars_general: dict, pars_engine: dict, pars_gearbox: dict,
                 pars_tires: dict):
        self.powertrain_type = powertrain_type
        self.pars_general = pars_general
        self.pars_engine = pars_engine
        self.pars_gearbox = pars_gearbox
        self.pars_tires = pars_tires

        # calculate static parts of tire load calculation
        self.f_z_calc_stat = {}

        g = self.pars_general["g"]
        m = self.pars_general["m"]
        l_tot = self.pars_general["lf"] + self.pars_general["lr"]
        h_cog = self.pars_general["h_cog"]

        # static load
        self.f_z_calc_stat["stat_load"] = np.zeros(4)
        self.f_z_calc_stat["stat_load"][0] = 0.5 * m * g * self.pars_general["lr"] / l_tot
        self.f_z_calc_stat["stat_load"][1] = 0.5 * m * g * self.pars_general["lr"] / l_tot
        self.f_z_calc_stat["stat_load"][2] = 0.5 * m * g * self.pars_general["lf"] / l_tot
        self.f_z_calc_stat["stat_load"][3] = 0.5 * m * g * self.pars_general["lf"] / l_tot

        # longitudinal load transfer
        self.f_z_calc_stat["trans_long"] = np.zeros(4)
        self.f_z_calc_stat["trans_long"][0] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][1] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][2] = 0.5 * m * h_cog / l_tot
        self.f_z_calc_stat["trans_long"][3] = 0.5 * m * h_cog / l_tot

        # lateral load transfer
        self.f_z_calc_stat["trans_lat"] = np.zeros(4)
        self.f_z_calc_stat["trans_lat"][0] = m * self.pars_general["lr"] / l_tot * h_cog / self.pars_general["sf"]
        self.f_z_calc_stat["trans_lat"][1] = m * self.pars_general["lr"] / l_tot * h_cog / self.pars_general["sf"]
        self.f_z_calc_stat["trans_lat"][2] = m * self.pars_general["lf"] / l_tot * h_cog / self.pars_general["sr"]
        self.f_z_calc_stat["trans_lat"][3] = m * self.pars_general["lf"] / l_tot * h_cog / self.pars_general["sr"]

        # aero downforce
        self.f_z_calc_stat["aero"] = np.zeros(4)
        self.f_z_calc_stat["aero"][0] = 0.5 * 0.5 * self.pars_general["c_z_a_f"] * self.pars_general["rho_air"]
        self.f_z_calc_stat["aero"][1] = 0.5 * 0.5 * self.pars_general["c_z_a_f"] * self.pars_general["rho_air"]
        self.f_z_calc_stat["aero"][2] = 0.5 * 0.5 * self.pars_general["c_z_a_r"] * self.pars_general["rho_air"]
        self.f_z_calc_stat["aero"][3] = 0.5 * 0.5 * self.pars_general["c_z_a_r"] * self.pars_general["rho_air"]

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_powertrain_type(self) -> str: return self.__powertrain_type
    def __set_powertrain_type(self, x: str) -> None: self.__powertrain_type = x
    powertrain_type = property(__get_powertrain_type, __set_powertrain_type)

    def __get_pars_general(self) -> dict: return self.__pars_general
    def __set_pars_general(self, x: dict) -> None: self.__pars_general = x
    pars_general = property(__get_pars_general, __set_pars_general)

    def __get_pars_engine(self) -> dict: return self.__pars_engine
    def __set_pars_engine(self, x: dict) -> None: self.__pars_engine = x
    pars_engine = property(__get_pars_engine, __set_pars_engine)

    def __get_pars_gearbox(self) -> dict: return self.__pars_gearbox
    def __set_pars_gearbox(self, x: dict) -> None: self.__pars_gearbox = x
    pars_gearbox = property(__get_pars_gearbox, __set_pars_gearbox)

    def __get_pars_tires(self) -> dict: return self.__pars_tires
    def __set_pars_tires(self, x: dict) -> None: self.__pars_tires = x
    pars_tires = property(__get_pars_tires, __set_pars_tires)

    def __get_f_z_calc_stat(self) -> dict: return self.__f_z_calc_stat
    def __set_f_z_calc_stat(self, x: dict) -> None: self.__f_z_calc_stat = x
    f_z_calc_stat = property(__get_f_z_calc_stat, __set_f_z_calc_stat)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def tire_force_pots(self, vel: float, a_x: float, a_y: float, mu: float) -> tuple:
        """
        The function is used to calculate the transmitted tire forces depending on the current longitudinal and lateral
        accelerations and velocity.
        Velocity input in m/s, accelerations in m/s^2. Calculates the currently acting tire loads f_z (considering
        dynamic load transfers) and the force potentials f_t of all four tires. Vehicle coordinate system: x - front,
        y - left, z - up. The tire model includes the reduction of the force potential with rising tire loads as
        tire_par2 are negativ.
        """

        # tire load calculation: static load, longitudinal load transfer, lateral load transfer, aero downforce
        f_z_fl = (self.f_z_calc_stat["stat_load"][0]
                  - a_x * self.f_z_calc_stat["trans_long"][0]
                  - a_y * self.f_z_calc_stat["trans_lat"][0]
                  + math.pow(vel, 2) * self.f_z_calc_stat["aero"][0])
        f_z_fr = (self.f_z_calc_stat["stat_load"][1]
                  - a_x * self.f_z_calc_stat["trans_long"][1]
                  + a_y * self.f_z_calc_stat["trans_lat"][1]
                  + math.pow(vel, 2) * self.f_z_calc_stat["aero"][1])
        f_z_rl = (self.f_z_calc_stat["stat_load"][2]
                  + a_x * self.f_z_calc_stat["trans_long"][2]
                  - a_y * self.f_z_calc_stat["trans_lat"][2]
                  + math.pow(vel, 2) * self.f_z_calc_stat["aero"][2])
        f_z_rr = (self.f_z_calc_stat["stat_load"][3]
                  + a_x * self.f_z_calc_stat["trans_long"][3]
                  + a_y * self.f_z_calc_stat["trans_lat"][3]
                  + math.pow(vel, 2) * self.f_z_calc_stat["aero"][3])

        # check tire loads
        """
        Commented since it often happens with the FB+ solver that very high lateral accelerations and therefore tire
        loads appear when it runs into a corner, i.e. a high curvature. As the tires limit the lateral acceleration due
        to their limited force potential afterwards this is usually not a problem. Use the lateral acceleration plot
        and the tire loads plot at the end of the calculation to check the validity of the lateral accelerations
        appearing.
        """

        if f_z_fl < 30.0:
            # print("WARNING: Very small tire load FL!")
            f_z_fl = 30.0
        if f_z_fr < 30.0:
            # print("WARNING: Very small tire load FR!")
            f_z_fr = 30.0
        if f_z_rl < 30.0:
            # print("WARNING: Very small tire load RL!")
            f_z_rl = 30.0
        if f_z_rr < 30.0:
            # print("WARNING: Very small tire load RR!")
            f_z_rr = 30.0

        # tire force potentials (dmux_dfz and dmuy_dfz are negativ -> quadratic malus in the force)
        """
        The function is derived as follows:
        F_x = mu_weather/track * mu_tire(F_z) * F_z (mu_tire hereby not constant as it decreases with rising tire loads)
            = mu_weather/track * (mu_tire + dmu_tire/dF_z * (F_z - F_z0)) * F_z (dmu_tire/dF_z is negative)
        """
        f_x_pot_fl = mu * (self.pars_tires["f"]["mux"]
                           + self.pars_tires["f"]["dmux_dfz"] * (f_z_fl - self.pars_tires["f"]["fz_0"])) * f_z_fl
        f_y_pot_fl = mu * (self.pars_tires["f"]["muy"]
                           + self.pars_tires["f"]["dmuy_dfz"] * (f_z_fl - self.pars_tires["f"]["fz_0"])) * f_z_fl

        f_x_pot_fr = mu * (self.pars_tires["f"]["mux"]
                           + self.pars_tires["f"]["dmux_dfz"] * (f_z_fr - self.pars_tires["f"]["fz_0"])) * f_z_fr
        f_y_pot_fr = mu * (self.pars_tires["f"]["muy"]
                           + self.pars_tires["f"]["dmuy_dfz"] * (f_z_fr - self.pars_tires["f"]["fz_0"])) * f_z_fr

        f_x_pot_rl = mu * (self.pars_tires["r"]["mux"]
                           + self.pars_tires["r"]["dmux_dfz"] * (f_z_rl - self.pars_tires["r"]["fz_0"])) * f_z_rl
        f_y_pot_rl = mu * (self.pars_tires["r"]["muy"]
                           + self.pars_tires["r"]["dmuy_dfz"] * (f_z_rl - self.pars_tires["r"]["fz_0"])) * f_z_rl

        f_x_pot_rr = mu * (self.pars_tires["r"]["mux"]
                           + self.pars_tires["r"]["dmux_dfz"] * (f_z_rr - self.pars_tires["r"]["fz_0"])) * f_z_rr
        f_y_pot_rr = mu * (self.pars_tires["r"]["muy"]
                           + self.pars_tires["r"]["dmuy_dfz"] * (f_z_rr - self.pars_tires["r"]["fz_0"])) * f_z_rr

        return (f_x_pot_fl, f_y_pot_fl, f_z_fl,
                f_x_pot_fr, f_y_pot_fr, f_z_fr,
                f_x_pot_rl, f_y_pot_rl, f_z_rl,
                f_x_pot_rr, f_y_pot_rr, f_z_rr)

    def plot_tire_characteristics(self) -> None:
        # calculate relevant data
        f_z_range = np.arange(500.0, 13000.0, 500.0)

        f_x_f = (self.pars_tires["f"]["mux"]
                 + self.pars_tires["f"]["dmux_dfz"] * (f_z_range - self.pars_tires["f"]["fz_0"])) * f_z_range
        f_y_f = (self.pars_tires["f"]["muy"]
                 + self.pars_tires["f"]["dmuy_dfz"] * (f_z_range - self.pars_tires["f"]["fz_0"])) * f_z_range
        f_x_r = (self.pars_tires["r"]["mux"]
                 + self.pars_tires["r"]["dmux_dfz"] * (f_z_range - self.pars_tires["r"]["fz_0"])) * f_z_range
        f_y_r = (self.pars_tires["r"]["muy"]
                 + self.pars_tires["r"]["dmuy_dfz"] * (f_z_range - self.pars_tires["r"]["fz_0"])) * f_z_range

        # plot
        plt.figure()

        plt.plot(f_z_range, f_x_f)
        plt.plot(f_z_range, f_y_f)
        plt.plot(f_z_range, f_x_r)
        plt.plot(f_z_range, f_y_r)

        plt.grid()
        plt.title("Tire force characteristics")
        plt.xlabel("F_z in N")
        plt.ylabel("Forces F_x and F_y in N")
        plt.legend(["F_x front", "F_y front", "F_x rear", "F_y rear"])

        plt.show()

    def __circumref_driven_tire(self, vel: float) -> float:
        """Velocity input in m/s. Reference speed for the circumreference calculation is 60 km/h. Output is in m."""

        if self.pars_engine["topology"] == "FWD":
            tire_circ_ref = self.pars_tires["f"]["circ_ref"]

        elif self.pars_engine["topology"] == "RWD":
            tire_circ_ref = self.pars_tires["r"]["circ_ref"]

        elif self.pars_engine["topology"] == "AWD":
            # use average circumreference in this case
            tire_circ_ref = 0.5 * (self.pars_tires["f"]["circ_ref"] + self.pars_tires["r"]["circ_ref"])

        else:
            raise RuntimeError("Powertrain topology unknown!")

        return tire_circ_ref * (1 + (vel * 3.6 - 60.0) * (0.045 / 200.0))

    def r_driven_tire(self, vel: float) -> float:
        """Velocity input in m/s. Output is in m."""

        return self.__circumref_driven_tire(vel=vel) / (2 * math.pi)

    def air_res(self, vel: float, drs: bool) -> float:
        """Velocity input in m/s. Output is in N."""

        # get relevant data
        rho_air = self.pars_general["rho_air"]
        c_w_a = self.pars_general["c_w_a"]

        if drs:
            return 0.5 * (1.0 - self.pars_general["drs_factor"]) * c_w_a * rho_air * math.pow(vel, 2)
        else:
            return 0.5 * c_w_a * rho_air * math.pow(vel, 2)

    def roll_res(self, f_z_tot: float) -> float:
        """Output is in N."""
        return f_z_tot * self.pars_general["f_roll"]

    def calc_lat_forces(self, a_y: float) -> tuple:
        """Lateral acceleration input in m/s^2. Output forces in N."""

        f_y = self.pars_general["m"] * a_y
        f_y_f = f_y * self.pars_general["lr"] / (self.pars_general["lf"] + self.pars_general["lr"])
        f_y_r = f_y * self.pars_general["lf"] / (self.pars_general["lf"] + self.pars_general["lr"])

        return f_y_f, f_y_r

    def v_max_cornering(self, kappa: float, mu: float, vel_subtr_corner: float = 0.5) -> float:
        """
        Curvature input in rad/m, vel_subtr_corner in m/s. This method determines the maximum drivable velocity for the
        pure cornering case, i.e. without the application of longitudinal acceleration. However, it is considered that
        the tires must be able to transmit enough longitudinal forces to overcome drag and rolling resistances. The
        calculation neglects the available force in the powertrain. The determined velocity is mostly used as velocity
        after deceleration phases. Using binary search technique to decrease calculation time. vel_subtr_corner is
        subtracted from the found cornering velocity because drivers in reality will not hit the maximum perfectly.
        """

        # user input
        no_steps = 546          # [-] number of steps is currently chosen such that the stepsize is 0.2 m/s
        vel_max = 110.0         # [m/s] cover speed range up to 400 km/h

        # create velocity array
        vel_range = np.linspace(1.0, vel_max, no_steps)

        # binary search for maximum velocity
        ind_first = 0
        ind_last = vel_range.size - 1
        ind_mid = math.ceil((ind_first + ind_last) / 2)

        while ind_first != ind_last:
            # calculate currently acting lateral acceleration and forces
            a_y = math.pow(vel_range[ind_mid], 2) * kappa
            f_y_f, f_y_r = self.calc_lat_forces(a_y=a_y)

            # calculate tire force potentials (using a_x = 0.0 at maximum cornering)
            f_x_pot_fl, f_y_pot_fl, f_z_fl, \
                f_x_pot_fr, f_y_pot_fr, f_z_fr, \
                f_x_pot_rl, f_y_pot_rl, f_z_rl, \
                f_x_pot_rr, f_y_pot_rr, f_z_rr = self.tire_force_pots(vel=vel_range[ind_mid],
                                                                      a_x=0.0,
                                                                      a_y=a_y,
                                                                      mu=mu)

            # calculate remaining tire potential at the driven axle(s) for longitudinal force
            """Axis-wise consideration of the lateral force potential makes sense because it is better to assume that
            the outer tire gets as worse as the inner tire gets better than to assume that they have to transfer the
            lateral forces according to the wheel load distribution. This would lead to an underestimation of the
            maximum possible cornering speed. A more exact model is not possible without considering slip angles, i.e.
            a kinematic vehicle model."""

            # check if potential is left overall and if f_x_poss is enough to overcome drag and rolling resistances
            if math.fabs(f_y_f) < f_y_pot_fl + f_y_pot_fr and math.fabs(f_y_r) < f_y_pot_rl + f_y_pot_rr:
                f_x_poss = self.calc_f_x_pot(f_x_pot_fl=f_x_pot_fl,
                                             f_x_pot_fr=f_x_pot_fr,
                                             f_x_pot_rl=f_x_pot_rl,
                                             f_x_pot_rr=f_x_pot_rr,
                                             f_y_pot_f=f_y_pot_fl + f_y_pot_fr,
                                             f_y_pot_r=f_y_pot_rl + f_y_pot_rr,
                                             f_y_f=f_y_f,
                                             f_y_r=f_y_r,
                                             force_use_all_wheels=False,
                                             limit_braking_weak_side=None)

                f_x_drag = (self.air_res(vel=vel_range[ind_mid], drs=False)
                            + self.roll_res(f_z_tot=f_z_fl + f_z_fr + f_z_rl + f_z_rr))

                if f_x_poss < f_x_drag:
                    potential_exceeded = True
                else:
                    potential_exceeded = False

            else:
                potential_exceeded = True

            # check if we are above or below force potential and set indices of velocity array accordingly
            if not potential_exceeded:
                # case: potential is left
                ind_first = ind_mid
            else:
                # case: potential is exceeded
                ind_last = ind_mid - 1

            # update middle index
            ind_mid = math.ceil((ind_first + ind_last) / 2)

        return vel_range[ind_mid] - vel_subtr_corner

    def calc_f_x_pot(self,
                     f_x_pot_fl: float,
                     f_x_pot_fr: float,
                     f_x_pot_rl: float,
                     f_x_pot_rr: float,
                     f_y_pot_f: float,
                     f_y_pot_r: float,
                     f_y_f: float,
                     f_y_r: float,
                     force_use_all_wheels: bool = False,
                     limit_braking_weak_side: None or str = None) -> float:
        """Calculate remaining tire potential for longitudinal force transmission considering driven axle(s). All forces
        in N. 'force_use_all_wheels' flag can be set to use this function also for braking with all four wheels.
        limit_braking_weak_side can be None, 'FA', 'RA', 'all'. This determines if the possible braking force should be
        determined based on the weak side, e.g. when braking into a corner. Can be set separately for both axles. This
        is not necessary during acceleration since a limited slip differential overcomes this problem."""

        exp_tmp = self.pars_tires["tire_model_exp"]

        # check input
        if limit_braking_weak_side is not None and not force_use_all_wheels:
            print("WARNING: It seems like the function is used for braking (because limit_braking_weak_side is set)"
                  " but force_use_all_wheels is not set True!")

        # determine axle potentials
        if limit_braking_weak_side is not None:
            if limit_braking_weak_side == 'FA':
                f_x_pot_f = 2 * min(f_x_pot_fl, f_x_pot_fr)
                f_x_pot_r = f_x_pot_rl + f_x_pot_rr
            elif limit_braking_weak_side == 'RA':
                f_x_pot_f = f_x_pot_fl + f_x_pot_fr
                f_x_pot_r = 2 * min(f_x_pot_rl, f_x_pot_rr)
            elif limit_braking_weak_side == 'all':
                f_x_pot_f = 2 * min(f_x_pot_fl, f_x_pot_fr)
                f_x_pot_r = 2 * min(f_x_pot_rl, f_x_pot_rr)
            else:
                raise RuntimeError("Unknown option %s!" % limit_braking_weak_side)
        else:
            f_x_pot_f = f_x_pot_fl + f_x_pot_fr
            f_x_pot_r = f_x_pot_rl + f_x_pot_rr

        # calculate radicands of the tire model and check if below zero (absolute values of lateral forces required)
        radicand_f = 1 - math.pow(math.fabs(f_y_f) / f_y_pot_f, exp_tmp)
        radicand_r = 1 - math.pow(math.fabs(f_y_r) / f_y_pot_r, exp_tmp)

        radicand_f = max(radicand_f, 0.0)
        radicand_r = max(radicand_r, 0.0)

        # calculate remaining force potential
        if self.pars_engine["topology"] == "AWD" or force_use_all_wheels:
            f_x_poss_f = f_x_pot_f * math.pow(radicand_f, 1.0 / exp_tmp)
            f_x_poss_r = f_x_pot_r * math.pow(radicand_r, 1.0 / exp_tmp)
        elif self.pars_engine["topology"] == "FWD":
            f_x_poss_f = f_x_pot_f * math.pow(radicand_f, 1.0 / exp_tmp)
            f_x_poss_r = 0.0
        elif self.pars_engine["topology"] == "RWD":
            f_x_poss_f = 0.0
            f_x_poss_r = f_x_pot_r * math.pow(radicand_r, 1.0 / exp_tmp)
        else:
            raise RuntimeError("Powertrain topology unknown!")

        return f_x_poss_f + f_x_poss_r

    def calc_max_ax(self, vel: float, a_y: float, mu: float, f_y_f: float, f_y_r: float) -> float:
        """Calculate maximum longitudinal acceleration at which the car stays on the track. vel in m/s, a_y in m/s^2,
        f_y_f and f_y_r in N. Using binary search technique to decrease calculation time."""

        # user input
        no_steps = 101
        a_x_max = 25.0  # [m/s^2]

        # create a_x array
        a_x_range = np.linspace(0.0, a_x_max, no_steps)

        # binary search
        ind_first = 0
        ind_last = a_x_range.size - 1
        ind_mid = math.ceil((ind_first + ind_last) / 2)

        while ind_first != ind_last:
            # calculate tire potentials at ind_mid
            _, f_y_pot_fl, _, \
                _, f_y_pot_fr, _, \
                _, f_y_pot_rl, _, \
                _, f_y_pot_rr, _ = self.tire_force_pots(vel=vel,
                                                        a_x=a_x_range[ind_mid],
                                                        a_y=a_y,
                                                        mu=mu)

            # check if we are above or below force potential
            if math.fabs(f_y_f) <= f_y_pot_fl + f_y_pot_fr and math.fabs(f_y_r) <= f_y_pot_rl + f_y_pot_rr:
                # case: potential is left
                ind_first = ind_mid
            else:
                # case: potential is exceeded
                ind_last = ind_mid - 1

            # update middle index
            ind_mid = math.ceil((ind_first + ind_last) / 2)

        return a_x_range[ind_mid]

    def find_gear(self, vel: float) -> tuple:
        """Velocity input in m/s. Output is the gear used for that velocity (zero based) as well as the corresponding
        engine rev in 1/s."""

        # calculate theoretical engine revs for all the gears
        n_gears = vel / (self.__circumref_driven_tire(vel=vel) * self.pars_gearbox["i_trans"])  # [1/s]

        # find largest gear below shift revs
        shift_bool = n_gears < self.pars_gearbox["n_shift"]

        if np.all(~shift_bool):
            # if max rev in final gear is reached do not shift up
            gear_ind = self.pars_gearbox["n_shift"].size - 1  # -1 due to zero based indexing
        else:
            # find first True value (zero based indexing of gears)
            gear_ind = int(np.argmax(shift_bool))

        return gear_ind, n_gears[gear_ind]

    def calc_m_requ(self, f_x: float, vel: float) -> float:
        """Function to calculate required powertrain torque to reach a specific longitudinal acceleration force f_x at
        the current velocity. Input f_x in N, vel in m/s. Output is the rquired powertrain torque in Nm."""

        # get gear at velocity
        gear = self.find_gear(vel=vel)[0]

        # calculate powertrain torque
        m_requ = (f_x * self.r_driven_tire(vel=vel) * self.pars_gearbox["i_trans"][gear]
                  * self.pars_gearbox["e_i"][gear] / self.pars_gearbox["eta_g"])

        return m_requ


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
