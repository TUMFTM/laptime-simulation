import numpy as np
import math
import matplotlib.pyplot as plt
from laptimesim.src.track import Track
from laptimesim.src.driver import Driver


class Lap(object):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The function provides the solver required to calculate the velocity profile and various other data
    step by step. The solver is based on a moving point mass instead of a kinematic model. However, the longitudinal
    and lateral load transfers are considered for the (steady-state) tire load calculations of every wheel. The tire
    model models the effect of reduced force potential with rising tire loads. The solver does not calculate a proper
    start velocity. This has to be ensured by re-running the solver a second time.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = ("__driverobj",
                 "__trackobj",
                 "__t_cl",
                 "__vel_cl",
                 "__n_cl",
                 "__m_eng",
                 "__m_e_motor",
                 "__m_requ",
                 "__es_cl",
                 "__gear_cl",
                 "__e_rec_e_motor",
                 "__a_x_final",
                 "__e_rec_e_motor_max",
                 "__pars_solver",
                 "__debug_opts",
                 "__fuel_cons_cl",
                 "__e_cons_cl",
                 "__tire_loads",
                 "__e_es_to_e_motor_max",
                 "__e_motor_power")

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, driverobj: Driver, trackobj: Track, pars_solver: dict, debug_opts: dict):

        # save driver and track objects
        self.driverobj = driverobj
        self.trackobj = trackobj

        # save solver parameters and debug options
        self.pars_solver = pars_solver
        self.debug_opts = debug_opts

        # adjust solver parameters
        if self.trackobj.pars_track["use_pit"]:
            # v_start is defined by pit speed limit
            self.pars_solver["v_start"] = self.trackobj.pars_track["pitspeed"]
            self.pars_solver["find_v_start"] = False

        # initialize lap variables
        self.t_cl = np.zeros(trackobj.no_points_cl)     # [s] lap time at the beginning of a step
        self.vel_cl = np.zeros(trackobj.no_points_cl)   # [m/s] velocity at the beginning of a step
        self.n_cl = np.zeros(trackobj.no_points_cl)     # [1/s] rev at the beginning of a step
        self.m_eng = np.zeros(trackobj.no_points)       # [Nm] used engine torque during current step, positive values
        self.m_e_motor = np.zeros(trackobj.no_points)   # [Nm] used e motor torque during current step, positive values
        self.m_requ = np.zeros(trackobj.no_points)      # [Nm] requested torque during current step, positive values
        self.es_cl = np.zeros(trackobj.no_points_cl)    # [J] energy storage state at the beginning of a step
        self.gear_cl = np.zeros(trackobj.no_points_cl, dtype=int)  # [-] gear during current step, zero based in solver
        # [kJ] energy recuperated during the current step available at the beginning of next step
        self.e_rec_e_motor = np.zeros(trackobj.no_points)
        self.a_x_final = 0.0
        self.fuel_cons_cl = np.zeros(trackobj.no_points_cl)     # [kg] consumed fuel mass until current point
        self.e_cons_cl = np.zeros(trackobj.no_points_cl)        # [J] consumed energy until current point
        self.tire_loads = np.zeros((trackobj.no_points, 4))     # [N] tire loads [FL, FR, RL, RR]

        # elemons add ons:
        self.e_motor_power = np.zeros(trackobj.no_points)

        # [J/lap] maximum amount of energy allowed to recuperate in e motor
        self.e_rec_e_motor_max = np.inf
        self.e_es_to_e_motor_max = np.inf

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_driverobj(self) -> Driver: return self.__driverobj
    def __set_driverobj(self, x: Driver) -> None: self.__driverobj = x
    driverobj = property(__get_driverobj, __set_driverobj)

    def __get_trackobj(self) -> Track: return self.__trackobj
    def __set_trackobj(self, x: Track) -> None: self.__trackobj = x
    trackobj = property(__get_trackobj, __set_trackobj)

    def __get_t_cl(self) -> np.ndarray: return self.__t_cl
    def __set_t_cl(self, x: np.ndarray) -> None: self.__t_cl = x
    t_cl = property(__get_t_cl, __set_t_cl)

    def __get_vel_cl(self) -> np.ndarray: return self.__vel_cl
    def __set_vel_cl(self, x: np.ndarray) -> None: self.__vel_cl = x
    vel_cl = property(__get_vel_cl, __set_vel_cl)

    def __get_n_cl(self) -> np.ndarray: return self.__n_cl
    def __set_n_cl(self, x: np.ndarray) -> None: self.__n_cl = x
    n_cl = property(__get_n_cl, __set_n_cl)

    def __get_m_eng(self) -> np.ndarray: return self.__m_eng
    def __set_m_eng(self, x: np.ndarray) -> None: self.__m_eng = x
    m_eng = property(__get_m_eng, __set_m_eng)

    def __get_m_e_motor(self) -> np.ndarray: return self.__m_e_motor
    def __set_m_e_motor(self, x: np.ndarray) -> None: self.__m_e_motor = x
    m_e_motor = property(__get_m_e_motor, __set_m_e_motor)

    def __get_m_requ(self) -> np.ndarray: return self.__m_requ
    def __set_m_requ(self, x: np.ndarray) -> None: self.__m_requ = x
    m_requ = property(__get_m_requ, __set_m_requ)

    def __get_es_cl(self) -> np.ndarray: return self.__es_cl
    def __set_es_cl(self, x: np.ndarray) -> None: self.__es_cl = x
    es_cl = property(__get_es_cl, __set_es_cl)

    def __get_gear_cl(self) -> np.ndarray: return self.__gear_cl
    def __set_gear_cl(self, x: np.ndarray) -> None: self.__gear_cl = x
    gear_cl = property(__get_gear_cl, __set_gear_cl)

    def __get_e_rec_e_motor(self) -> np.ndarray: return self.__e_rec_e_motor
    def __set_e_rec_e_motor(self, x: np.ndarray) -> None: self.__e_rec_e_motor = x
    e_rec_e_motor = property(__get_e_rec_e_motor, __set_e_rec_e_motor)

    def __get_a_x_final(self) -> float: return self.__a_x_final
    def __set_a_x_final(self, x: float) -> None: self.__a_x_final = x
    a_x_final = property(__get_a_x_final, __set_a_x_final)

    def __get_e_rec_e_motor_max(self) -> float: return self.__e_rec_e_motor_max
    def __set_e_rec_e_motor_max(self, x: float) -> None: self.__e_rec_e_motor_max = x
    e_rec_e_motor_max = property(__get_e_rec_e_motor_max, __set_e_rec_e_motor_max)

    def __get_pars_solver(self) -> dict: return self.__pars_solver
    def __set_pars_solver(self, x: dict) -> None: self.__pars_solver = x
    pars_solver = property(__get_pars_solver, __set_pars_solver)

    def __get_debug_opts(self) -> dict: return self.__debug_opts
    def __set_debug_opts(self, x: dict) -> None: self.__debug_opts = x
    debug_opts = property(__get_debug_opts, __set_debug_opts)

    def __get_fuel_cons_cl(self) -> np.ndarray: return self.__fuel_cons_cl
    def __set_fuel_cons_cl(self, x: np.ndarray) -> None: self.__fuel_cons_cl = x
    fuel_cons_cl = property(__get_fuel_cons_cl, __set_fuel_cons_cl)

    def __get_e_cons_cl(self) -> np.ndarray: return self.__e_cons_cl
    def __set_e_cons_cl(self, x: np.ndarray) -> None: self.__e_cons_cl = x
    e_cons_cl = property(__get_e_cons_cl, __set_e_cons_cl)

    def __get_tire_loads(self) -> np.ndarray: return self.__tire_loads
    def __set_tire_loads(self, x: np.ndarray) -> None: self.__tire_loads = x
    tire_loads = property(__get_tire_loads, __set_tire_loads)

    def __get_e_es_to_e_motor_max(self) -> float: return self.__e_es_to_e_motor_max
    def __set_e_es_to_e_motor_max(self, x: float) -> None: self.__e_es_to_e_motor_max = x
    e_es_to_e_motor_max = property(__get_e_es_to_e_motor_max, __set_e_es_to_e_motor_max)

    def __get_e_motor_power(self) -> np.ndarray: return self.__e_motor_power
    def __set_e_motor_power(self, x: np.ndarray) -> None: self.__e_motor_power = x
    e_motor_power = property(__get_e_motor_power, __set_e_motor_power)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def reset_lap(self):
        # reset driver object
        self.driverobj.reset_driver(trackobj=self.trackobj)

        # reset lap variables
        self.t_cl = np.zeros(self.trackobj.no_points_cl)
        self.vel_cl = np.zeros(self.trackobj.no_points_cl)
        self.n_cl = np.zeros(self.trackobj.no_points_cl)
        self.m_eng = np.zeros(self.trackobj.no_points)
        self.m_e_motor = np.zeros(self.trackobj.no_points)
        self.m_requ = np.zeros(self.trackobj.no_points)
        self.es_cl = np.zeros(self.trackobj.no_points_cl)
        self.gear_cl = np.zeros(self.trackobj.no_points_cl, dtype=int)
        self.e_rec_e_motor = np.zeros(self.trackobj.no_points)
        self.a_x_final = 0.0
        self.fuel_cons_cl = np.zeros(self.trackobj.no_points_cl)
        self.e_cons_cl = np.zeros(self.trackobj.no_points_cl)
        self.tire_loads = np.zeros((self.trackobj.no_points, 4))

        self.e_rec_e_motor_max = np.inf

        # elemons add ons:
        self.e_motor_power = np.zeros(self.trackobj.no_points)

    def simulate_lap(self):
        """
        Main method used to simulate a lap. It includes the required solver calls and hybrid system calculations.

        Solver calls appear in the following order (EM = LBP/LS):
        1) Initial call without EM
        2) Second call without EM starting with the proper start velocity based on previous result
        3) Loop -> at first the EM strategy is calculated based on remaining energy at end of lap from previous result,
            then solver runs. This is done until either the maximum number of iterations is reached or no more changes
            in the energy storage state appear. In the first runs we never use all the energy due to two reasons:
            1) The boost will increase the velocity and thereby reduce the time consumed in one track segment and
            therefore reduce the amount of energy consumed there. 2) Additional recuperated energy due to the increased
            velocity. This additional energy remains at the end of the lap and can be used to recalculate the EM boost
            points for the next run. We only add new boost points and do net remove old ones to avoid endless loops (we
            do not know how much energy is available for EM strategy otherwise).
            This process helps to get a good approximation without too many loop iterations, i.e. calculation time.

        Solver calls appear in the following order (EM = FCFB):
        1) Initial call with EM
        2) Second call with EM starting with the proper start velocity based on previous result
        3) If lift&coast is used: Based on the previous velocity profile, the brake points are determined. Virtual
            accelerator pedal is set to zero in front of those points. lift&coast points are determined only once
            because brake points do not change depending on the velocity profile (recalculation would lead to premature
            brake points and therefore to a new acceleration phase in between). Furthermore, this mode is meant to just
            calculate the lap time loss compared to the energy savings if enough energy is available. Otherwise it
            will of course improve the lap time instead to decrease it.
        """

        # --------------------------------------------------------------------------------------------------------------
        # CALL SOLVER (WITHOUT EM) -------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # initial solver run
        if self.debug_opts["use_print"]:
            print("-" * 50)
            print("Starting solver run (1)")

        self.__fbplus(v_start=self.pars_solver["v_start"],
                      a_x_start=0.0)

        # use previous result to rerun the simulation with proper start velocity (if desired by user)
        if self.pars_solver["find_v_start"]:
            if self.debug_opts["use_print"]:
                print("Starting solver run (2) (considering new start velocity)")

            self.__fbplus(v_start=self.vel_cl[-1],
                          a_x_start=self.a_x_final)

        # --------------------------------------------------------------------------------------------------------------
        # CALL SOLVER (WITH EM) ----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # recalculation only required if EM strategy is not "pure FCFB"
        if self.driverobj.pars_driver["em_strategy"] in ["LBP", "LS"]:
            """Due to the mutual influence between velocity profile and EM strategy we need some iterations until an
            equilibrium was found."""

            # create required loop variables
            i = 0
            es_prev = 0.0

            while math.fabs(self.es_cl[-1] - es_prev) > self.pars_solver["es_diff_max"] \
                    and i < self.pars_solver["max_no_em_iters"]:

                i += 1
                es_prev = self.es_cl[-1]

                if self.debug_opts["use_print"]:
                    print("Starting recalculation considering hybrid system (%i)" % i)

                # calculate hybrid system application (boost) based on the previous result
                self.driverobj.calc_em_boost_use(t_cl=self.t_cl,
                                                 vel_cl=self.vel_cl,
                                                 n_cl=self.n_cl,
                                                 m_requ=self.m_requ,
                                                 es_final=self.es_cl[-1])

                # rerun solver to get new velocity profile
                if self.pars_solver["find_v_start"]:
                    self.__fbplus(v_start=self.vel_cl[-1],
                                  a_x_start=self.a_x_final)
                else:
                    self.__fbplus(v_start=self.pars_solver["v_start"],
                                  a_x_start=0.0)

                if self.debug_opts["use_print"]:
                    print("Remaining energy in ES: %.0f kJ" % (self.es_cl[-1] / 1000.0))

        elif self.driverobj.pars_driver["em_strategy"] == "FCFB" and self.driverobj.pars_driver["use_lift_coast"]:
            """Case FCFB + lift&coast."""

            if self.debug_opts["use_print"]:
                print("Starting recalculation considering hybrid system (1)")

            # calculate hybrid system application (boost) based on the previous result
            self.driverobj.calc_em_boost_use(t_cl=self.t_cl,
                                             vel_cl=self.vel_cl,
                                             n_cl=self.n_cl,
                                             m_requ=self.m_requ,
                                             es_final=self.es_cl[-1])

            # rerun solver to get new velocity profile
            if self.pars_solver["find_v_start"]:
                self.__fbplus(v_start=self.vel_cl[-1],
                              a_x_start=self.a_x_final)
            else:
                self.__fbplus(v_start=self.pars_solver["v_start"],
                              a_x_start=0.0)

        if self.debug_opts["use_print"]:
            print("Finished solver calculations")

        # --------------------------------------------------------------------------------------------------------------
        # FUEL/ENERGY CONSUMPTION --------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        if not self.driverobj.carobj.powertrain_type == "electric":
            self.fuel_cons_cl = self.driverobj.carobj.fuel_cons(t_cl=self.t_cl,
                                                                n_cl=self.n_cl,
                                                                m_eng=self.m_eng)

        self.e_cons_cl = self.driverobj.carobj.e_cons(t_cl=self.t_cl,
                                                      n_cl=self.n_cl,
                                                      m_e_motor=self.m_e_motor)

    def __fbplus(self, v_start: float, a_x_start: float = 0.0):
        """
        Returned arrays t_cl, vel_cl, n_cl, es_cl and gear_cl are closed, the rest is unclosed.
        """

        # --------------------------------------------------------------------------------------------------------------
        # USER INPUT ---------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # loop options
        tol = 1e-2          # [m/s] termination criterion (must be greater than force_conv)
        # force_conv = 5e-3   # [m/s] velocity malus per iteration to force convergence (should be about half of 'tol')

        # --------------------------------------------------------------------------------------------------------------
        # INITIALIZE VARIABLES -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        vel_lim_cl = np.append(self.trackobj.vel_lim, self.trackobj.vel_lim[0])
        self.e_rec_e_motor[:] = 0.0  # must be reset for every run

        # --------------------------------------------------------------------------------------------------------------
        # SET START CONDITIONS -----------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        self.vel_cl[0] = v_start
        self.es_cl[0] = self.driverobj.pars_driver["initial_energy"]

        # find gear at start
        self.gear_cl[0], self.n_cl[0] = self.driverobj.carobj.find_gear(vel=self.vel_cl[0])

        # --------------------------------------------------------------------------------------------------------------
        # LOOP THROUGH ALL THE POINTS ----------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        i = 0
        a_x = a_x_start

        while i < self.trackobj.no_points:
            # calculate currently acting lateral acceleration and forces
            a_y = math.pow(self.vel_cl[i], 2) * self.trackobj.kappa[i]
            f_y_f, f_y_r = self.driverobj.carobj.calc_lat_forces(a_y=a_y)

            # calculate tire force potentials (using a_x = 0.0 (maximum cornering) to find out if we can stay on track)
            f_x_pot_fl, f_y_pot_fl, self.tire_loads[i, 0], \
                f_x_pot_fr, f_y_pot_fr, self.tire_loads[i, 1], \
                f_x_pot_rl, f_y_pot_rl, self.tire_loads[i, 2], \
                f_x_pot_rr, f_y_pot_rr, self.tire_loads[i, 3] = self.driverobj.carobj.\
                tire_force_pots(vel=self.vel_cl[i],
                                a_x=0.0,
                                a_y=a_y,
                                mu=self.trackobj.mu[i])

            # ----------------------------------------------------------------------------------------------------------
            # CASE 1: some tire potential is left and no velocity limit is set -> accelerate ---------------------------
            # ----------------------------------------------------------------------------------------------------------

            """Axis-wise consideration of the lateral force potential makes sense because it is better to assume that
            the outer tire gets as worse as the inner tire gets better than to assume that they have to transfer the
            lateral forces according to the wheel load distribution. This would lead to an underestimation of the
            maximum possible cornering speed. A more exact model is not possible without considering slip angles, i.e.
            a kinematic vehicle model."""

            if math.fabs(f_y_f) <= f_y_pot_fl + f_y_pot_fr and math.fabs(f_y_r) <= f_y_pot_rl + f_y_pot_rr \
                    and self.vel_cl[i] <= self.trackobj.vel_lim[i]:
                """Since it is obviously possible to stay on track assuming no longitudinal acceleration, we need to
                get a proper assumption of the maximum longitudinal acceleration a_x that can be handled without
                leaving the track (the connection originates in the a_x influence to the wheel loads)."""

                # obtain maximum longitudinal acceleration
                a_x_max = self.driverobj.carobj.calc_max_ax(vel=self.vel_cl[i],
                                                            a_y=a_y,
                                                            mu=self.trackobj.mu[i],
                                                            f_y_f=f_y_f,
                                                            f_y_r=f_y_r)

                # approximate current a_x for tire load calc. either based on previous iteration or based on a_x_max
                if a_x > 0.0:
                    a_x = min(a_x, a_x_max)
                else:  # a_x not usable if previous point was in deceleration phase
                    a_x = a_x_max

                # recalculate tire force potentials based on approximated a_x
                f_x_pot_fl, f_y_pot_fl, self.tire_loads[i, 0], \
                    f_x_pot_fr, f_y_pot_fr, self.tire_loads[i, 1], \
                    f_x_pot_rl, f_y_pot_rl, self.tire_loads[i, 2], \
                    f_x_pot_rr, f_y_pot_rr, self.tire_loads[i, 3] = self.driverobj.carobj.\
                    tire_force_pots(vel=self.vel_cl[i],
                                    a_x=a_x,
                                    a_y=a_y,
                                    mu=self.trackobj.mu[i])

                # calculate remaining tire potential at front and rear axle for longitudinal force transmission
                f_x_poss = self.driverobj.carobj.calc_f_x_pot(f_x_pot_fl=f_x_pot_fl,
                                                              f_x_pot_fr=f_x_pot_fr,
                                                              f_x_pot_rl=f_x_pot_rl,
                                                              f_x_pot_rr=f_x_pot_rr,
                                                              f_y_pot_f=f_y_pot_fl + f_y_pot_fr,
                                                              f_y_pot_r=f_y_pot_rl + f_y_pot_rr,
                                                              f_y_f=f_y_f,
                                                              f_y_r=f_y_r,
                                                              force_use_all_wheels=False,
                                                              limit_braking_weak_side=None)

                # calculate torque distribution within the hybrid system (trying to reach the possible force f_x)
                self.m_requ[i], self.m_eng[i], self.m_e_motor[i] = self.driverobj.carobj.\
                    calc_torque_distr_f_x(f_x=f_x_poss,
                                          n=self.n_cl[i],
                                          throttle_pos=self.driverobj.throttle_pos[i],
                                          es=self.es_cl[i],
                                          em_boost_use=self.driverobj.em_boost_use[i],
                                          vel=self.vel_cl[i])

                # calculate available acceleration force in powertrain
                f_x_powert = (self.driverobj.carobj.pars_gearbox["eta_g"] * (self.m_eng[i] + self.m_e_motor[i])
                              / (self.driverobj.carobj.pars_gearbox["i_trans"][self.gear_cl[i]]
                                 * self.driverobj.carobj.r_driven_tire(vel=self.vel_cl[i])
                                 * self.driverobj.carobj.pars_gearbox["e_i"][self.gear_cl[i]]))
                
                # account for available force due to change in elevation
                if self.trackobj.pars_track["use_elevation"]:
                    if i == len(self.trackobj.elevation_profile) - 1:
                        elevation_change = self.trackobj.elevation_profile[0] - self.trackobj.elevation_profile[i]
                    else:
                        elevation_change = self.trackobj.elevation_profile[i + 1] - self.trackobj.elevation_profile[i]
                    elevation_energy = (
                        elevation_change * 
                        self.driverobj.carobj.pars_general["m"] *
                        self.driverobj.carobj.pars_general["g"]
                    )

                    f_x_powert = f_x_powert - elevation_energy / self.trackobj.stepsize


                # calculate reached longitudinal acceleration
                a_x = ((f_x_powert - self.driverobj.carobj.air_res(vel=self.vel_cl[i], drs=self.trackobj.drs[i])
                       - self.driverobj.carobj.roll_res(f_z_tot=float(np.sum(self.tire_loads[i]))))
                       / self.driverobj.carobj.pars_general["m"])

                # calculate velocity in the next point
                try:
                    self.vel_cl[i + 1] = math.sqrt(math.pow(self.vel_cl[i], 2) + 2 * a_x * self.trackobj.stepsize)
                except ValueError as v:
                    result = math.pow(self.vel_cl[i], 2) + 2 * a_x * self.trackobj.stepsize
                    print("velocity: {}, a_x: {}, step_size: {}, result: {}"
                        .format(self.vel_cl[i], a_x, self.trackobj.stepsize, result))
                    raise(v)
                # consider velocity limit if reaching it during this step
                """This if statement is intended to prevent unnecessary backward loops. Therefore it should only come
                into operation if the velocity limit is reached from below (i.e. with acceleration) during the current
                step."""
                if self.vel_cl[i] <= vel_lim_cl[i + 1] < self.vel_cl[i + 1]:
                    # calculate a_x required to reach the velocity limit
                    a_x = (math.pow(vel_lim_cl[i + 1], 2)
                           - math.pow(self.vel_cl[i], 2)) / (2 * self.trackobj.stepsize)

                    f_x_target = (self.driverobj.carobj.air_res(vel=self.vel_cl[i], drs=False)
                                  + self.driverobj.carobj.roll_res(f_z_tot=float(np.sum(self.tire_loads[i])))
                                  + self.driverobj.carobj.pars_general["m"] * a_x)

                    # calculate torque distribution within the hybrid system (trying to reach the possible force f_x)
                    self.m_requ[i], self.m_eng[i], self.m_e_motor[i] = self.driverobj.carobj.\
                        calc_torque_distr_f_x(f_x=f_x_target,
                                              n=self.n_cl[i],
                                              throttle_pos=self.driverobj.throttle_pos[i],
                                              es=self.es_cl[i],
                                              em_boost_use=self.driverobj.em_boost_use[i],
                                              vel=self.vel_cl[i])

                    # set velocity accordingly
                    self.vel_cl[i + 1] = vel_lim_cl[i + 1]

                # check shifting -> calculate gear and rev in the next point
                self.gear_cl[i + 1], self.n_cl[i + 1] = self.driverobj.carobj.find_gear(vel=self.vel_cl[i + 1])

                # calculate time at start of next point
                self.t_cl[i + 1] = self.t_cl[i] + 2 * self.trackobj.stepsize / (self.vel_cl[i] + self.vel_cl[i + 1])

                # calculate energy recuperated during current step by electric turbocharger in [J] (only during acc.)
                if self.driverobj.carobj.powertrain_type == "hybrid" and self.driverobj.pars_driver["use_recuperation"]:
                    e_rec_etc = (self.driverobj.carobj.pars_engine["eta_etc_re"] * self.n_cl[i] * self.m_eng[i]
                                 * 2 * math.pi * (self.t_cl[i + 1] - self.t_cl[i]))
                else:
                    e_rec_etc = 0.0

                # calculate energy used by e motor during current step in [J]
                e_cons_e_motor = (self.driverobj.carobj.power_demand_e_motor_drive(n=self.n_cl[i],
                                                                                   m_e_motor=self.m_e_motor[i])
                                  * (self.t_cl[i + 1] - self.t_cl[i]))

                # calculate changes in the hybrid energy storage [J]
                self.es_cl[i + 1] = self.es_cl[i] + e_rec_etc - e_cons_e_motor

                if not self.driverobj.carobj.powertrain_type == "electric" and self.es_cl[i + 1] < 0.0:
                    self.es_cl[i + 1] = 0.0
                
                self.e_motor_power[i] = self.driverobj.carobj.power_demand_e_motor_drive(n=self.n_cl[i],
                                                                                         m_e_motor=self.m_e_motor[i])

                # increment
                i += 1

            # ----------------------------------------------------------------------------------------------------------
            # CASE 2: lateral force is greater than poss. tot. tire force of an axle or speed limit must be kept > brake
            # ----------------------------------------------------------------------------------------------------------

            else:
                # check if start velocity is too high
                if i == 1:
                    raise RuntimeError("Reduce start velocity! (it could be that braking would affect points within the"
                                       + " previous lap)")

                # get maximum current velocity depending on speed limit or lateral acceleration limit due to curvature
                self.vel_cl[i] = min(self.driverobj.carobj.
                                     v_max_cornering(kappa=self.trackobj.kappa[i],
                                                     mu=self.trackobj.mu[i],
                                                     vel_subtr_corner=self.driverobj.pars_driver["vel_subtr_corner"]),
                                     self.trackobj.vel_lim[i])

                # ------------------------------------------------------------------------------------------------------
                # BACKWARD ITERATIONS -> MAXIMUM CURRENT VELOCITY SHOULD BE KEPT AT CURRENT POINT i --------------------
                # ------------------------------------------------------------------------------------------------------

                j = 0
                a_x = 0.0  # reset longitudinal acceleration (almost zero during maximum cornering)

                while True:
                    """Subsequent while loop is used to obtain the best possible approximaten of the velocity at the
                    previous point vel_tmp. The termination criterion is checked using vel_tmp_old and tol. To
                    counteract infinite loops convergence is forced with an increasing loop counter."""

                    # loop until a good approximation for the velocity in the previous point is found
                    vel_tmp = self.vel_cl[i - j]    # [m/s] applied velocity (current point used as a starting value)
                    vel_tmp_old = 0.0  # [m/s] used to save the old value to compare for the termination criterion
                    vels_tmp = []  # used to save velocity values to be able to calculate a mean value for convergence
                    tire_loads_tmp = np.zeros(4)    # [N] tire loads [FL, FR, RL, RR]
                    counter = 0                     # [-] loop counter

                    while math.fabs(vel_tmp - vel_tmp_old) > tol:
                        # increase counter and store previous value to be able to check for the termination criterion
                        counter += 1
                        vel_tmp_old = vel_tmp

                        # calculate lat. acceleration and forces with temporary stored velocity and previous curvature
                        a_y = math.pow(vel_tmp, 2) * self.trackobj.kappa[i - j - 1]
                        f_y_f, f_y_r = self.driverobj.carobj.calc_lat_forces(a_y=a_y)

                        # calculate tire force potentials
                        f_x_pot_fl, f_y_pot_fl, tire_loads_tmp[0], \
                            f_x_pot_fr, f_y_pot_fr, tire_loads_tmp[1], \
                            f_x_pot_rl, f_y_pot_rl, tire_loads_tmp[2], \
                            f_x_pot_rr, f_y_pot_rr, tire_loads_tmp[3] = self.driverobj.carobj.\
                            tire_force_pots(vel=vel_tmp,
                                            a_x=a_x,
                                            a_y=a_y,
                                            mu=self.trackobj.mu[i - j - 1])

                        # flag to only use regen, this is used in the next function
                        # to determine if the possible deceleration force is calculated
                        # using all the wheels (as it would in braking) or with only 
                        # the driven wheels (as in regen)
                        regen_only = self.driverobj.pars_driver["disable_braking"]

                        # only limit the braking forces on the weak side when braking is
                        # being used, we can avoid using this because I'm assuming we will
                        # have a limited slip that can handle putting maximum torque
                        # to each driven tire
                        if regen_only: 
                            limit_braking_weak_side = None
                        else:
                            limit_braking_weak_side = self.pars_solver["limit_braking_weak_side"]

                        # calculate remaining tire potential for deceleration using all wheels
                        # assumption: potential always usable by proper brake force distribution
                        f_x_poss = self.driverobj.carobj.\
                            calc_f_x_pot(f_x_pot_fl=f_x_pot_fl,
                                        f_x_pot_fr=f_x_pot_fr,
                                        f_x_pot_rl=f_x_pot_rl,
                                        f_x_pot_rr=f_x_pot_rr,
                                        f_y_pot_f=f_y_pot_fl + f_y_pot_fr,
                                        f_y_pot_r=f_y_pot_rl + f_y_pot_rr,
                                        f_y_f=f_y_f,
                                        f_y_r=f_y_r,
                                        force_use_all_wheels= not regen_only,
                                        limit_braking_weak_side=limit_braking_weak_side)

                        # the above may still result in an invalid force even if the force_use_all_wheels
                        # variable is false, we need to do checks on motor torque and motor power to make sure
                        # the output force of the drivetrain is minimum of the motor torque, motor power, 
                        # and tire limitations if regen_only is set
                        if regen_only:
                            max_motor_torque = self.driverobj.carobj.pars_engine["torque_e_motor_max"]
                            max_motor_power = self.driverobj.carobj.pars_engine["pow_e_motor"]

                            # calculate the maximum longitudinal force based on max motor torque
                            # I divided by efficiency in the below equation because we are braking and 
                            # the efficiency loss of the drivetrain will be ADDED to the output because
                            # we are slowing down not speeding up
                            f_x_poss_torque = ( max_motor_torque 
                                / self.driverobj.carobj.pars_gearbox["eta_g"]
                                / (self.driverobj.carobj.pars_gearbox["i_trans"][self.gear_cl[i - j - 1]]
                                * self.driverobj.carobj.r_driven_tire(vel=self.vel_cl[i - j - 1])
                                * self.driverobj.carobj.pars_gearbox["e_i"][self.gear_cl[i - j - 1]]))
                            
                            # calculate the maximum longitudinal force based on max motor power
                            # same rational as above for dividing by gearbox efficiency instead of multiplying
                            # power = force * velocity -> force = power / velocity
                            f_x_poss_power = (max_motor_power 
                                / self.driverobj.carobj.pars_engine["eta_e_motor_re"]
                                / self.driverobj.carobj.pars_gearbox["eta_g"]
                                / vel_tmp)

                            f_x_poss = min(f_x_poss, f_x_poss_torque, f_x_poss_power)
                        
                        # calculate the energy change due to elevation change
                        if self.trackobj.pars_track["use_elevation"]:
                            elevation_change = self.trackobj.elevation_profile[i - j - 1 + 1] \
                                             - self.trackobj.elevation_profile[i - j - 1]
                            elevation_energy = (
                                elevation_change * 
                                self.driverobj.carobj.pars_general["m"] *
                                self.driverobj.carobj.pars_general["g"]
                            )

                            # + sign because the f_x_poss is positive for deceleration
                            # and positive elevation change will also decelerate the vehicle
                            f_x_poss = f_x_poss + elevation_energy / self.trackobj.stepsize

                        # calculate deceleration
                        a_x = (-(f_x_poss + self.driverobj.carobj.air_res(vel=vel_tmp, drs=False)
                                 + self.driverobj.carobj.roll_res(f_z_tot=float(np.sum(tire_loads_tmp))))
                               / self.driverobj.carobj.pars_general["m"])

                        # calculate previous velocity (-a_x because we go backwards and therefore need a positive
                        # acceleration within the equation) and append it to vels_tmp
                        vels_tmp.append(math.sqrt(math.pow(self.vel_cl[i - j], 2) + 2 * -a_x * self.trackobj.stepsize))

                        # calculate applied velocity as average of all to get a robust convergence characteristic
                        vel_tmp = sum(vels_tmp) / len(vels_tmp)

                        # start forcing the algorithm to converge after 10 iterations by reducing the temporary velocity
                        # if counter > 10:
                        #     vel_tmp -= (counter - 10) * force_conv

                    # check if the calculated velocity is greater than the original one -> break the loop
                    if vel_tmp >= self.vel_cl[i - j - 1]:
                        break  # without incrementing -> i - j - 1 is last unchanged point
                    else:
                        self.vel_cl[i - j - 1] = vel_tmp
                        self.tire_loads[i - j - 1, :] = tire_loads_tmp

                    # increment to previous point
                    j += 1

                    if i - j - 1 < 0:
                        raise RuntimeError("Reduce start velocity (it could be that braking would affect points within"
                                           " the previous lap)!")

                # ------------------------------------------------------------------------------------------------------
                # MODIFIED VELOCITY PROFILE IS KNOWN -> RECALCULATION OF REMAINING DATA ON THIS BASIS ------------------
                # ------------------------------------------------------------------------------------------------------

                # recalculate gears and revs for all changed points including current point
                for k in range(i - j, i + 1):
                    self.gear_cl[k], self.n_cl[k] = self.driverobj.carobj.find_gear(vel=self.vel_cl[k])

                # recalculate lap times starting from the last unchanged point i - j - 1
                for k in range(i - j - 1, i):
                    self.t_cl[k + 1] = self.t_cl[k] + 2 * self.trackobj.stepsize / (self.vel_cl[k + 1] + self.vel_cl[k])

                # recalculate energy related quantities starting from the last unchanged point i - j - 1
                for k in range(i - j - 1, i):
                    # calculate resistance force (in first point, DRS is set as it is prescribed by the track to avoid
                    # a sudden rise of the drag resistance which cannot be overcome by the available powertrain torque
                    # in some cases where the DRS was active in the forward loop, in all other points it is deactivated
                    # to consider the maximal possible deceleration that was used to calculate the velocity profile also
                    # in the energy calculations)
                    if k == i - j - 1:
                        drs_tmp = self.trackobj.drs[k]
                    else:
                        drs_tmp = False
                    

                    f_x_resi = (self.driverobj.carobj.air_res(vel=self.vel_cl[k], drs=drs_tmp)
                                + self.driverobj.carobj.roll_res(f_z_tot=float(np.sum(self.tire_loads[k]))))

                    # calculate the longitudinal acceleration and force required for the given velocities
                    a_x_requ = (math.pow(self.vel_cl[k + 1], 2)
                                - math.pow(self.vel_cl[k], 2)) / (2 * self.trackobj.stepsize)
                    f_x_requ = self.driverobj.carobj.pars_general["m"] * a_x_requ
                    
                    # calculate force that must be provided by the powertrain (or brakes) to reach this acc. force
                    f_x_powert = f_x_requ + f_x_resi 
                    
                    if self.trackobj.pars_track["use_elevation"]:
                        elevation_change = self.trackobj.elevation_profile[k + 1] - self.trackobj.elevation_profile[k]
                        elevation_energy = (
                            elevation_change * 
                            self.driverobj.carobj.pars_general["m"] *
                            self.driverobj.carobj.pars_general["g"]
                        )

                        f_x_powert = f_x_powert + elevation_energy / self.trackobj.stepsize

                    # check for the two cases "engine demanded" and "engine not demanded"
                    if f_x_powert > 0.0:
                        """Engine demanded (this is the case if resistances must be overcome or if the car is
                        accelerating). Therefore, we have to recalculate the torque distribution and the energy storage
                        state."""
                        self.e_rec_e_motor[k] = 0.0

                        # calculate torque distribution within the hybrid system (trying to reach possible force f_x)
                        self.m_requ[k], self.m_eng[k], self.m_e_motor[k] = self.driverobj.carobj.\
                            calc_torque_distr_f_x(f_x=f_x_powert,
                                                  n=self.n_cl[k],
                                                  throttle_pos=self.driverobj.throttle_pos[k],
                                                  es=self.es_cl[k],
                                                  em_boost_use=self.driverobj.em_boost_use[k],
                                                  vel=self.vel_cl[k])

                        # check torques provided and requested
                        if not math.isclose(self.m_eng[k] + self.m_e_motor[k], self.m_requ[k]):
                            print(
                                "WARNING: It seems like if the requested torque could not be supplied by the"
                                + " powertrain (maybe because of energy storage changes during recalculation or because"
                                + " the throttle position was set to 0.0 during EM strategy calculation). Be aware"
                                + " that this fact is not considered and the further calculation is processed as if the"
                                + " torque was supplied! The difference amounts to %.1f Nm."
                                % (self.m_requ[k] - (self.m_eng[k] + self.m_e_motor[k])))

                        # calculate energy recuperated during current step by el. turbocharger in [J] (only during acc.)
                        if self.driverobj.carobj.powertrain_type == "hybrid" and \
                                self.driverobj.pars_driver["use_recuperation"]:
                            e_rec_etc = (self.driverobj.carobj.pars_engine["eta_etc_re"] * self.n_cl[k]
                                         * self.m_eng[k] * 2 * math.pi * (self.t_cl[k + 1] - self.t_cl[k]))
                        else:
                            e_rec_etc = 0.0

                        # calculate energy used by e motor during current step in [J]
                        e_cons_e_motor = (self.driverobj.carobj.power_demand_e_motor_drive(n=self.n_cl[k],
                                                                                           m_e_motor=self.m_e_motor[k])
                                          * (self.t_cl[k + 1] - self.t_cl[k]))
                        
                        self.e_motor_power[i] = self.driverobj.carobj.power_demand_e_motor_drive(n=self.n_cl[i],
                                                                                         m_e_motor=self.m_e_motor[i])

                        # calculate changes in the hybrid energy storage [J]
                        self.es_cl[k + 1] = self.es_cl[k] + e_rec_etc - e_cons_e_motor

                        if not self.driverobj.carobj.powertrain_type == "electric" and self.es_cl[k + 1] < 0.0:
                            self.es_cl[k + 1] = 0.0

                    else:
                        """Engine not demanded -> no powertrain torques required. However, we have to calculate kinetic
                        energy recuperation."""
                        self.m_eng[k] = 0.0
                        self.m_requ[k] = 0.0

                        # energy recuperation by e motor in [J] under the assumption of e motor being able to recuperate
                        # all kinetic energy remaining after subtraction of the resistances
                        if np.sum(self.e_rec_e_motor) < self.e_rec_e_motor_max \
                                and self.driverobj.pars_driver["use_recuperation"]:

                            # calculate the energy regenerated as the minimum of the energy recuped by
                            # calculating from the tire forces, the maximum motor torque, and maximum motor power
                            # it is possible that the power through the tires exceeds the maximum capability
                            # of the motors if braking is being used, and the motor can only regen at the 
                            # capability of the motor

                            # these energy calculations are done "at the tire" that to say: the efficiencies
                            # are applied to the calculations so that for motor torque and power the equivalent
                            # energy at the tire is found. This is to have an apples to apples comparison.

                            # then after that calculation is done the actual motor energy stored is recalculated

                            e_rec_e_motor_tires = math.fabs(f_x_powert) * self.trackobj.stepsize

                            max_motor_torque = self.driverobj.carobj.pars_engine["torque_e_motor_max"]
                            max_motor_power = self.driverobj.carobj.pars_engine["pow_e_motor"] 

                            # get max energy based on torque by dividing by gear ratio and tire diameter
                            # to get tire force and then multiplying by step size
                            e_rec_e_motor_torque = (max_motor_torque
                                / self.driverobj.carobj.pars_gearbox["eta_g"]
                                / self.driverobj.carobj.pars_gearbox["i_trans"][self.gear_cl[k]] 
                                / self.driverobj.carobj.r_driven_tire(vel=self.vel_cl[k])
                                * self.trackobj.stepsize
                            )

                            # energy = power * time
                            # TODO: check that efficiencies are applied correctly (and which ones!)
                            e_rec_e_motor_power = (max_motor_power 
                                / self.driverobj.carobj.pars_engine["eta_e_motor_re"]
                                / self.driverobj.carobj.pars_gearbox["eta_g"]
                                * (self.t_cl[k] - self.t_cl[k - 1])
                            )

                            e_rec_tmp = min(e_rec_e_motor_tires, e_rec_e_motor_torque, e_rec_e_motor_power)
                            self.e_rec_e_motor[k] = (e_rec_tmp
                                * self.driverobj.carobj.pars_gearbox["eta_g"]
                                * self.driverobj.carobj.pars_engine["eta_e_motor_re"]
                            ) 
                            # get actual motor power and torque from the energy recuperated
                            # this needs to be back calculated this way instead of using the motor torque and
                            # rpm because we are using the energy regenerated as the actual motor condition
                            # then calculating power from energy and torque from power. Is this the best way
                            # to do that??
                            self.e_motor_power[k] = - self.e_rec_e_motor[k] / (self.t_cl[k + 1] - self.t_cl[k])
                            
                            self.m_e_motor[k] = (self.e_motor_power[k] 
                                / (self.n_cl[k] * 2 * math.pi) 
                                * self.driverobj.carobj.pars_engine["eta_e_motor_re"]
                            )

                            if abs(self.m_e_motor[k]) > max_motor_torque:
                                raise(Exception("Maximum motor torque exceeded: {}, max: {}".format(self.m_e_motor[k], max_motor_torque)))

                            if abs(self.m_e_motor[k]) > max_motor_torque:
                                raise(Exception("Maximum motor torque exceeded: {}, max: {}".format(self.m_e_motor[k], max_motor_torque)))

                            # check that the maximum power of the drivetrain is not violated
                            if abs(self.e_motor_power[k]) > max_motor_power:
                                raise(Exception("Maximum Motor power exceeded: {} max: {}".format(self.e_motor_power[k], max_motor_power)))

                        else:
                            self.e_rec_e_motor[k] = 0.0

                        # update energy storage (no energy harvested in el. turbocharger while engine is not demanded)
                        self.es_cl[k + 1] = self.es_cl[k] + self.e_rec_e_motor[k]


                # reset longitudinal acceleration for next step (almost zero during maximum cornering)
                a_x = 0.0

        # --------------------------------------------------------------------------------------------------------------
        # PREPARE FOR RETURN -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # return real instead of index based gear values
        self.gear_cl = self.gear_cl + 1

        # save final a_x value for a possible recalculation of the lap
        self.a_x_final = a_x

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (ANALYSIS) -----------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def plot_lat_acc(self):
        a_y_tmp = np.power(self.vel_cl[:-1], 2) * self.trackobj.kappa

        a_y_valid = 30.0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], a_y_tmp)
        ax.axhline(y=-a_y_valid, color="k", linestyle="--", linewidth=3.0)  # valid lateral acceleration limit
        ax.axhline(y=a_y_valid, color="k", linestyle="--", linewidth=3.0)  # valid lateral acceleration limit
        ax.set_title("Lateral acceleration profile")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("lateral acceleration ay in m/s2")
        plt.grid()
        plt.show()

    def plot_torques(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_eng)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_e_motor)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_eng + self.m_e_motor)
        plt.plot(self.trackobj.dists_cl[:-1], self.m_requ)
        ax.set_title("Provided and requested (i.e. transmittable by the tires) torque")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("torque in Nm")
        plt.legend(["combustion engine", "electric motor", "powertrain total", "requested"])
        plt.grid()
        plt.show()

    def plot_power(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.e_motor_power)
        ax.set_title("Motor Power")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("Power in Watts")
        plt.grid()
        plt.show()

    def plot_throttle(self):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.driverobj.throttle_pos)
        ax.set_title("Throttle Position")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("Throttle position (0-1)")
        plt.grid()
        plt.show()
    
    

    def plot_tire_loads(self):
        f_z_stat_avg = 0.25 * self.driverobj.carobj.pars_general["m"] * self.driverobj.carobj.pars_general["g"]

        f_z_dyn_valid = f_z_stat_avg * 3.0
        legend_text = "3 * avg. static"

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 0])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 1])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 2])
        plt.plot(self.trackobj.dists_cl[:-1], self.tire_loads[:, 3])
        # plt.plot(self.trackobj.dists_cl[:-1], np.sum(self.tire_loads, axis=1))
        ax.axhline(y=f_z_stat_avg, color="k", linestyle="--", linewidth=3.0)   # valid tire load range
        ax.axhline(y=f_z_dyn_valid, color="k", linestyle="--", linewidth=3.0)  # valid tire load range
        ax.set_title("Tire loads")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("tire load F_z in N")
        plt.legend(["front left", "front right", "rear left", "rear right", "avg. static", legend_text])
        # plt.legend(["front left", "front right", "rear left", "rear right", "total", "avg. static",
        #             "7 * avg. static"])
        plt.grid()
        plt.show()

    def plot_aero_forces(self):
        c_z_a_f = self.driverobj.carobj.pars_general["c_z_a_f"]
        c_z_a_r = self.driverobj.carobj.pars_general["c_z_a_r"]
        c_w_a = self.driverobj.carobj.pars_general["c_w_a"]
        rho_air = self.driverobj.carobj.pars_general["rho_air"]

        f_x_aero = 0.5 * c_w_a * rho_air * np.power(self.vel_cl[:-1], 2)
        f_z_aero = 0.5 * (c_z_a_f + c_z_a_r) * rho_air * np.power(self.vel_cl[:-1], 2)

        f_z_aero_valid = 6e3

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(self.trackobj.dists_cl[:-1], f_x_aero)
        plt.plot(self.trackobj.dists_cl[:-1], f_z_aero)
        ax.axhline(y=f_z_aero_valid, color="k", linestyle="--", linewidth=3.0)   # valid downforce range
        ax.set_title("Aero forces (no DRS considered!)")
        ax.set_xlabel("distance s in m")
        ax.set_ylabel("amplitude in N")
        plt.legend(["drag", "downforce", "valid downforce range"])
        plt.grid()
        plt.show()

    def plot_enginespeed_gears(self):
        fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=True)
        fig.suptitle("Engine speed and gear selection")

        ax1.plot(self.trackobj.dists_cl[:-1], self.n_cl[:-1] * 60.0)
        ax1.set_xlabel("distance s in m")
        ax1.set_ylabel("engine speed in 1/min")
        ax1.grid()

        ax2.plot(self.trackobj.dists_cl[:-1], self.gear_cl[:-1])
        ax2.set_xlabel("distance s in m")
        ax2.set_ylabel("gear in -")
        ax2.grid()

        plt.show()

    def plot_overview(self):
        # set bigger font size
        plt.rcParams["font.size"] = 16.0

        # create figure
        plt.figure(1, figsize=(12.0, 9.0))

        # --------------------------------------------------------------------------------------------------------------
        # VELOCITY PROFILE ---------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        # plot velocity profile (two columns)
        plt.subplot(2, 1, 1)
        plt.plot(self.trackobj.dists_cl, self.vel_cl * 3.6, "k-")
        plt.xlabel("distance in m")
        plt.ylabel("velocity in km/h")
        plt.grid()

        # plot global title
        title = "lap time: %.3f s" % self.t_cl[-1]
        plt.title(title)

        # --------------------------------------------------------------------------------------------------------------
        # TRACK MAP ----------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        ax1 = plt.subplot(2, 2, 3)

        # plot raceline
        ax1.plot(self.trackobj.raceline[:, 0], self.trackobj.raceline[:, 1], "k-")

        # plot DRS zones
        if self.trackobj.pars_track["use_drs1"]:
            if self.trackobj.zone_inds["drs1_a"] < self.trackobj.zone_inds["drs1_d"]:
                # common case
                ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["drs1_a"]:self.trackobj.zone_inds["drs1_d"], 0],
                         self.trackobj.raceline[self.trackobj.zone_inds["drs1_a"]:self.trackobj.zone_inds["drs1_d"], 1],
                         "g--", linewidth=3.0)
            else:
                # DRS zone is split by start/finish line
                ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["drs1_a"]:, 0],
                         self.trackobj.raceline[self.trackobj.zone_inds["drs1_a"]:, 1],
                         "g--", linewidth=3.0)
                ax1.plot(self.trackobj.raceline[:self.trackobj.zone_inds["drs1_d"], 0],
                         self.trackobj.raceline[:self.trackobj.zone_inds["drs1_d"], 1],
                         "g--", linewidth=3.0)

        if self.trackobj.pars_track["use_drs2"]:
            if self.trackobj.zone_inds["drs2_a"] < self.trackobj.zone_inds["drs2_d"]:
                # common case
                ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["drs2_a"]:self.trackobj.zone_inds["drs2_d"], 0],
                         self.trackobj.raceline[self.trackobj.zone_inds["drs2_a"]:self.trackobj.zone_inds["drs2_d"], 1],
                         "g--", linewidth=3.0)
            else:
                # DRS zone is split by start/finish line
                ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["drs2_a"]:, 0],
                         self.trackobj.raceline[self.trackobj.zone_inds["drs2_a"]:, 1],
                         "g--", linewidth=3.0)
                ax1.plot(self.trackobj.raceline[:self.trackobj.zone_inds["drs2_d"], 0],
                         self.trackobj.raceline[:self.trackobj.zone_inds["drs2_d"], 1],
                         "g--", linewidth=3.0)

        # plot yellow sectors
        if self.driverobj.pars_driver["yellow_s1"]:
            ax1.plot(self.trackobj.raceline[:self.trackobj.zone_inds["s12"], 0],
                     self.trackobj.raceline[:self.trackobj.zone_inds["s12"], 1],
                     "y--")
        if self.driverobj.pars_driver["yellow_s2"]:
            ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["s12"]:self.trackobj.zone_inds["s23"], 0],
                     self.trackobj.raceline[self.trackobj.zone_inds["s12"]:self.trackobj.zone_inds["s23"], 1],
                     "y--")
        if self.driverobj.pars_driver["yellow_s3"]:
            ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["s23"]:, 0],
                     self.trackobj.raceline[self.trackobj.zone_inds["s23"]:, 1],
                     "y--")

        # plot pit
        if self.trackobj.pars_track["use_pit"]:
            ax1.plot(self.trackobj.raceline[:self.trackobj.zone_inds["pit_out"], 0],
                     self.trackobj.raceline[:self.trackobj.zone_inds["pit_out"], 1],
                     "r--", linewidth=3.0)
            ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["pit_in"]:, 0],
                     self.trackobj.raceline[self.trackobj.zone_inds["pit_in"]:, 1],
                     "r--", linewidth=3.0)

        # plot arrow showing the driving direction
        ax1.arrow(self.trackobj.raceline[0, 0], self.trackobj.raceline[0, 1],
                  self.trackobj.raceline[10, 0] - self.trackobj.raceline[0, 0],
                  self.trackobj.raceline[10, 1] - self.trackobj.raceline[0, 1],
                  head_width=30.0, width=10.0)

        # plot dots at start/finish and at the sector boundaries
        ax1.plot(self.trackobj.raceline[0, 0], self.trackobj.raceline[0, 1], "k.", markersize=13.0)
        ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["s12"], 0],
                 self.trackobj.raceline[self.trackobj.zone_inds["s12"], 1], "k.", markersize=13.0)
        ax1.plot(self.trackobj.raceline[self.trackobj.zone_inds["s23"], 0],
                 self.trackobj.raceline[self.trackobj.zone_inds["s23"], 1], "k.", markersize=13.0)

        # plot velocity dependent colors into track map
        cmap = plt.get_cmap("RdYlGn")
        normalize = plt.Normalize(vmin=-np.amax(self.vel_cl), vmax=-np.amin(self.vel_cl))
        colors = [cmap(normalize(cur_vel)) for cur_vel in -self.vel_cl[:-1]]
        ax1.scatter(self.trackobj.raceline[:, 0], self.trackobj.raceline[:, 1], c=colors, s=5)

        plt.axis("equal")
        plt.title("track map: " + self.trackobj.pars_track["trackname"])
        plt.xlabel("x in m")
        plt.ylabel("y in m")
        plt.grid()

        # --------------------------------------------------------------------------------------------------------------
        # ENERGY CONSUMPTION -------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        ax1 = plt.subplot(2, 2, 4)
        ax1.plot(self.trackobj.dists_cl, self.es_cl / 1000.0, "k-")  # [J] -> [kJ]

        if not self.driverobj.carobj.powertrain_type == "electric":
            ax2 = ax1.twinx()
            ax2.plot(self.trackobj.dists_cl, self.fuel_cons_cl, "r-")
            ax2.set_ylabel("cumulated fuel consumption in kg", color="r")
            ax2.tick_params("y", colors="r")

        plt.title("consumption")
        ax1.set_xlabel("distance in m")
        ax1.set_ylabel("energy in kJ")
        plt.grid()

        # set tight plot layout and show plot
        plt.tight_layout()
        plt.show()

        # reset font size
        plt.rcParams["font.size"] = 10.0

    def plot_revs_gears(self):
        # --------------------------------------------------------------------------------------------------------------
        # REVS AND GEARS -----------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------

        fig, ax1 = plt.subplots(figsize=(12.0, 9.0))

        ax1.plot(self.trackobj.dists_cl, self.n_cl * 60.0)
        ax1.set_xlabel("distance in m")
        ax1.set_ylabel("rev in 1/min")

        ax2 = ax1.twinx()
        ax2.plot(self.trackobj.dists_cl, self.gear_cl, "r")
        ax2.set_ylabel("gear")

        fig.tight_layout()
        plt.show()


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
