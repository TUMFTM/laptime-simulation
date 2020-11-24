import numpy as np
import math
import matplotlib.pyplot as plt
import json
from laptimesim.src.car import Car
import configparser


class CarHybrid(Car):
    """
    author:
    Alexander Heilmeier (based on the term thesis of Maximilian Geisslinger)

    date:
    23.12.2018

    .. description::
    The file provides functions related to the vehicle, e.g. power and torque calculations.
    """

    # ------------------------------------------------------------------------------------------------------------------
    # SLOTS ------------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    __slots__ = "__z_pow_engine"

    # ------------------------------------------------------------------------------------------------------------------
    # CONSTRUCTOR ------------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, parfilepath: str):

        # load vehicle parameters
        parser = configparser.ConfigParser()

        if not parser.read(parfilepath):
            raise RuntimeError('Specified config file does not exist or is empty!')

        pars_veh_tmp = json.loads(parser.get('VEH_PARS', 'veh_pars'))

        # unit conversions
        pars_veh_tmp["engine"]["n_begin"] /= 60.0   # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["n_max"] /= 60.0     # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["n_end"] /= 60.0     # [1/min] -> [1/s]
        pars_veh_tmp["engine"]["be_max"] /= 3600.0  # [kg/h] -> [kg/s]

        for i, item in enumerate(pars_veh_tmp["gearbox"]["n_shift"]):
            pars_veh_tmp["gearbox"]["n_shift"][i] = item / 60.0  # [1/min] -> [1/s]

        # convert gearbox arrays to numpy arrays
        pars_veh_tmp["gearbox"]["i_trans"] = np.array(pars_veh_tmp["gearbox"]["i_trans"])
        pars_veh_tmp["gearbox"]["n_shift"] = np.array(pars_veh_tmp["gearbox"]["n_shift"])
        pars_veh_tmp["gearbox"]["e_i"] = np.array(pars_veh_tmp["gearbox"]["e_i"])

        # initialize base class object
        Car.__init__(self,
                     powertrain_type=pars_veh_tmp["powertrain_type"],
                     pars_general=pars_veh_tmp["general"],
                     pars_engine=pars_veh_tmp["engine"],
                     pars_gearbox=pars_veh_tmp["gearbox"],
                     pars_tires=pars_veh_tmp["tires"])

        # calculate ICE power curve LES coefficients (z_pow_engine)
        pow_max = self.pars_engine["pow_max"]
        pow_diff = self.pars_engine["pow_diff"]
        n_begin = self.pars_engine["n_begin"]
        n_max = self.pars_engine["n_max"]
        n_end = self.pars_engine["n_end"]
        pow_begend = pow_max - pow_diff

        a = np.array([[math.pow(n_begin, 3), math.pow(n_begin, 2), n_begin, 1],
                      [3 * math.pow(n_max, 2), 2 * n_max, 1, 0],
                      [math.pow(n_max, 3), math.pow(n_max, 2), n_max, 1],
                      [math.pow(n_end, 3), math.pow(n_end, 2), n_end, 1]])
        b = np.array([[pow_begend], [0], [pow_max], [pow_begend]])
        self.z_pow_engine = np.linalg.solve(a, b)

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __get_z_pow_engine(self) -> np.ndarray: return self.__z_pow_engine
    def __set_z_pow_engine(self, x: np.ndarray) -> None: self.__z_pow_engine = x
    z_pow_engine = property(__get_z_pow_engine, __set_z_pow_engine)

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def __power_engine(self, n: float or np.ndarray):
        """
        Power curve is approximated by a peak power pow_max at n_max and equal drops on both sides at n_begin and n_end.
        Rev input is in 1/s, output is in W.
        """

        # get relevant data
        n_begin = self.pars_engine["n_begin"]
        n_end = self.pars_engine["n_end"]

        # limit engine speed to valid range of power curve
        n_use = np.copy(n)
        n_use[n_use < 0.75 * n_begin] = 0.75 * n_begin
        n_use[n_use > 1.2 * n_end] = 1.2 * n_end

        # calculate power
        p_eng = (self.z_pow_engine[0] * np.power(n_use, 3) + self.z_pow_engine[1] * np.power(n_use, 2)
                 + self.z_pow_engine[2] * n_use + self.z_pow_engine[3])
        p_eng[p_eng < 0.0] = 0.0  # assure that no negativ powers appear

        return p_eng

    def plot_power_engine(self) -> None:
        # plot
        n_range = np.arange(7000.0, 15100.0, 100.0) / 60.0  # [1/s]

        plt.figure()
        plt.plot(n_range * 60.0, self.__power_engine(n=n_range) / 1000.0 * 1.36)
        plt.title("Engine power characteristics")
        plt.xlabel("n in 1/min")
        plt.ylabel("P in PS")

        plt.show()

    def torque(self, n: float) -> float:
        """Rev input in 1/s. Output is the maximum torque in Nm."""

        return float(self.__power_engine(n=n)) / (2 * math.pi * n)

    def torque_e_motor(self, n: float) -> float:
        """Rev input in 1/s. Output is the maximum torque in Nm."""

        torque_tmp = self.pars_engine["pow_e_motor"] / (2 * math.pi * n)

        if torque_tmp > self.pars_engine["torque_e_motor_max"]:
            torque_tmp = self.pars_engine["torque_e_motor_max"]

        return torque_tmp

    def fuel_cons(self, t_cl: np.ndarray, n_cl: np.ndarray, m_eng: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is the consumed fuel mass until the current point in kg
        (closed)."""

        be_kgs = self.__injectionmap(n=n_cl[:-1],
                                     m_eng=m_eng)  # [kg/s]

        # integrate
        consumpt_kg_part = np.diff(t_cl) * be_kgs
        consumpt_kg_cl = np.insert(np.cumsum(consumpt_kg_part), 0, 0.0)  # [kg]

        return consumpt_kg_cl

    def __injectionmap(self, n: np.ndarray, m_eng: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is in kg/s. Model of the engine fuel consumption."""

        pow_actual = 2 * math.pi * n * m_eng  # [W]
        pow_max = self.__power_engine(n=n)    # [W]
        be = np.sqrt(pow_actual / pow_max) * self.pars_engine["be_max"]  # [kg/s]

        return be

    def e_cons(self, t_cl: np.ndarray, n_cl: np.ndarray, m_e_motor: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is the consumed energy in J until the current point(closed).
        Calculates used energy including the efficiency."""

        be_w = self.power_demand_e_motor_drive(n=n_cl[:-1],
                                               m_e_motor=m_e_motor)  # [W]

        # integrate
        e_consumpt_j_part = np.diff(t_cl) * be_w  # [J]
        e_consumpt_j_cl = np.insert(np.cumsum(e_consumpt_j_part), 0, 0.0)  # [J]

        return e_consumpt_j_cl

    def power_demand_e_motor_drive(self, n: np.ndarray, m_e_motor: np.ndarray) -> np.ndarray:
        """Rev input in 1/s, torque input in Nm. Output is in W. Calculates used power including the efficiency."""

        return (2 * math.pi * n * m_e_motor) / self.pars_engine["eta_e_motor"]

    def calc_torque_distr(self, n: float, m_requ: float, throttle_pos: float, es: float,
                          em_boost_use: bool, vel: float) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # get torque potential of engine and e motor
        eng_torque_max = self.torque(n=n)
        e_motor_torque_max = self.torque_e_motor(n=n)

        if m_requ <= eng_torque_max:  # ICE only
            m_eng = throttle_pos * m_requ
            m_e_motor = 0.0

        elif m_requ <= eng_torque_max + e_motor_torque_max:  # ICE + e motor (partly)
            m_eng = throttle_pos * eng_torque_max

            if es > 0.0 and em_boost_use and vel >= self.pars_engine["vel_min_e_motor"]:
                m_e_motor = throttle_pos * (m_requ - eng_torque_max)
            else:
                m_e_motor = 0.0

        else:  # ICE + e motor (fully)
            m_eng = throttle_pos * eng_torque_max

            if es > 0.0 and em_boost_use and vel >= self.pars_engine["vel_min_e_motor"]:
                m_e_motor = throttle_pos * e_motor_torque_max
            else:
                m_e_motor = 0.0

        return m_eng, m_e_motor

    def calc_torque_distr_f_x(self, f_x: float, n: float, throttle_pos: float, es: float,
                              em_boost_use: bool, vel: float) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # calculate required torque to reach f_x
        m_requ = self.calc_m_requ(f_x=f_x,
                                  vel=vel)

        # get torque potential of engine and e motor
        m_eng, m_e_motor = self.calc_torque_distr(n=n,
                                                  m_requ=m_requ,
                                                  throttle_pos=throttle_pos,
                                                  es=es,
                                                  em_boost_use=em_boost_use,
                                                  vel=vel)

        return m_requ, m_eng, m_e_motor


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
