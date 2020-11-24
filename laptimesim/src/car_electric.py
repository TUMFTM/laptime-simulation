import numpy as np
import math
import json
from laptimesim.src.car import Car
import configparser


class CarElectric(Car):
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

    # none

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

    # ------------------------------------------------------------------------------------------------------------------
    # GETTERS / SETTERS ------------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    # none

    # ------------------------------------------------------------------------------------------------------------------
    # METHODS (CALCULATIONS) -------------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------------

    def torque_e_motor(self, n: float) -> float:
        """Rev input in 1/s. Output is the maximum torque in Nm."""

        torque_tmp = self.pars_engine["pow_e_motor"] / (2 * math.pi * n)

        if torque_tmp > self.pars_engine["torque_e_motor_max"]:
            torque_tmp = self.pars_engine["torque_e_motor_max"]

        return torque_tmp

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
                          **kwargs) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # get torque potential of e motor
        e_motor_torque_max = self.torque_e_motor(n=n)

        if m_requ <= e_motor_torque_max and es > 0.0:
            m_e_motor = throttle_pos * m_requ
        elif es > 0.0:
            m_e_motor = throttle_pos * e_motor_torque_max
        else:
            m_e_motor = 5.0  # set minimum torque such that the lap can be finished

        return 0.0, m_e_motor

    def calc_torque_distr_f_x(self, f_x: float, n: float, throttle_pos: float, es: float, vel: float,
                              **kwargs) -> tuple:
        """n in 1/s, torque_req in Nm, es in J. Function returns torques delivered by engine and e motor in
        Nm."""

        # calculate required torque to reach f_x
        m_requ = self.calc_m_requ(f_x=f_x,
                                  vel=vel)

        # get torque potential of e motor (es state is currently not considered since this makes no real sense
        # in FE calculations)
        m_e_motor = self.calc_torque_distr(n=n,
                                           m_requ=m_requ,
                                           throttle_pos=throttle_pos,
                                           es=np.inf)[1]

        return m_requ, 0.0, m_e_motor


# ----------------------------------------------------------------------------------------------------------------------
# TESTING --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    pass
