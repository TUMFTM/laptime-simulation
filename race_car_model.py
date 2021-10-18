# classes to store the relationship of race car properties
from definitions import *

class generalParameters():
    """Class for holding general parameters. These variables are a mirror
    of what is in the config file and some additional params that are 
    calculated implicitly to the race car model."""
    def __init__(self):
        self.lf = -1
        self.lr = -1
        self.h_cog = -1
        self.sf = -1
        self.sr = -1
        self.m = -1
        self.f_roll = -1
        self.c_w_a = -1
        self.c_z_a_f = -1
        self.c_z_a_r = -1
        self.g = -1
        self.rho_air = -1
        self.drs_factor = -1
        self.coefficient_of_drag = -1
        self.vehicle_curb_mass = -1
        self.mass_reduction = -1
        self.chassis_motor_mass_factor = -1
        self.chassis_battery_mass_factor = -1
        self.car_density = -1
        self.max_vehicle_weight_ratio = -1
        self.rolling_resistance_mass_factor = -1
        self.pit_time = -1
        self.maximum_allowable_vehicle_mass = -1
        self.frontal_area = -1
        self.net_chassis_mass = -1

    
    def return_dict_for_laptimesim(self):
        """Return general parameters in dictionary. Variables named
        in original TUM format."""
        parameters = {}
        parameters["lf"] = self.lf
        parameters["lr"] = self.lr
        parameters["h_cog"] = self.h_cog
        parameters["sf"] = self.sf
        parameters["sr"] = self.sr
        parameters["m"] = self.m
        parameters["f_roll"] = self.f_roll
        parameters["c_w_a"] = self.c_w_a
        parameters["c_z_a_f"] = self.c_z_a_f
        parameters["c_z_a_r"] = self.c_z_a_r
        parameters["g"] = self.g
        parameters["rho_air"] = self.rho_air
        parameters["drs_factor"] = self.drs_factor

        return parameters

    def return_dict_for_output_csv(self):
        """Return dictionary of parameters in the object.
        variable naming for output to csv."""
        parameters = {}
        parameters[LF_TAG] = self.lf
        parameters[LR_TAG] = self.lr
        parameters[H_COG_TAG] = self.h_cog
        parameters[SF_TAG] = self.sf
        parameters[SR_TAG] = self.sr
        parameters[M_TAG] = self.m
        parameters[F_ROLL_TAG] = self.f_roll
        parameters[C_W_A_LAPTIMESIM_TAG] = self.c_w_a
        parameters[C_Z_A_F_TAG] = self.c_z_a_f
        parameters[C_Z_A_R_TAG] = self.c_z_a_r
        parameters[GRAVITY_TAG] = self.g
        parameters[AIR_DENSITY_TAG] = self.rho_air
        parameters[DRS_FACTOR_TAG] = self.drs_factor
        parameters[COEFFICIENT_OF_DRAG_TAG] = self.coefficient_of_drag 
        parameters[VEHICLE_CURB_MASS] = self.vehicle_curb_mass
        parameters[MASS_REDUCTION_TAG] = self.mass_reduction
        parameters[CHASSIS_MOTOR_MASS_FACTOR_TAG] = self.chassis_motor_mass_factor
        parameters[CHASSIS_BATTERY_MASS_FACTOR_TAG] = self.chassis_battery_mass_factor
        parameters[CAR_DENSITY_TAG] = self.car_density
        parameters[MAX_VEHICLE_WEIGHT_RATIO_TAG] = self.max_vehicle_weight_ratio
        parameters[ROLLING_RESISTANCE_MASS_FACTOR_TAG] = self.rolling_resistance_mass_factor
        parameters[PIT_TIME_TAG] = self.pit_time
        parameters[MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG] = self.maximum_allowable_vehicle_mass
        parameters[FRONTAL_AREA_TAG] = self.frontal_area
        parameters[NET_CHASSIS_MASS_TAG] = self.net_chassis_mass

        return parameters

class batteryParameters():
    """Class for holding battery parameters. These variables are a mirror
    of what is in the config file and some additional params that are 
    calculated implicitly to the race car model."""
    def __init__(self):
        self.size = -1
        self.energy_density = -1
        self.change_constant = -1
        self.mass_pit_factor = -1
        self.power_output_factor = -1
        self.mass = -1
    
    def return_dict_for_output_csv(self):
        """Return dictionary of parameters in the object.
        variable naming for output to csv."""
        parameters = {}

        parameters[BATTERY_SIZE_TAG] = self.size
        parameters[BATTERY_ENERGY_DENSITY_TAG] = self.energy_density
        parameters[BATTERY_CHANGE_CONSTANT_TAG] = self.change_constant
        parameters[BATTERY_MASS_PIT_FACTOR_TAG] = self.mass_pit_factor
        parameters[BATTERY_POWER_OUTPUT_FACT0R_TAG] = self.power_output_factor
        parameters[BATTERY_MASS_TAG] = self.mass
        
        return parameters
class engineParameters():
    """Class for holding engine parameters. These variables are a mirror
    of what is in the config file and some additional params that are 
    calculated implicitly to the race car model."""
    def __init__(self):
        self.topology = -1
        self.eta_e_motor = -1
        self.eta_e_motor_re = -1
        self.motor_max_torque = -1
        self.motor_constant = -1
        self.motor_torque_density = -1
        self.motor_max_power = -1
        self.motor_mass = -1
    
    def return_dict_for_laptimesim(self):
        """Return engine parameters in dictionary. Variables named
        in original TUM format."""
        parameters = {}
        parameters["topology"] = self.topology
        parameters["pow_e_motor"] = self.motor_max_power
        parameters["eta_e_motor"] = self.eta_e_motor
        parameters["eta_e_motor_re"] = self.eta_e_motor_re
        parameters["torque_e_motor_max"] = self.motor_max_torque

        return parameters

    def return_dict_for_output_csv(self):
        """Return dictionary of parameters in the object.
        variable naming for output to csv."""
        parameters = {}
        parameters[TOPOLOGY_TAG] = self.topology
        parameters[POW_E_MOTOR_TAG] = self.motor_max_power
        parameters[MOTOR_EFFICICENCY_TAG] = self.eta_e_motor
        parameters[MOTOR_EFFICICENCY_REGEN_TAG] = self.eta_e_motor_re
        parameters[TORQUE_E_MOTOR_MAX_TAG] = self.motor_max_torque
        parameters[MOTOR_CONSTANT_TAG] = self.motor_constant
        parameters[MOTOR_TORQUE_DENSITY_TAG] = self.motor_torque_density
        parameters[MOTOR_MASS_TAG] = self.motor_mass

        return parameters

class gearboxParameters():
    """Class for holding gearbox parameters. These variables are a mirror
    of what is in the config file and some additional params that are 
    calculated implicitly to the race car model."""
    def __init__(self):
        self.i_trans = -1
        self.n_shift = -1
        self.e_i = -1
        self.eta_g = -1
    
    def return_dict_for_laptimesim(self):
        """Return dictionary of parameters. Parameters named in
        original TUM names."""
        parameters = {}
        parameters["i_trans"] = self.i_trans
        parameters["n_shift"] = self.n_shift
        parameters["e_i"] = self.e_i
        parameters["eta_g"] = self.eta_g

    def return_dict_for_output_csv(self):
        """Return dictionary of parameters in the object.
        variable naming for output to csv."""
        parameters = {}
        parameters[GEAR_RATIO_TAG] = self.i_trans
        parameters[MOTOR_SHIFT_RPM_TAG] = self.n_shift
        parameters[GEARBOX_TORSIONAL_MASS_FACTOR_TAG] = self.e_i
        parameters[GEARBOX_EFFICIENCY] = self.eta_g

        return parameters
class tireParameters():
    """Class for holding all tire parameters.
    These variables are a mirror
    of what is in the config file."""
    def __init__(self):
        self.tire_model_exp = tire_model_exp
        self.front_tires = tireParametersSingleAxle()
        self.rear_tires = tireParametersSingleAxle()
    
    def return_dict_for_laptimesim(self):
        """Return dictionary of parameters in the object.
        variable naming to match how the original TUM sim vars are named."""
        parameters = {}
        parameters["f"] = self.front_tires.return_dict_for_laptimesim()
        parameters["r"] = self.rear_tires.return_dict_for_laptimesim()
        parameters["tire_model_exp"] = self.tire_model_exp

        return parameters

    def return_dict_for_output_csv(self):
        """Return dictionary of parameters in the object.
        variable naming for output to csv."""
        parameters = {}
        front_tires = self.front_tires.return_dict_for_laptimesim()
        rear_tires = self.rear_tires.return_dict_for_laptimesim()

        parameters[REAR_TIRE_REFERENCE_CIRCUMFERENCE_TAG] = rear_tires["circ_ref"]
        parameters[REAR_TIRE_FZ_0_TAG] = rear_tires["fz_0"]
        parameters[REAR_TIRE_MUX_TAG] = rear_tires["mux"]
        parameters[REAR_TIRE_MUY_TAG] = rear_tires["muy"]
        parameters[REAR_TIRE_DMUX_DFZ_TAG] = rear_tires["dmux_dfz"]
        parameters[REAR_TIRE_DMUY_DFZ_TAG] = rear_tires["dmuy_dfz"]
        parameters[FRONT_TIRE_REFERENCE_CIRCUMFERENCE_TAG] = front_tires["circ_ref"]
        parameters[FRONT_TIRE_FZ_0_TAG] = front_tires["fz_0"]
        parameters[FRONT_TIRE_MUX_TAG] = front_tires["mux"]
        parameters[FRONT_TIRE_MUY_TAG] = front_tires["muy"]
        parameters[FRONT_TIRE_DMUX_DFZ_TAG] = front_tires["dmux_dfz"]
        parameters[FRONT_TIRE_DMUY_DFZ_TAG] = front_tires["dmuy_dfz"]
        parameters[TIRE_MODEL_EXPONENT_TAG] = self.tire_model_exp

        return parameters

class tireParametersSingleAxle():
    """Class for holding tire parameters for a single axle.
    These variables are a mirror
    of what is in the config file"""
    def __init__(self):
        self.circ_ref = -1
        self.fz_0 = -1
        self.mux = -1
        self.muy = -1
        self.dmux_dfz = -1
        self.dmuy_dfz = -1

    def return_dict_for_laptimesim(self):
        """Return parameters in the object in dictionary format with keys
        matching the laptimesim config names."""
        parameters = {}
        parameters["circ_ref"] = self.circ_ref
        parameters["fz_0"] = self.fz_0
        parameters["mux"] = self.mux
        parameters["muy"] = self.muy
        parameters["dmux_dfz"] = self.dmux_dfz
        parameters["dmuy_dfz"] = self.dmuy_dfz

        return parameters

class RaceCarModel():
    """Class that captures the relationship
    between the independent variables
    and dependent variables of a racecar in our model.

    This is the way this class is intended to be used:
    1. Set input, relationship, general, engine, gearbox, and tire variables accessing the values directly
    2. Calculate output variables by using the calculate_car_properties() function
    3. Get car properties with the get_*() functions, there is one for using in the lap sim, and another
    for when printing to results

    References for how output variables are calculated:
    https://docs.google.com/presentation/d/1Fe2ebMumncOxJ_XGSEA_o8PTu6qjORcLRfJFTbei6lo/edit#slide=id.p
    https://docs.google.com/document/d/1X2Aovz6VcKqkIUcsu5Z-QPbyDjBc7IG8CdWpZJr58eI/edit#
    
    
    """
    def __init__(self):
        """Initialization. Working with SI units here.
        """

        # Tracking Variables
        self._outputs_set = False
        self._is_allowable_weight = False

        # Emulate the veh_pars_ parameter structure with these datastores
        self.powertrain_type = "electric"  # always electric

        self.general_parameters = generalParameters()
        self.battery_parameters = batteryParameters()
        self.engine_parameters = engineParameters()
        self.gearbox_parameters = gearboxParameters()
        self.tire_parameters = tireParameters()

    def calculate_car_properties(self): 
        """Calculates if there is a rules legal car
        given the input variables and desired weight.

        This is the business of the relationships

        Inputs:
            Nothing
        
        Outputs:
            Nothing
        
        Raises:
            Exception: if the vehicle mass violates the rules (too much mass compared to the gvw)
        
        battery_mass (float): mass of battery in kg
        pit_time (float): pit time in minutes
        motor_mass (float): weight of motor in kg
        motor_max_power (float): maximum power output of motor in watts
        net_chassis_mass (float): weight of the chassis with all non-electric, non-racing stuff removed in kg
        total_vehicle_mass (float): total weight of the electric reaccar in kg
        maximum_allowable_vehicle_mass (float): maximum allowable weight of car based on rules in kg
        frontal_area (float): frontal area of car in meters squared
        is_allowable_weight (bool): bool represenging if the total vehicle weight is allowable
        c_w_a (float): variable representing the frontal area multiplied by c_d for the car
        rolling_resistance (float): rolling resistance for the car

        """

        self._outputs_set = True

        # Battery
        self.battery_parameters.battery_mass = (
            self.battery_parameters.battery_size /
            self.battery_parameters.battery_energy_density
        )

        self.general_parameters.pit_time = (
            self.battery_parameters.battery_mass *
            self.battery_parameters.battery_mass_pit_factor +
            self.battery_parameters.battery_change_constant
        )

        # Motor
        self.engine_parameters.motor_mass = (
            self.engine_parameters.motor_max_torque /
            self.engine_parameters.motor_torque_density
        )
        self.engine_parameters.motor_max_power = self._calculate_max_motor_power()

        self.engine_parameters.motor_max_power = self._calculate_max_motor_power()

        # Chassis
        self.general_parameters.net_chassis_mass = (
            self.general_parameters.vehicle_curb_mass - 
            self.general_parameters.mass_reduction
        )
        
        chassis_battery_mass = (
            self.battery_parameters.battery_mass *
            self.general_parameters.chassis_battery_mass_factor
        )
        
        chassis_motor_mass = (
            self.engine_parameters.motor_mass *
            self.general_parameters.chassis_motor_mass_factor
        )

        self.general_parameters.m = (
            self.general_parameters.net_chassis_mass +
            chassis_battery_mass +
            chassis_motor_mass +
            self.engine_parameters.motor_mass +
            self.battery_parameters.battery_mass
        )

        self.general_parameters.maximum_allowable_vehicle_mass = (
            self.general_parameters.vehicle_curb_mass *
            self.general_parameters.max_vehicle_weight_ratio
        )
        self.general_parameters.frontal_area = self._frontal_area_cube_calculation()

        self._is_allowable_weight = (
            self.general_parameters.m <
            self.general_parameters.maximum_allowable_vehicle_mass
        )

        if not self._is_allowable_weight:
            max_mass = self.general_parameters.maximum_allowable_vehicle_mass
            vehicle_mass = self.general_parameters.m
            raise(Exception("Vehicle Is over weight and out of rule spec. Max: {}, actual: {}"
                .format(max_mass, vehicle_mass)))

        # Drag
        self.dependent_variables.c_w_a = (
            self.general_parameters.frontal_area *
            self.general_parameters.coefficient_of_drag
        )

        self.general_parameters.f_roll = (
            self.general_parameters.m *
            self.general_parameters.rolling_resistance_mass_factor
        )

    def _calculate_max_motor_power(self):
        """Calculate the maximum motor power
        given a known motor constant and maximum motor torque.
        
        Reference: https://en.wikipedia.org/wiki/Motor_constants

        motor_constant = max_torque/sqrt(max_power)

        rearranging for max_power:
        max_power = (max_torque/motor_constant) ** 2

        Motor constant in N*m/sqrt(W)
        maximum torque in N*m
        
        Inputs:
            - self
        
        Outputs:
            - maximum_motor_power (float): maximum output power of motor in watts
        """

        maximum_motor_power = (
            self.engine_parameters.motor_max_torque/
            self.engine_parameters.motor_constant
        ) ** 2

        battery_power_limit = (
            self.battery_parameters.battery_size *
            self.battery_parameters.battery_power_output_factor
        )

        maximum_motor_power = min(
            maximum_motor_power,
            battery_power_limit
        )

        return maximum_motor_power
    
    def _frontal_area_cube_calculation(self):
        """Calculate the frontal area of the car based on the cube 
        assumption: the front of the car varies based on the mass of the 
        car where the car has a constant density and the car is shaped like a cube.

        Inputs:
            None
        
        Outputs:
            - frontal_area (float): frontal area of the car in meters squared
        
        Raises:
            Nothing
        
        """

        total_volume = (
            self.general_parameters.m /
            self.general_parameters.car_density
        )
        
        side_length = total_volume ** (1/3)

        frontal_area = side_length ** 2

        return frontal_area


    def get_vehicle_properties_for_csv_output(self):
        """Return car properites if all inputs and relationships are set,
        and outputs are calculated.

        This is the only intended public getter function for this class.

        Inputs:
            None
        
        Outputs:
            car_properties (dict): dictionary of car properties
        
        Raises:
            Exception if inputs, relatonships, or outputs have not been set
        
        """

        if not self._inputs_set or \
           not self._relationships_set or \
           not self._outputs_set:
            raise(Exception("Must set inputs, relationships, and calculate outputs before getting these values"))

        output = {}
        general_params = self.general_parameters.return_dict_for_output_csv()
        for key in general_params:
            output[key] = general_params[key]
        
        engine_params = self.engine_parameters.return_dict_for_output_csv()
        for key in engine_params:
            output[key] = engine_params[key]
        
        gearbox_params = self.gearbox_parameters.return_dict_for_output_csv()
        for key in gearbox_params:
            output[key] = gearbox_params[key]
        
        tire_params = self.tire_parameters.return_dict_for_output_csv()
        for key in tire_params:
            output[key] = tire_params[key]
        
        return output
    
    def get_car_parameters_for_laptimesim(self):
        """Returns the car parameters to be used in the lap simulation.
        These parameters should then be put into the laptimesim.src.car.Car object
        These parameters are the same as the original 'veh_pars_' section of the car config
        
        Inputs:
            - None
        
        Outpust:
            - veh_pars_ (dict): dictionary with same structure as `veh_pars_` part of 
            the car toml config
        
        Raises:
            - general exception if not all car parameters have been set or calculated
        """

        veh_pars_ = {}
        veh_pars_["powertrain_type"] = self.powertrain_type
        veh_pars_["general"] = self.general_parameters.return_dict_for_laptimesim()
        for key in veh_pars_["general"].get_keys():
            if veh_pars_["general"][key] == -1:
                raise(Exception("Item not set before retrieving it: general, {}".format(key)))

        veh_pars_["engine"] = self.engine_parameters.return_dict_for_laptimesim()
        for key in veh_pars_["engine"].keys():
            if veh_pars_["engine"][key] == -1:
                raise(Exception("Item not set before retrieving it: engine, {}".format(key)))
        
        veh_pars_["gearbox"] = self.gearbox_parameters.return_dict_for_laptimesim()
        for key in veh_pars_["gearbox"].keys():
            if veh_pars_["gearbox"][key] == -1:
                raise(Exception("Item not set before retrieving it: gearbox, {}".format(key)))

        veh_pars_["tires"] = self.tire_parameters.return_dict_for_laptimesim()
        for key in veh_pars_["tires"].keys():
            if veh_pars_["tires"][key] == -1:
                raise(Exception("Item not set before retrieving it: tires, {}".format(key)))
            if key == 'f' or key == 'r':
                for tire_key in veh_pars_["tires"][key].keys():
                    if veh_pars_["tires"][key][tire_key] == -1:
                        raise(Exception("Item not set before retrieving it: tires, {}, {}".format(key, tire_key)))

        return veh_pars_
    
    def set_params(self, input_vars):
        """Take input of flattened car config and set the
        parameter values to the correct value in the model
        
        inputs:
            - input_vars (dict): key is flattened paramter name from config, value is that value
        
        returns:
            - nothing
        
        raises:
            - keyError if not all values required are present
        """

        self.powertrain_type = input_vars["powertrain_type"]
        self.general_parameters.lf = input_vars["general.lf"]
        self.general_parameters.lr = input_vars["general.lr"]
        self.general_parameters.h_cog = input_vars["general.h_cog"]
        self.general_parameters.c_z_a_f = input_vars["general.c_z_a_f"]
        self.general_parameters.c_z_a_r = input_vars["general.c_z_a_r"]
        self.general_parameters.g = input_vars["general.g"]
        self.general_parameters.rho_air = input_vars["general.rho_air"]
        self.general_parameters.drs_factor = input_vars["general.drs_factor"]
        self.general_parameters.coefficient_of_drag = input_vars["general.coefficient_of_drag"]
        self.general_parameters.vehicle_curb_mass = input_vars["general.vehicle_curb_mass"]
        self.general_parameters.mass_reduction = input_vars["general.mass_reduction"]
        self.general_parameters.chassis_motor_mass_factor = input_vars["general.chassis_motor_mass_factor"]
        self.general_parameters.chassis_battery_mass_factor = input_vars["general.chassis_battery_mass_factor"]
        self.general_parameters.car_density = input_vars["general.car_density"]
        self.general_parameters.max_vehicle_weight_ratio = input_vars["general.max_vehicle_weight_ratio"]
        self.general_parameters.rolling_resistance_mass_factor = input_vars["general.rolling_resistanc_mass_factor"]

        self.battery_parameters.size = input_vars["battery.size"]
        self.battery_parameters.energy_density = input_vars["battery.energy_density"]
        self.battery_parameters.change_constant = input_vars["battery.change_constant"]
        self.battery_parameters.mass_pit_factor = input_vars["battery.mass_pit_factor"]
        self.battery_parameters.mass = input_vars["battery.mass"]

        self.engine_parameters.topology = input_vars["engine.topology"]
        self.engine_parameters.eta_e_motor = input_vars["engine.eta_e_motor"]
        self.engine_parameters.eta_e_motor_re = input_vars["engine.eta_e_motor_re"]
        self.engine_parameters.motor_max_torque = input_vars["engine.motor_max_torque"]
        self.engine_parameters.motor_constant = input_vars["engine.motor_constant"]
        self.engine_parameters.motor_torque_density = input_vars["engine.motor_torque_density"]

        self.gearbox_parameters.i_trans = input_vars["gearbox.i_trans"]
        self.gearbox_parameters.n_shift = input_vars["gearbox.n_shift"]
        self.gearbox_parameters.e_i = input_vars["gearbox.e_i"]
        self.gearbox_parameters.eta_g = input_vars["gearbox.eta_g"]

        self.tire_parameters.tire_model_exp = input_vars["tires.tire_model_exp"]
        
        self.tire_parameters.front_tires.circ_ref = input_vars["tires.f.circ_ref"]
        self.tire_parameters.front_tires.fz_0 = input_vars["tires.f.fz_0"]
        self.tire_parameters.front_tires.mux = input_vars["tires.f.mux"]
        self.tire_parameters.front_tires.muy = input_vars["tires.f.muy"]
        self.tire_parameters.front_tires.dmux_dfz = input_vars["tires.f.dumx_dfz"]
        self.tire_parameters.front_tires.dmuy_dfz = input_vars["tires.f.dmuy_dfz"]
        
        self.tire_parameters.rear_tires.circ_ref = input_vars["tires.r.circ_ref"]
        self.tire_parameters.rear_tires.fz_0 = input_vars["tires.r.fz_0"]
        self.tire_parameters.rear_tires.mux = input_vars["tires.r.mux"]
        self.tire_parameters.rear_tires.muy = input_vars["tires.r.muy"]
        self.tire_parameters.rear_tires.dmux_dfz = input_vars["tires.r.dmux_dfz"]
        self.tire_parameters.rear_tires.dmuy_dfz = input_vars["tires.r.dmuy_dfz"]
        


# Tests
if __name__ == '__main__':
    # do tests here
    race_car = RaceCarModel()

    race_car.set_inputs(
        battery_size=10,
        battery_change_constant=3,
        motor_max_torque=50,
        gross_vehicle_weight=1000,
        weight_reduction=200,
        coefficient_of_drag=0.35
    )

    race_car.set_relationship_variables(
        battery_energy_density=0.5,
        battery_mass_pit_factor=0.1,
        motor_constant=3,
        motor_torque_density=40,
        max_vehicle_weight_ratio=1.25,
        car_density=100,
        chassis_battery_mass_factor=0.01,
        chassis_motor_mass_factor=0.01,
        rolling_resistance_mass_factor=0.0001
    )
    
    race_car.calculate_car_properties()

    race_car_properties = race_car.get_vehicle_propreties()

    print(race_car_properties)