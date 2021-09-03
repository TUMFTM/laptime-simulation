# classes to store the relationship of race car properties
from definitions import (
    BATTERY_POWER_OUTPUT_FACT0R_TAG,
    MOTOR_TORQUE_DENSITY_TAG,
    BATTERY_MASS_TAG, PIT_TIME_TAG,
    MOTOR_MASS_TAG, MOTOR_MAX_POWER_TAG, NET_CHASSIS_MASS_TAG,
    TOTAL_VEHICLE_MASS_TAG, MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG,
    FRONTAL_AREA_TAG, BATTERY_SIZE_TAG, MOTOR_MAX_TORQUE_TAG,
    GROSS_VEHICLE_WEIGHT_TAG, WEIGHT_REDUCTION_TAG,
    COEFFICIENT_OF_DRAG_TAG, BATTERY_ENERGY_DENSITY_TAG,
    BATTERY_MASS_PIT_FACTOR_TAG, MOTOR_CONSTANT_TAG,
    MAX_VEHICLE_WEIGHT_RATIO_TAG, CAR_DENSITY_TAG,
    CHASSIS_BATTERY_MASS_FACTOR_TAG, CHASSIS_MOTOR_MASS_FACTOR_TAG,
    ROLLING_RESISTANCE_MASS_FACTOR_TAG, C_W_A_TAG,
    ROLLING_RESISTANCE_TAG, BATTERY_CHANGE_CONSTANT_TAG
)

class RaceCarModel():
    """Class that captures the relationship
    between the independent variables
    and dependent variables of a racecar in our model.

    This is the way this class is intended to be used:
    1. Set input variables using the set_inputs() funciton
    2. Set relationahip variables using the set_relationships() function
    3. Calculate output variables by using the calculate_car_properties() function
    4. Get car properties with the get_vehicle_properties() function

    References:
    https://docs.google.com/presentation/d/1Fe2ebMumncOxJ_XGSEA_o8PTu6qjORcLRfJFTbei6lo/edit#slide=id.p
    https://docs.google.com/document/d/1X2Aovz6VcKqkIUcsu5Z-QPbyDjBc7IG8CdWpZJr58eI/edit#
    
    
    """
    def __init__(self):
        """Initialization. Working with SI units here.
        """

        # Tracking Variables
        self._inputs_set = False
        self._relationships_set = False
        self._outputs_set = False
        self._is_allowable_weight = False

        self._race_car_properties = {}

    def set_inputs(self,
                   battery_size,
                   battery_change_constant,
                   motor_max_torque,
                   gross_vehicle_weight,
                   weight_reduction,
                   coefficient_of_drag
                   ):
        """Set the input/independent variables.

        Inputs:
            - battery_size (float): battery size in kWh
            - battery_change_constant (float): constant penalty for changing battery in minutes
            - motor_max_torque (float): maximum continuous motor torque in Nm
            - gross_vehicle_weight (float): vehicle weight from factory in kg
            - weight_reduction (float): weight reduced from removing unnecessary car components in kg
            - coefficient_of_drag (float): coefficient of drag of car (unitless)
        
        Outputs:
            None
        
        Raises:
            Nothing
        """

        self._inputs_set = True

        # Battery Inputs/Independent Variables
        self._race_car_properties[BATTERY_SIZE_TAG] = battery_size
        self._race_car_properties[BATTERY_CHANGE_CONSTANT_TAG] = battery_change_constant
    
        # Motor Inputs/Independent Variables
        self._race_car_properties[MOTOR_MAX_TORQUE_TAG] = motor_max_torque

        # Chassis Inputs/Independent Variables
        self._race_car_properties[GROSS_VEHICLE_WEIGHT_TAG] = gross_vehicle_weight
        self._race_car_properties[WEIGHT_REDUCTION_TAG] = weight_reduction

        # Drag Inputs/Indepdenent Variables
        self._race_car_properties[COEFFICIENT_OF_DRAG_TAG] = coefficient_of_drag

    def set_inputs_dict(self,
                   inputs_variables
                   ):
        """Set the input/independent variables. Uses a dictionary
        input structure instead of individual variable arguments

        Inputs:
            - inputs_variables (dict): dictionary of input variables.
            see the set_inputs function for input variable definitions.

        Outputs:
            None
        
        Raises:
            Nothing
        """
        self.set_inputs(
            battery_size=inputs_variables[BATTERY_SIZE_TAG],
            battery_change_constant=inputs_variables[BATTERY_CHANGE_CONSTANT_TAG],
            motor_max_torque=inputs_variables[MOTOR_MAX_TORQUE_TAG],
            gross_vehicle_weight=inputs_variables[GROSS_VEHICLE_WEIGHT_TAG],
            weight_reduction=inputs_variables[WEIGHT_REDUCTION_TAG],
            coefficient_of_drag=inputs_variables[COEFFICIENT_OF_DRAG_TAG]
        )

    def set_relationship_variables(self,
                                   battery_energy_density,
                                   battery_mass_pit_factor,
                                   battery_maximum_output_factor,
                                   motor_constant,
                                   motor_torque_density,
                                   max_vehicle_weight_ratio,
                                   car_density,
                                   chassis_battery_mass_factor,
                                   chassis_motor_mass_factor,
                                   rolling_resistance_mass_factor
                                   ):
        """Setting the relationship variables.

        Inputs:
            - battery_mass_pit_factor (float): how long to pit per kg of battery (minutes/kg)
            - battery_energy_density (float): energy density of battery kWh/kg
            - battery_maximum_output_factor (float): maximum output of battery pack's relation to size (W/kWh)
            - motor_constant (float): motor constant of motor Nm/sqrt(W)
            - motor_torque_density (float): torque per kilogram of motor (Nm/kg)
            - max_vehicle_weight_ratio (float): rules based maximum weight of vehicle based on gvw (unitless)
            - car_density (float): density of car (kg/m^3)
            - chassis_battery_mass_factor (float): ratio of supporting chassis mass per battery mass (kg/kg)
            - chassis_motor_mass_factor (float): ratio of supporting chassis mass per motor mass (kg/kg)
            - rolling_resistance_mass_factor (float): ratio of rolling resistance increase per kg of total vehicle mass (1/kg)
        
        Outputs:
            None
        
        Raises:
            Nothing
        """

        self._relationships_set = True

        # Battery Relationships
        self._race_car_properties[BATTERY_ENERGY_DENSITY_TAG] = battery_energy_density
        self._race_car_properties[BATTERY_MASS_PIT_FACTOR_TAG] = battery_mass_pit_factor
        self._race_car_properties[BATTERY_POWER_OUTPUT_FACT0R_TAG] = battery_maximum_output_factor

        # Motor Relationships
        self._race_car_properties[MOTOR_CONSTANT_TAG] = motor_constant
        self._race_car_properties[MOTOR_TORQUE_DENSITY_TAG] = motor_torque_density

        # Chassis Relationships
        self._race_car_properties[MAX_VEHICLE_WEIGHT_RATIO_TAG] = max_vehicle_weight_ratio
        self._race_car_properties[CAR_DENSITY_TAG] = car_density
        self._race_car_properties[CHASSIS_BATTERY_MASS_FACTOR_TAG] = chassis_battery_mass_factor
        self._race_car_properties[CHASSIS_MOTOR_MASS_FACTOR_TAG] = chassis_motor_mass_factor

        # Drag Relationships
        self._race_car_properties[ROLLING_RESISTANCE_MASS_FACTOR_TAG] = rolling_resistance_mass_factor

    def set_relationship_variables_dict(self,
                                   relationship_variables
                                   ):
        """Setting the relationship variables. Input is a dictionary
        of variables instead of adding individual values as arguments to the function

        Inputs:
            - relationship_variables (dict): dictionary of relationship variables
            see the set_relationship_variables function for definition of variables
            required.
        Outputs:
            None
        
        Raises:
            Nothing
        """

        self.set_relationship_variables(
            battery_energy_density=relationship_variables[BATTERY_ENERGY_DENSITY_TAG],
            battery_mass_pit_factor=relationship_variables[BATTERY_MASS_PIT_FACTOR_TAG],
            battery_maximum_output_factor=relationship_variables[BATTERY_POWER_OUTPUT_FACT0R_TAG],
            motor_constant=relationship_variables[MOTOR_CONSTANT_TAG],
            motor_torque_density=relationship_variables[MOTOR_TORQUE_DENSITY_TAG],
            max_vehicle_weight_ratio=relationship_variables[MAX_VEHICLE_WEIGHT_RATIO_TAG],
            car_density=relationship_variables[CAR_DENSITY_TAG],
            chassis_battery_mass_factor=relationship_variables[CHASSIS_BATTERY_MASS_FACTOR_TAG],
            chassis_motor_mass_factor=relationship_variables[CHASSIS_MOTOR_MASS_FACTOR_TAG],
            rolling_resistance_mass_factor=relationship_variables[ROLLING_RESISTANCE_MASS_FACTOR_TAG]
        )

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
        self._race_car_properties[BATTERY_MASS_TAG] = (
            self._race_car_properties[BATTERY_SIZE_TAG] /
            self._race_car_properties[BATTERY_ENERGY_DENSITY_TAG]
        )

        self._race_car_properties[PIT_TIME_TAG] = (
            self._race_car_properties[BATTERY_MASS_TAG] *
            self._race_car_properties[BATTERY_MASS_PIT_FACTOR_TAG] +
            self._race_car_properties[BATTERY_CHANGE_CONSTANT_TAG]
        )

        # Motor
        self._race_car_properties[MOTOR_MASS_TAG] = (
            self._race_car_properties[MOTOR_MAX_TORQUE_TAG] /
            self._race_car_properties[MOTOR_TORQUE_DENSITY_TAG]
        )
        self._race_car_properties[MOTOR_MAX_POWER_TAG] = self._calculate_max_motor_power()

        # Chassis
        self._race_car_properties[NET_CHASSIS_MASS_TAG] = (
            self._race_car_properties[GROSS_VEHICLE_WEIGHT_TAG] -
            self._race_car_properties[WEIGHT_REDUCTION_TAG]
        )
        
        chassis_battery_mass = (
            self._race_car_properties[BATTERY_MASS_TAG] *
            self._race_car_properties[CHASSIS_BATTERY_MASS_FACTOR_TAG]
        )
        
        chassis_motor_mass = (
            self._race_car_properties[MOTOR_MASS_TAG] *
            self._race_car_properties[CHASSIS_MOTOR_MASS_FACTOR_TAG]
        )

        self._race_car_properties[TOTAL_VEHICLE_MASS_TAG] = (
            self._race_car_properties[NET_CHASSIS_MASS_TAG] +
            chassis_battery_mass + 
            chassis_motor_mass +
            self._race_car_properties[MOTOR_MASS_TAG] +
            self._race_car_properties[BATTERY_MASS_TAG]
        )

        self._race_car_properties[MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG] = (
            self._race_car_properties[GROSS_VEHICLE_WEIGHT_TAG] *
            self._race_car_properties[MAX_VEHICLE_WEIGHT_RATIO_TAG]
        )
        self._race_car_properties[FRONTAL_AREA_TAG] = self._frontal_area_cube_calculation()
        
        self._is_allowable_weight = (
            self._race_car_properties[TOTAL_VEHICLE_MASS_TAG] <
            self._race_car_properties[MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG]
        )

        if not self._is_allowable_weight:
            max_mass = self._race_car_properties[MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG]
            vehicle_mass = self._race_car_properties[TOTAL_VEHICLE_MASS_TAG]
            raise(Exception("Vehicle Is over weight and out of rule spec. Max: {}, actual: {}"
                .format(max_mass, vehicle_mass)))

        # Drag
        self._race_car_properties[C_W_A_TAG] = (
            self._race_car_properties[FRONTAL_AREA_TAG] *
            self._race_car_properties[COEFFICIENT_OF_DRAG_TAG]
        )
        self._race_car_properties[ROLLING_RESISTANCE_TAG] = (
            self._race_car_properties[TOTAL_VEHICLE_MASS_TAG] *
            self._race_car_properties[ROLLING_RESISTANCE_MASS_FACTOR_TAG]
        )

    def get_vehicle_properties(self):
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

        return self._race_car_properties

    
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
            self._race_car_properties[MOTOR_MAX_TORQUE_TAG] /
            self._race_car_properties[MOTOR_CONSTANT_TAG]
        ) ** 2

        battery_power_limit = (
            self._race_car_properties[BATTERY_SIZE_TAG] *
            self._race_car_properties[BATTERY_POWER_OUTPUT_FACT0R_TAG]
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
            self._race_car_properties[TOTAL_VEHICLE_MASS_TAG] /
            self._race_car_properties[CAR_DENSITY_TAG]
        )
        
        side_length = total_volume ** (1/3)

        frontal_area = side_length ** 2

        return frontal_area


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