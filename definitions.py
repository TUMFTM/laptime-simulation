# Constants to use for data tags in the simulation

# Inputs
BATTERY_SIZE_TAG = "Battery Size (kWh)"
BATTERY_CHANGE_CONSTANT_TAG = "Battery Change Constant (minutes)"
MOTOR_MAX_TORQUE_TAG = "Maximum Motor Torque (Nm)"
VEHICLE_CURB_MASS = "Vehicle Curb Mass (kg)"
MASS_REDUCTION_TAG = "Mass Reduction (kg)"
COEFFICIENT_OF_DRAG_TAG = "Coefficient of Drag (unitless)"

GWC_TIMES_TAG = "gwc times (minutes)"
WINNING_GAS_CAR_LAPS = "Winning Gas Car Laps"
PIT_DRIVE_THROUGH_PENALTY_TIME = "Time to drive through pits (minutes)"

# Relationships
BATTERY_ENERGY_DENSITY_TAG = "Battery Energy Density (kWh/kg)"
BATTERY_MASS_PIT_FACTOR_TAG = "Battery Mass Pit Factor (minutes/kg)"
BATTERY_POWER_OUTPUT_FACT0R_TAG = "Battery Output Factor (W/kWh)"
MOTOR_CONSTANT_TAG = "Motor Constant (Nm/sqrt(W))"
MOTOR_TORQUE_DENSITY_TAG = "Motor Torque Density (Nm/kg)"
MAX_VEHICLE_WEIGHT_RATIO_TAG = "Max Vehicle Weight Ratio (unitless)"
CAR_DENSITY_TAG = "Car Density (kg/m^3)"
CHASSIS_BATTERY_MASS_FACTOR_TAG = "Chassis Battery Mass Factor (kg/kg)"
CHASSIS_MOTOR_MASS_FACTOR_TAG = "Chassis Motor Mass Factor (kg/kg)"
ROLLING_RESISTANCE_MASS_FACTOR_TAG = "Rolling Resistance Mass Factor (1/kg)"

# Other car related parameters
LF_TAG = "Front-back distance from front axle to center of gravity (meters)"
LR_TAG = "Front-back distance from rear axle to center of gravity (meters)"
H_COG_TAG = "Height of center of gravity above ground (meters)"
SF_TAG = "track width front (meters)"
SR_TAG = "track width rear (meters)"
M_TAG = "mass of car in laptimesim (kg)"
F_ROLL_TAG = "Rolling resistance in laptimesim"
C_W_A_LAPTIMESIM_TAG = "c_w_a use in laptimesim (m^2)"
C_Z_A_F_TAG = "coefficient of downforce * wing area of front wing (meters^2)"
C_Z_A_R_TAG = "coefficient of downforce * wing area of rear wing (meters^2)"
GRAVITY_TAG = "gravity (m/s^2)"
AIR_DENSITY_TAG = "air density (kg/m^3)"
DRS_FACTOR_TAG = "reduction of air resistance due to drs (percent)"
TOPOLOGY_TAG = "drive configuration of car: fwd, rwd, or awd"
POW_E_MOTOR_TAG = "Maximum motor power used in laptimesim (Watts)"
MOTOR_EFFICICENCY_TAG = "Efficiency of motor (%)"
MOTOR_EFFICICENCY_REGEN_TAG = "Efficiency of motor in regeneration (%)"
TORQUE_E_MOTOR_MAX_TAG = "Max torque of motor used in laptimesim (Nm)"
GEAR_RATIO_TAG = "List of gear ratios, output rev/input rev"
MOTOR_SHIFT_RPM_TAG = "gear shift rpm for gearbox (rpm)"
GEARBOX_TORSIONAL_MASS_FACTOR_TAG = "Torsional mass factor of gearbox per gear"
GEARBOX_EFFICIENCY = "Efficiency of gearbox (%)"
TIRE_MODEL_EXPONENT_TAG = "Tire model exponent, used in tire model"
REAR_TIRE_REFERENCE_CIRCUMFERENCE_TAG = "rear tire loaded reference circumference (meters)"
REAR_TIRE_FZ_0_TAG = "rear tire nominal tire load (N)"
REAR_TIRE_MUX_TAG = "rear tire coefficient of friction at nominal tire load mux"
REAR_TIRE_MUY_TAG = "rear tire coefficient of friction at nominal tire load muy"
REAR_TIRE_DMUX_DFZ_TAG = "rear tire reduction of force potential with rising tire load x direction"
REAR_TIRE_DMUY_DFZ_TAG = "rear tire reduction of force potential with rising tire load y direction"
FRONT_TIRE_REFERENCE_CIRCUMFERENCE_TAG = "front tire loaded reference circumference (meters)"
FRONT_TIRE_FZ_0_TAG = "front tire nominal tire load (N)"
FRONT_TIRE_MUX_TAG = "front tire coefficient of friction at nominal tire load mux"
FRONT_TIRE_MUY_TAG = "front tire coefficient of friction at nominal tire load muy"
FRONT_TIRE_DMUX_DFZ_TAG = "front tire reduction of force potential with rising tire load x direction"
FRONT_TIRE_DMUY_DFZ_TAG = "front tire reduction of force potential with rising tire load y direction"

# Results/outputs
ITER_TAG = "Iteration"
VEHICLE_TAG = "Vehicle Name"
TOTAL_LAPS_TAG = "Total Laps Completed"
LAPTIME_TAG = "Laptime (s)"
LAP_ENERGY_TAG = "Engery Per Lap (kJ)"
BATTERY_MASS_TAG = "Battery Mass (kg)"
PIT_TIME_TAG = "Pit Time (minutes)"
TOTAL_PITS_TAG = "total Number of Pits"
MOTOR_MASS_TAG = "Motor Mass (kg)"
MOTOR_MAX_POWER_TAG = "Motor Maximum Power (W)"
NET_CHASSIS_MASS_TAG = "Net Chassis Mass (kg)"
TOTAL_VEHICLE_MASS_TAG = "Total Vehicle Mass (kg)"
MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG = "Maximum Allowable Vehicle Mass (kg)"
FRONTAL_AREA_TAG = "Frontal Area (m^2)"
C_W_A_RACE_CAR_MODEL_TAG = "c_w_a calculated by race car model (m^2)"
ROLLING_RESISTANCE_TAG = "Rolling Resistance Coefficient (unitless)"

ENERGY_REMAINING_TAG = "Energy Remaining (sum of %)"

WINNING_ELECTRIC_CAR_TAG = "Does this electric car win? (bool)"

HEADER_ROW = [
    ITER_TAG, VEHICLE_TAG, TOTAL_LAPS_TAG, LAPTIME_TAG,
    LAP_ENERGY_TAG, BATTERY_MASS_TAG, PIT_TIME_TAG,
    TOTAL_PITS_TAG, ENERGY_REMAINING_TAG,
    MOTOR_MASS_TAG, MOTOR_MAX_POWER_TAG, NET_CHASSIS_MASS_TAG,
    TOTAL_VEHICLE_MASS_TAG, MAXIMUM_ALLOWABLE_VEHICLE_MASS_TAG,
    FRONTAL_AREA_TAG, BATTERY_SIZE_TAG, MOTOR_MAX_TORQUE_TAG,
    VEHICLE_CURB_MASS, MASS_REDUCTION_TAG,
    COEFFICIENT_OF_DRAG_TAG, BATTERY_ENERGY_DENSITY_TAG,
    BATTERY_MASS_PIT_FACTOR_TAG, MOTOR_CONSTANT_TAG,
    MOTOR_TORQUE_DENSITY_TAG,
    MAX_VEHICLE_WEIGHT_RATIO_TAG, CAR_DENSITY_TAG,
    CHASSIS_BATTERY_MASS_FACTOR_TAG, CHASSIS_MOTOR_MASS_FACTOR_TAG,
    ROLLING_RESISTANCE_MASS_FACTOR_TAG, C_W_A_RACE_CAR_MODEL_TAG,
    ROLLING_RESISTANCE_TAG, GWC_TIMES_TAG, BATTERY_POWER_OUTPUT_FACT0R_TAG,
    WINNING_ELECTRIC_CAR_TAG, PIT_DRIVE_THROUGH_PENALTY_TIME, BATTERY_CHANGE_CONSTANT_TAG,
    WINNING_GAS_CAR_LAPS, LF_TAG, LR_TAG, H_COG_TAG, SF_TAG, SR_TAG, C_Z_A_F_TAG, C_Z_A_R_TAG,
    GRAVITY_TAG, AIR_DENSITY_TAG, DRS_FACTOR_TAG, TOPOLOGY_TAG, MOTOR_EFFICICENCY_REGEN_TAG, 
    MOTOR_EFFICICENCY_TAG, GEAR_RATIO_TAG, MOTOR_SHIFT_RPM_TAG, GEARBOX_TORSIONAL_MASS_FACTOR_TAG,
    GEARBOX_EFFICIENCY, TIRE_MODEL_EXPONENT_TAG, REAR_TIRE_REFERENCE_CIRCUMFERENCE_TAG,
    REAR_TIRE_FZ_0_TAG, REAR_TIRE_MUX_TAG, REAR_TIRE_MUY_TAG, REAR_TIRE_DMUX_DFZ_TAG,
    REAR_TIRE_DMUY_DFZ_TAG, FRONT_TIRE_REFERENCE_CIRCUMFERENCE_TAG, 
    FRONT_TIRE_FZ_0_TAG, FRONT_TIRE_MUX_TAG, FRONT_TIRE_MUY_TAG, FRONT_TIRE_DMUX_DFZ_TAG,
    FRONT_TIRE_DMUY_DFZ_TAG, F_ROLL_TAG, M_TAG
]
