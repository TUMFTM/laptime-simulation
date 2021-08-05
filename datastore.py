# datastore for storing and outputting data, intended to be compatible with multiple threads
from typing import IO

from numpy.core.fromnumeric import var
from race_sim import RaceSim
from datetime import datetime
import threading
import csv
import numpy as np
import itertools

TOTAL_LAPS_TAG = "total laps"
ITER_TAG = "iteration"
VEHICLE_TAG = "vehicle"
MASS_TAG = "mass (kg)"
MAX_MOTOR_TORQUE_TAG = "max motor torque (Nm)"
C_D_TAG = "Cd(c_w_a)"
LAPTIME_TAG = "laptime (s)"
LAP_ENERGY_TAG = "energy (kJ)"
HEADER_ROW = [ITER_TAG, VEHICLE_TAG, MASS_TAG,
                C_D_TAG, MAX_MOTOR_TORQUE_TAG, 
                LAPTIME_TAG, LAP_ENERGY_TAG, TOTAL_LAPS_TAG]
REQUIRED_INPUTS = [MASS_TAG, MAX_MOTOR_TORQUE_TAG, C_D_TAG]

class SingleIterationData():
    """Class to hold all data related to a single iteration including
    inputs, results, and has the simulation been completed
    
    """
    def __init__(self, iteration_number, vehicle_name,
                 vehicle_mass, vehicle_c_d, vehicle_max_torque):
        """Initialize data for a single iteration,
        set all inputs to the simulation
        
        """
        self._iteration_complete = False
        self.iteration_number = iteration_number
        self.vehicle_name = vehicle_name
        self.vehicle_mass = vehicle_mass
        self.vehicle_max_torque = vehicle_max_torque
        self.vehicle_c_d = vehicle_c_d
        self.lap_time = -1
        self.total_laps = -1
        self.energy_per_lap = -1
        self.results_list = {}  # formatting to output to file

    def set_results(self, lap_time, total_laps, energy_per_lap):
        """ Set results from the completed lap and set the
        iteration complete flag to allow accessing results.
        """

        self._iteration_complete = True
        self.lap_time = lap_time
        self.total_laps = total_laps
        self.energy_per_lap = energy_per_lap
        
        self.results_list = {
            ITER_TAG: self.iteration_number,
            VEHICLE_TAG: self.vehicle_name,
            MASS_TAG: self.vehicle_mass,
            MAX_MOTOR_TORQUE_TAG: self.vehicle_max_torque,
            C_D_TAG: self.vehicle_c_d,
            LAPTIME_TAG: lap_time,
            LAP_ENERGY_TAG: energy_per_lap,
            TOTAL_LAPS_TAG: total_laps
        }
    
    def get_results(self):
        if self._iteration_complete:
            return self.results_list
        else:
            raise("Iteration is not complete, must set results first")


class dataStore():
    def __init__(self):
        """ Initialize datastore and start output file
        
        This datastore is intended to be the main interaction to
        - generate all combinations of simulations that should be completed
        - store all results
        - output results to file
        - present results in graphing format

        The pieces of data that are required to interact with this datastore
        are listed in the HEADER_ROW list. This list is used to check that all data
        is present for simulations to start, to output to the results file, for variable names
        and for keeping all data straight.

        Data for a single iteration is stored in the class SingleIterationData

        All pieces of data in the REQUIRED_INPUTS list must be added
        to the datastore before running any simulation

        Usage:
        1. initialize datastore
        2. add static and sa_range variables, all variables in REQUIRED_INPUTS 
        list must be present before moving to next step
        3. create all input variable combinations 
        4. Run simulations by accessing each iterations data in SingleIterationData class,
        add results data back to SingleIterationData class
        """

        results_filename = "./race-sim-results-{}".format(datetime.now.utc())

        self.add_results_lock = threading.lock()
        results_file = open(results_filename, "w")
        self.results_file_writer = csv.DictWriter(results_file,
                                                  fieldnames=HEADER_ROW)
        self.results_list = []

        with self.results_file_lock:
            self.results_file_writer.writeheader()

        self.input_data_ranges = {}
        self.single_iteration_data = {}
        self._total_iterations = -1
        

    def add_static_input_variables(self, input_variables):
        """Function for adding in simulation variables that does not
        vary through each iteration. For each variable passed in it:
            - validates that it is a valid variable name (must be in REQUIRED_INPUT list)
            - adds an entry to the input_data_ranges dict that has the same format as variables
            that will change to make generating all unique combinations of variables easier later

        Inputs:
            - input_variables (dict): 
                - key: name of variable being added
                - value: value of the variable of name key in the simulation
        
        Outputs:
            - none
        
        Raises:
            - Exception on invalid data names being passed in
        
        """
        for key, value in input_variables:
            if key not in REQUIRED_INPUTS:
                raise("invalid key passed in")
            # create a "sa_range" list like other variables that
            # will only generate 1 value after calling a numpy.linspace on it
            self.input_data_ranges[key] = [value,
                                           value,
                                           1]
    
    def add_sa_input_variables(self, sa_opts_ranges):
        """Function for adding in simulation variables that do
        vary through each iteration. For each variable passed in it:
            - validates that it is a valid variable name (must be in REQUIRED_INPUT list)
            - validates that each value 
            - adds an entry to the input_data_ranges dict

        Inputs:
            - sa_opts_ranges (dict): dict of variables to iterate over.
            Must be at least 1 variable, can be any number of supported variables.
            Key is name of variable, value is list of 3 elements: 
            [min_value, max_value, number_of_steps]. This is the same as laid out
            in the sa_opts part of the optimization file
                This is the same format as laid out in the sim_config.toml
        
        Outputs:
            - none
        
        Raises:
            - Exception on invalid data names being passed in
        
        """
        for key, value in sa_opts_ranges:
            if key not in REQUIRED_INPUTS:
                raise("invalid key passed in")
            if len(value) != 3:
                raise("Invalid length list")
            # create a "sa_range" list like other variables that
            # will only generate 1 value after calling a numpy.linspace on it
            self.input_data_ranges[key] = value
        
    def generate_unique_sa_combinations(self):
        """ Function that generates all unique combinations of 
        sensitivity analysis variables.

        This is intended to be called just before running the simulation

        Inputs:
            - None
        
        Outputs:
            - None, sets variables internal to the datastore
        
        Raises:
            - Exception if there is an unsupported iteration variable or not all variables
            are present

        """

        # validate keys all keys are present before creating unique combinations
        for key in REQUIRED_INPUTS:
            if key not in self.input_data_ranges.keys():
                raise("incorrect variables present in input data ranges")
        
        # turn iteration variables into a list of unique 
        # simulation conditions to iterate through
        # 1. List all unique values for each sensitivity analysis (sa) variable
        # 2. Create all unique combinations with all variables
        # approach based on this stack overflow post:
        # https://stackoverflow.com/questions/798854/all-combinations-of-a-list-of-lists
        
        # 3. Re associate each resulting piece of data from 2 to a vairable name so 
        # the data can be explicitly added to the single iteration result class

        # 1
        sa_opts_explicit_values = []
        sa_opts_names = []
        for key, sa_range in self.input_data_ranges:
            entry_explicit_values = np.linspace(sa_range[0],
                                                sa_range[1],
                                                sa_range[2])
            sa_opts_names.append(key)
            sa_opts_explicit_values.append(entry_explicit_values)

        # 2
        """The format of variable_combinations will be as follows:
        - overall data structure is a list of tuples
        - each tuple has length n, where n is the number of sa_opts variables.
        i.e. if the only sa_opt is mass then each tuple will have a length of 1.
        If there are 3 sa_opts (mass, c_d, max_torque) then each tuple will have 3 elements

        The associated variable name of each tuple is the same index of sa_opts_names
        """
        variable_combinations = list(itertools.product(*sa_opts_explicit_values))

        # Add all unique combinations 
        for i, entry in enumerate(variable_combinations):

            # 3. remap each variable in the entry tuple to have a variable
            # name from sa_opts_names to the values can be passed
            # explicitly to the SingleIterationData class later
            # This is possible because of the way the itertools.product 
            # function works.
            # 
            # The first value in each tuple comes from 
            # the first list that was passed in.
            # The first list that was passed in has the variable name 
            # in the first position of the list of the sa_opts_names list
            # and so on for the 2nd, 3rd, etc.
            input_vars = {}
            for j, variable_name in enumerate(sa_opts_names):
                input_vars[variable_name] = entry[j]

            iteration_data = SingleIterationData(iteration=i,
                                                 vehicle_name="FIXME",
                                                 vehicle_mass=input_vars[MASS_TAG],
                                                 vehicle_c_d=input_vars[C_D_TAG],
                                                 vehicle_max_torque=input_vars[MAX_MOTOR_TORQUE_TAG]
                                                )
            self.single_iteration_data[i] = iteration_data
            self._total_iterations = i


    def set_single_iteration_results(self, iteration, lap_time, lap_energy, total_laps):
        """Method to set results of a single lap.
        Saves to datastore and writes out to csv file
        
        Inputs:
            - iteration (int): iteration number of simulation, retreived from
            the data store before calculation
            - lap_time (float): lap time for current iteration in seconds
            - lap_energy (float): energy consumed for current iteration in kJ
            - total_laps (int): total laps completed over the whole race

        Outputs: Nothing
        
        Raises: Nothing
        """
        self.single_iteration_data[iteration].set_results(lap_time=lap_time,
                                                          energy_per_lap=lap_energy,
                                                          total_laps=total_laps)

        # Write results to file
        # this should work magically because the csv writer is type DictWriter
        # all data should be in the properly labeled columns
        with self.add_results_lock:
            results = self.single_iteration_data[iteration].get_results()
            self.results_file_writer.writerow(results)
            
            
        # write results to results list

    def get_graph_data(self):
        """ Method to return data that is graphable
        by matplotlib. (needs to interface with what
        we have there now)

        All simulation iterations must be completed for this to work

        """

        sa_t_lap = np.zeros(self._total_iterations)
        sa_fuel_cons = np.zeros(self._total_iterations)
        sa_iter = np.zeros(self._total_iterations)
        sa_mass = np.zeros(self._total_iterations)
        sa_c_d = np.zeros(self._total_iterations)
        sa_torque = np.zeros(self._total_iterations)
        sa_total_laps = np.zeros(self._total_iterations)

        for key, single_iteration_data in self.single_iteration_data:
            iteration_results = single_iteration_data.get_results()

            sa_t_lap[key] = iteration_results[LAPTIME_TAG]
            sa_fuel_cons[key] = iteration_results[LAP_ENERGY_TAG]
            sa_iter[key] = iteration_results[ITER_TAG]
            sa_mass[key] = iteration_results[MASS_TAG]
            sa_c_d[key] = iteration_results[C_D_TAG]
            sa_torque[key] = iteration_results[MAX_MOTOR_TORQUE_TAG]
            sa_total_laps[key] = iteration_results[TOTAL_LAPS_TAG]
        
        return sa_t_lap, sa_fuel_cons, sa_iter, sa_mass, sa_c_d, sa_torque, sa_total_laps

