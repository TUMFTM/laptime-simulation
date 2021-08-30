# datastore for storing and outputting data, intended to be compatible with multiple threads
from typing import IO

from numpy.core.fromnumeric import var
from race_sim import RaceSim
from datetime import datetime
import threading
import csv
import numpy as np
import itertools
from race_car_model import (
    RaceCarModel
)

from definitions import (
    GWC_TIMES_TAG, ITER_TAG, TOTAL_PITS_TAG, TOTAL_PITS_TAG, 
    VEHICLE_TAG, TOTAL_LAPS_TAG, LAPTIME_TAG,
    LAP_ENERGY_TAG, ENERGY_REMAINING_TAG,

    REQUIRED_INPUTS, HEADER_ROW,
    INPUT_VARIABLES, RELATIONSHIP_VARIABLES
)



class SingleIterationData():
    """Class to hold all data related to a single iteration including
    inputs, results, and has the simulation been completed
    
    """
    def __init__(self, iteration_number, vehicle_name,
                 race_car_model: RaceCarModel):
        """Initialize data for a single iteration,
        set all inputs to the simulation
        
        """
        self._iteration_complete = False
        self.iteration_number = iteration_number
        self.race_car_model = race_car_model

        self._results_list = {VEHICLE_TAG: vehicle_name}

    def set_results(
            self, lap_time, total_laps, energy_per_lap, total_pits,
            gwc_times, energy_remaining
        ):
        """ Set results from the completed lap and set the
        iteration complete flag to allow accessing results.
        """

        self._iteration_complete = True

        self._results_list = self.race_car_model.get_vehicle_properties()
        self._results_list[ITER_TAG] = self.iteration_number
        self._results_list[LAPTIME_TAG] = lap_time
        self._results_list[TOTAL_LAPS_TAG] = total_laps
        self._results_list[LAP_ENERGY_TAG] = energy_per_lap
        self._results_list[TOTAL_PITS_TAG] = total_pits
        self._results_list[GWC_TIMES_TAG] = gwc_times
        self._results_list[ENERGY_REMAINING_TAG] = energy_remaining
        
    
    def get_results(self):
        if self._iteration_complete:
            return self._results_list
        else:
            raise(Exception("Iteration is not complete, must set results first"))


class DataStore():
    def __init__(self, results_file_name):
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

        Inputs:
            - results_file_name (str): file name for output file 
        """

        self.add_results_lock = threading.Lock()
        results_file = open(results_file_name, "w", newline='')
        self.results_file_writer = csv.DictWriter(results_file,
                                                  fieldnames=HEADER_ROW,
                                                  )
        self.results_list = []

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
        for key, value in input_variables.items():
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
        print(sa_opts_ranges)
        for key, value in sa_opts_ranges.items():
            if key not in REQUIRED_INPUTS:
                raise("invalid key passed in")
            if len(value) != 3:
                raise("Invalid length list")
            # create a "sa_range" list like other variables that
            # will only generate 1 value after calling a numpy.linspace on it
            self.input_data_ranges[key] = value
    
    def parse_car_properties(self, car_properties):
        """Function to parse the car properties dictionary
        from the config. 

        Inputs:
            - car_properties (dict): dictionary of car properties from config
                that has static and ranges of variables
        
        Outputs: 
            None
        
        Raises:
            Nothing
        
        """
 
        # iterate over list passed in
        for key in car_properties:

            # data passed in must be in the predefined list of
            # keys
            if key not in REQUIRED_INPUTS:
                raise(Exception("invalid key passed in {}".format(key)))

            # check type of the value, if its not a list its assumed
            # to be a single value and is then made into a list
            # that is in the same format as the varying lists
            if type(car_properties[key]) is not list:
                self.input_data_ranges[key] = [float(car_properties[key]),
                                               float(car_properties[key]),
                                               1]
            # type is a list, just add to list
            else:
                # change type at index 2 for linspace operation later
                # in generate_unique_sa_combinations
                car_properties[key][2] = int(car_properties[key][2])
                self.input_data_ranges[key] = car_properties[key]
        
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
                raise(
                    Exception("incorrect variables present in input data ranges, {} not present".format(key))
                )
        
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
        for key, sa_range in self.input_data_ranges.items():
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
            
            # The first value in each tuple comes from 
            # the first list that was passed in.
            # The first list that was passed in has the variable name 
            # in the first position of the list of the sa_opts_names list
            # and so on for the 2nd, 3rd, etc.
            input_vars = {}
            for j, variable_name in enumerate(sa_opts_names):
                input_vars[variable_name] = entry[j]
            
            # Make racecar property model and calculate parameters
            race_car_model = RaceCarModel()

            racecar_input_vars = {}
            for key in INPUT_VARIABLES:
                racecar_input_vars[key] = input_vars[key]
            race_car_model.set_inputs_dict(racecar_input_vars)

            racecar_relationship_vars = {}
            for key in RELATIONSHIP_VARIABLES:
                racecar_relationship_vars[key] = input_vars[key]
            
            race_car_model.set_relationship_variables_dict(racecar_relationship_vars)

            # Catch condition where the total mass of the vehicle
            # is too high and don't add to the iteration
            try:
                race_car_model.calculate_car_properties()
            except Exception as e:
                print("Exception in calculating car properties {}".format(e))
                continue

            iteration_data = SingleIterationData(iteration_number=i,
                                                 vehicle_name="FIXME",
                                                 race_car_model=race_car_model
                                                )
            self.single_iteration_data[i] = iteration_data
            self._total_iterations = i + 1 # enumerate is 0 based


    def set_single_iteration_results(
        self, iteration, lap_time, lap_energy, total_laps,
        total_pits, gwc_times, energy_remaining
    ):
        """Method to set results of a single lap.
        Saves to datastore and writes out to csv file
        
        Inputs:
            - iteration (int): iteration number of simulation, retreived from
            the data store before calculation
            - lap_time (float): lap time for current iteration in seconds
            - lap_energy (float): energy consumed for current iteration in kJ
            - total_laps (int): total laps completed over the whole race
            - total_pits (int): total number of pits in race,
            - gwc_times (list): green-white-checkered flag times
            - energy_remaining (float): sum of percentage of battery energy remaining in race

        Outputs: Nothing
        
        Raises: Nothing
        """
        with self.add_results_lock:
            self.single_iteration_data[iteration].set_results(lap_time=lap_time,
                                                              energy_per_lap=lap_energy,
                                                              total_laps=total_laps,
                                                              energy_remaining=energy_remaining,
                                                              total_pits=total_pits,
                                                              gwc_times=gwc_times)

            # Write results to file
            # this should work magically because the csv writer is type DictWriter
            # all data should be in the properly labeled columns

            results = self.single_iteration_data[iteration].get_results()
            self.results_file_writer.writerow(results)
