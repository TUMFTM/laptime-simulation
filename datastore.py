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
    GWC_TIMES_TAG, ITER_TAG, PIT_DRIVE_THROUGH_PENALTY_TIME, TOTAL_PITS_TAG, TOTAL_PITS_TAG, 
    VEHICLE_TAG, TOTAL_LAPS_TAG, LAPTIME_TAG,
    LAP_ENERGY_TAG, ENERGY_REMAINING_TAG,

    HEADER_ROW, WINNING_ELECTRIC_CAR_TAG, WINNING_GAS_CAR_LAPS
)

# Config keys that cannot be multiple values
# through the simulation, these can only be one
# value per simulation run because they are more not
# single number data types, they are strings or lists
EXCEPTION_KEYS = [
    "powertrain_type",
    "engine.topology",
    "gearbox.i_trans",
    "gearbox.n_shift",
    "gearbox.e_i",
]


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
            energy_remaining, is_winning_car_configuration
        ):
        """ Set results from the completed lap and set the
        iteration complete flag to allow accessing results.
        """

        self._iteration_complete = True

        race_car_properties = self.race_car_model.get_vehicle_properties()

        for key in race_car_properties:
            self._results_list[key] = race_car_properties[key]
        self._results_list[ITER_TAG] = self.iteration_number
        self._results_list[LAPTIME_TAG] = lap_time
        self._results_list[TOTAL_LAPS_TAG] = total_laps
        self._results_list[LAP_ENERGY_TAG] = energy_per_lap
        self._results_list[TOTAL_PITS_TAG] = total_pits
        self._results_list[ENERGY_REMAINING_TAG] = energy_remaining
        self._results_list[WINNING_ELECTRIC_CAR_TAG] = is_winning_car_configuration

    
    def get_results(self):
        if self._iteration_complete:
            return self._results_list
        else:
            raise(Exception("Iteration is not complete, must set results first"))


class DataStore():
    def __init__(self, results_file_name, track_pars, car_name):
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
            - track_pars (dict): dictionary of track parameters of a single track from track_pars.toml
            - car_name (str): name of car used in the simulation
        """

        self.add_results_lock = threading.Lock()
        results_file = open(results_file_name, "w", newline='')
        self.results_file_writer = csv.DictWriter(results_file,
                                                  fieldnames=HEADER_ROW,
                                                  )
        self.track_pars = track_pars
        self.car_name = car_name
        self.results_list = []

        self.results_file_writer.writeheader()

        self.input_data_ranges = {}
        self.single_iteration_data = {}
        self._total_iterations = -1
    
    def parse_car_config(self, car_config):
        """Function to parse the car properties dictionary
        from the config. 

        Inputs:
            - car_config (dict): dictionary of car properties, raw from config file
        
        Outputs: 
            None
        
        Raises:
            Nothing
        
        """

        # flatten dictionary. all keys are assumed to be the form: "section.parameter"
        # ex: general.lf or gearbox.n_shift
        flattened_car_config = car_config.flatten()
 
        # iterate over list passed in
        for key in flattened_car_config:
            if key in EXCEPTION_KEYS:
                # make a list
                self.input_data_ranges[key] = flattened_car_config[key]
                make sure this should be pass not continue
                pass

            # check type of the value, if its not a list its assumed
            # to be a single value and is then made into a list
            # that is in the same format as the varying lists
            if type(flattened_car_config[key]) is not list:
                self.input_data_ranges[key] = [float(flattened_car_config[key]),
                                               float(flattened_car_config[key]),
                                               1]
            # type is a list, just add to list
            else:
                # change type at index 2 for linspace operation later
                # in generate_unique_sa_combinations
                flattened_car_config[key][2] = int(flattened_car_config[key][2])
                self.input_data_ranges[key] = flattened_car_config[key]
        
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
        for key, value in self.input_data_ranges.items():
            if key in EXCEPTION_KEYS:
                # value is no a list, make a 1 entry list
                entry_explicit_values = [value]
            else:
                entry_explicit_values = np.linspace(value[0],
                                                    value[1],
                                                    value[2])
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

            race_car_model.set_params(input_vars)

            # Catch condition where the total mass of the vehicle
            # is too high and don't add to the iteration
            try:
                race_car_model.calculate_car_properties()
            except Exception as e:
                print("Exception in calculating car properties {}".format(e))
                continue

            iteration_data = SingleIterationData(iteration_number=i,
                                                 vehicle_name=self.car_name,
                                                 race_car_model=race_car_model
                                                 )
            self.single_iteration_data[i] = iteration_data
            self._total_iterations = i + 1 # enumerate is 0 based


    def set_single_iteration_results(
        self, iteration, lap_time, lap_energy, total_laps,
        total_pits, energy_remaining, is_winning_car_configuration
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
            - is_winning_car_configuration (bool): True if the electric car does more laps in the race
                than the winning gas car

        Outputs: Nothing
        
        Raises: Nothing
        """
        with self.add_results_lock:
            self.single_iteration_data[iteration].set_results(lap_time=lap_time,
                                                              energy_per_lap=lap_energy,
                                                              total_laps=total_laps,
                                                              energy_remaining=energy_remaining,
                                                              total_pits=total_pits,
                                                              is_winning_car_configuration=is_winning_car_configuration)

            # Write results to file
            # this should work magically because the csv writer is type DictWriter
            # all data should be in the properly labeled columns

            results = self.single_iteration_data[iteration].get_results()
            results[WINNING_GAS_CAR_LAPS] = self.track_pars[WINNING_GAS_CAR_LAPS]
            results[PIT_DRIVE_THROUGH_PENALTY_TIME] = self.track_pars[PIT_DRIVE_THROUGH_PENALTY_TIME]
            results[GWC_TIMES_TAG] = self.track_pars[GWC_TIMES_TAG]
            
            self.results_file_writer.writerow(results)
    
    def get_best_result(self):
        """Gets information about the result that 
        did the best after all of the simulations have run
        
        Inputs: 
            - None
        Outputs: 
            - best_results (dict): dictionary returned from SingleIterationData.get_results()
            - multiple_optimum_results (bool): True if there are multiple results that have the same 
                                               total number of laps
        Raises: 
            - Nothing
        """

        # Find best iteration

        max_laps = 0
        best_iteration = 0
        multiple_optimum_results = False

        for key in self.single_iteration_data:
            results = self.single_iteration_data[key].get_results()

            if results[TOTAL_LAPS_TAG] == max_laps:
                multiple_optimum_results = True
            if results[TOTAL_LAPS_TAG] > max_laps:
                best_iteration = results[ITER_TAG]
                max_laps = results[TOTAL_LAPS_TAG]
                multiple_optimum_results = False

        best_single_iteration = self.single_iteration_data[best_iteration]
        best_results = best_single_iteration.get_results()

        return best_results, multiple_optimum_results
