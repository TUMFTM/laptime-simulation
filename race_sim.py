# helper classes and objects to make race calculations

class RaceDayInfo():
    """Class to contain information about
    one day of a race.
    
    race_time [m] time to race on day
    total_laps [-] total laps on day
    number_of_pits [-] number of pits on day
    time_of_pits [m] minutes after start for start of pit
    energy_remaining [%] percent of charge left at end of day

    """
    def __init__(self, race_time):
        self.race_time = race_time
        self.total_laps = 0
        self.number_of_pits = 0
        self.time_of_pits = []
        self.energy_remaining = 0


class RaceSim():
    """The racesim takes in calculated lap time and energy used per lap
    along with an estimated pit time and time on track to calculate
    a number of laps over the course of a race.
    
    On all race days the battery is assumed to start fully charged.
 
    """
    def __init__(self, pit_penalty,
                 race_time_day_1,
                 race_time_day_2,
                 lap_time,
                 energy_per_lap,
                 battery_capacity):
        """
        Inputs:
            - pit_pentalty (float): time to pit and change batteries in minutes
            - race_time_day_1 (float): time to race day 1 in minutes
            - race_time_day_2 (float): time to race day 2 in minutes
            - lap_time (float): time to complete one lap in seconds (output from laptime sim)
            - energy_per_lap (float): energy consumed to complete one lap in joules (output from laptime sim) 
            - battery_capacity (float): energy capacity of battery in kWh (car property)

        """

        self.race_days = [RaceDayInfo(race_time_day_1), RaceDayInfo(race_time_day_2)]
        
        self._pit_penalty = pit_penalty
        self._lap_time_minutes = lap_time / 60

        self._energy_per_lap = energy_per_lap
        self._battery_capacity_kwh = battery_capacity
        self._battery_capacity_joules = self._battery_capacity_kwh*3.6e6

        # floor division
        self.laps_per_battery = self._battery_capacity_joules // self._energy_per_lap
        self.left_over_energy = self._battery_capacity_joules - (self.laps_per_battery * 
                                self._energy_per_lap)
        
        self.race_time_per_battery = self.laps_per_battery * self._lap_time_minutes

    def calculate(self):
        """Calculate the number of laps completed and other
        numbers over all race days.
        """
        # do this calculation over all race days
        for race_day in self.race_days:

            accumulated_time = 0

            # go until the total time racing and doing pit stops is > than 
            # the race time
            while accumulated_time < race_day.race_time:
                possible_laps = (race_day.race_time - accumulated_time) // \
                                (self._lap_time_minutes)

                # If possible laps < laps per battery the race is near the end 
                # of the day and no more battery swaps are necessary. The race
                # day can be finished on one battery pack. Add stuff up and
                # exit the loop
                if possible_laps < self.laps_per_battery:
                    race_day.total_laps += possible_laps
                    race_day.energy_remaining = (possible_laps * self._energy_per_lap) / \
                                                 self._battery_capacity_joules
                    break
                else:
                    # still in the middle of the race.
                    # a pit stop must be completed to get more energy in the 
                    # car to keep going
                    race_day.total_laps += self.laps_per_battery

                    accumulated_time += self.race_time_per_battery
                    race_day.time_of_pits.append(accumulated_time)
                    accumulated_time += self._pit_penalty
                    race_day.number_of_pits += 1

# Testing
if __name__ == '__main__':
    lap_time = 55  # Joules
    battery_capacity = 50  # kWh
    energy_per_lap = 5.4e6  # 1.5kWh/lap
    race = RaceSim(pit_penalty=5,
                   race_time_day_1=360,
                   race_time_day_2=240,
                   lap_time=lap_time,
                   energy_per_lap=energy_per_lap,  
                   battery_capacity=battery_capacity)
    
    race.calculate()

    print("Battery capacity (kWh): {}\nEnergy per lap (J): {}\nLap time (s): {}"
        .format(battery_capacity, energy_per_lap, lap_time))
    for i, race_day in enumerate(race.race_days):
        print("\nRace Day: {}".format(i + 1))
        print("Total Race Time: {}".format(race_day.race_time))
        print("Total Laps: {}".format(race_day.total_laps))
        print("Number of pits: {}, time of pits: {}".format(race_day.number_of_pits, 
                                                            race_day.time_of_pits))
        print("Percent of battery remaining: {}".format(race_day.energy_remaining))

       