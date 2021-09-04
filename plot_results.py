# plotting script, reads in previously created csv
# and plots    
import argparse
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt 

from definitions import *

def parse_args():
    arg_parser = argparse.ArgumentParser("Plot racesim results with matplotlib")

    arg_parser.add_argument('-f','--results-file', required=True,
                            help='path to results csv file')

    args = arg_parser.parse_args()

    args.results_file = os.path.abspath(args.results_file)

    return args

def csv_reader(file_path):
    data = pd.read_csv(file_path)
    print(data[TOTAL_LAPS_TAG])
    print(type(data[TOTAL_LAPS_TAG]))

    return data

def plot_data(data):
    # https://matplotlib.org/stable/
    index = data.get(ITER_TAG)

    lap_time = data.get(LAPTIME_TAG)

    lap_energy = data.get(LAP_ENERGY_TAG)

    fig = plt.figure()

    ax1 = fig.add_subplot(111)
    ax1.set_xlabel(ITER_TAG)
    ax1.set_ylabel(LAPTIME_TAG)
    ax1.set_title(LAPTIME_TAG)
    ax1.plot(index, lap_time)    

    #         # Good old data mainpulation to get it graphing. I did this with 
    #         # lots of googling on stack exchange            
    laptime_dataframe = pd.DataFrame({
            BATTERY_SIZE_TAG: data.get(BATTERY_SIZE_TAG),
            MOTOR_MAX_TORQUE_TAG: data.get(MOTOR_MAX_TORQUE_TAG),
            LAPTIME_TAG: data.get(LAPTIME_TAG)
        })

    energy_dataframe = pd.DataFrame({
            BATTERY_SIZE_TAG: data.get(BATTERY_SIZE_TAG),
            MOTOR_MAX_TORQUE_TAG: data.get(MOTOR_MAX_TORQUE_TAG),
            LAP_ENERGY_TAG: data.get(LAP_ENERGY_TAG)
        })

    total_laps_dataframe = pd.DataFrame({
        BATTERY_SIZE_TAG: data.get(BATTERY_SIZE_TAG),
        MOTOR_MAX_TORQUE_TAG: data.get(MOTOR_MAX_TORQUE_TAG),
        TOTAL_LAPS_TAG: data.get(TOTAL_LAPS_TAG)
    })

    Energy_array = energy_dataframe.pivot_table(
        index=BATTERY_SIZE_TAG,
        columns=MOTOR_MAX_TORQUE_TAG,
        values=LAP_ENERGY_TAG
    ).T.values

    Laptime_array = laptime_dataframe.pivot_table(
        index=BATTERY_SIZE_TAG,
        columns=MOTOR_MAX_TORQUE_TAG,
        values=LAPTIME_TAG
    ).T.values

    total_laps_array = total_laps_dataframe.pivot_table(
        index=BATTERY_SIZE_TAG,
        columns=MOTOR_MAX_TORQUE_TAG,
        values=TOTAL_LAPS_TAG
    ).T.values

    battery_size_unique = np.sort(np.unique(data.get(BATTERY_SIZE_TAG)))
    motor_max_torque_unique = np.sort(np.unique(data.get(MOTOR_MAX_TORQUE_TAG)))

    battery_size_array, motor_max_torque_array = np.meshgrid(battery_size_unique, motor_max_torque_unique)

    fig = plt.figure()
    ax1 = fig.add_subplot(111,projection='3d')
    ax1.set_xlabel(BATTERY_SIZE_TAG)
    ax1.set_ylabel(MOTOR_MAX_TORQUE_TAG)
    ax1.set_zlabel(LAP_ENERGY_TAG)
    ax1.set_title(LAP_ENERGY_TAG)
    ax1.plot_surface(battery_size_array, motor_max_torque_array, Energy_array)

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111,projection='3d')
    ax2.set_xlabel(BATTERY_SIZE_TAG)
    ax2.set_ylabel(MOTOR_MAX_TORQUE_TAG)
    ax2.set_zlabel(LAPTIME_TAG)
    ax2.set_title(LAPTIME_TAG)
    ax2.plot_surface(battery_size_array, motor_max_torque_array, Laptime_array)

    fig3 = plt.figure()
    ax3 = fig3.add_subplot(111,projection='3d')
    ax3.set_xlabel(BATTERY_SIZE_TAG)
    ax3.set_ylabel(MOTOR_MAX_TORQUE_TAG)
    ax3.set_zlabel(TOTAL_LAPS_TAG)
    ax3.set_title(TOTAL_LAPS_TAG)
    ax3.plot_surface(battery_size_array, motor_max_torque_array, total_laps_array)
    plt.show()

def main():
    args = parse_args()

    data = csv_reader(args.results_file)

    plot_data(data)

if __name__ == '__main__':
    main()

    #     elif sa_opts["sa_type"] == "elemons_mass_cd_torque":
    #         # https://stackoverflow.com/questions/14995610/how-to-make-a-4d-plot-with-matplotlib-using-arbitrary-data
    #         # graph mass and cd vs energy consumption/time at each max torque
    #         fig = plt.figure()
    #         fig2 = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax2 = fig2.add_subplot(111, projection='3d')

    #         ax.set_xlabel("Mass (kg)")
    #         ax.set_ylabel("C_d")
    #         ax.set_zlabel("Max Torque")
    #         ax.set_title("Energy Consumption (J) vs \n Mass, C_d, Max_torque")
    #         ax2.set_xlabel("Mass (kg)")
    #         ax2.set_ylabel("C_d")
    #         ax2.set_zlabel("Max Torque")
    #         ax2.set_title("Time of lap (s) vs \n Mass, C_d, Max_torque")


    #         img = ax.scatter(sa_mass, sa_c_d, sa_torque, c=sa_fuel_cons, cmap=plt.hot())
    #         img2 = ax2.scatter(sa_mass, sa_c_d, sa_torque, c=sa_t_lap, cmap=plt.hot())
    #         fig.colorbar(img)
    #         fig2.colorbar(img2)
    #         plt.show()