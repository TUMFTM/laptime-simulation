# plotting script, reads in previously created csv
# and plots    
import argparse
import pandas
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
    data = pandas.read_csv(file_path)

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
    plt.show()

    #         # Good old data mainpulation to get it graphing. I did this with 
    #         # lots of googling on stack exchange            
    #         Laptime_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], LAPTIME_TAG: sa_t_lap[:]})
    #         Energy_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], LAP_ENERGY_TAG: sa_fuel_cons[:]})
    #         total_laps_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], TOTAL_LAPS_TAG: sa_total_laps[:]})

    #         Energy_array = Energy_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=LAP_ENERGY_TAG).T.values
    #         Laptime_array = Laptime_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=LAPTIME_TAG).T.values
    #         total_laps_array = total_laps_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=TOTAL_LAPS_TAG).T.values

    #         mass_unique = np.sort(np.unique(sa_mass))
    #         c_d_unique = np.sort(np.unique(sa_c_d))

    #         mass_array, c_d_array = np.meshgrid(mass_unique, c_d_unique)

    #         fig = plt.figure()
    #         fig2 = plt.figure()
    #         ax1 = fig.add_subplot(111,projection='3d')
    #         ax1.set_xlabel("Mass (kg)")
    #         ax1.set_ylabel("Coeff of Drag - Cd")
    #         ax1.set_zlabel("Energy per lap (kJ) * ")
    #         ax1.set_title('Energy Per Lap\nvehicle: ' +  solver_opts["vehicle"] + '\ntrack: ' + track_opts["trackname"])
    #         ax1.plot_surface(mass_array, c_d_array, Energy_array)

    #         ax2 = fig2.add_subplot(111,projection='3d')
    #         ax2.set_xlabel('Mass (kg)')
    #         ax2.set_ylabel('Coeff of Drag - Cd')
    #         ax2.set_zlabel('Lap Time (sec)')
    #         ax2.set_title('Lap Times\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         ax2.plot_surface(mass_array, c_d_array, Laptime_array)
 
    #         fig3 = plt.figure()
    #         ax3 = fig3.add_subplot(111,projection='3d')
    #         ax3.set_xlabel('Mass (kg)')
    #         ax3.set_ylabel('Coeff of Drag - Cd')
    #         ax3.set_zlabel('Total Laps')
    #         ax3.set_title('Total Laps\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         ax3.plot_surface(mass_array, c_d_array, total_laps_array)
    #         plt.show()

    

def main():
    args = parse_args()

    data = csv_reader(args.results_file)

    plot_data(data)

if __name__ == '__main__':
    main()

    # # read in csv file and return data in a pythonic structure
    # # ------------------------------------------------------------------------------------------------------------------
    # # PLOTS ------------------------------------------------------------------------------------------------------------
    # # ------------------------------------------------------------------------------------------------------------------

    # if not sa_opts["use_sa"]:
    #     if debug_opts["use_plot"]:
    #         lap.plot_overview()
    #         # lap.plot_revs_gears()
    # else:
    #     sa_t_lap, sa_fuel_cons, sa_iter, sa_mass, sa_c_d, sa_torque, sa_total_laps = datastore.get_graph_data()
        
    #     if sa_opts["sa_type"] == "mass":
    #         # lap time
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(sa_mass, sa_t_lap, "x")
    #         ax.set_xlim(sa_mass[0], sa_mass[-1])
    #         ax.set_ylim(sa_t_lap[0], sa_t_lap[-1])
    #         ax.set_title("SA of lap time to mass")
    #         ax.set_xlabel("mass m in kg")
    #         ax.set_ylabel("lap time t in s")
    #         plt.grid()
    #         plt.show()

    #         # fuel consumption
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(sa_mass, sa_fuel_cons, "x")
    #         ax.set_xlim(sa_mass[0], sa_mass[-1])
    #         ax.set_ylim(sa_fuel_cons[0], sa_fuel_cons[-1])
    #         ax.set_title("SA of fuel consumption to mass")
    #         ax.set_xlabel("mass m in kg")
    #         ax.set_ylabel("fuel consumption in kg/lap")
    #         plt.grid()
    #         plt.show()

    #     elif sa_opts["sa_type"] == "elemons_mass":
    #         # lap time
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(sa_mass, sa_t_lap, "x")
    #         ax.set_title("SA of lap time to mass")
    #         ax.set_xlabel("mass m in kg")
    #         ax.set_ylabel("lap time t in s")
    #         ax.set_title('Lap Times\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         plt.grid()
    #         plt.show()

    #         # fuel (energy) consumption
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111)
    #         ax.plot(sa_mass, sa_fuel_cons, "x")
    #         ax.set_title("SA of energy consumption to mass")
    #         ax.set_xlabel("mass m in kg")
    #         ax.set_ylabel("energy consumption in kJ/lap")
    #         ax.set_title('Energy Consumption\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         plt.grid()
    #         plt.show()

    #     elif sa_opts["sa_type"] == "elemons_mass_cd":
    #         # Good old data mainpulation to get it graphing. I did this with 
    #         # lots of googling on stack exchange            
    #         Laptime_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], LAPTIME_TAG: sa_t_lap[:]})
    #         Energy_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], LAP_ENERGY_TAG: sa_fuel_cons[:]})
    #         total_laps_dataframe = pd.DataFrame({MASS_TAG: sa_mass[:], C_D_TAG: sa_c_d[:], TOTAL_LAPS_TAG: sa_total_laps[:]})

    #         Energy_array = Energy_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=LAP_ENERGY_TAG).T.values
    #         Laptime_array = Laptime_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=LAPTIME_TAG).T.values
    #         total_laps_array = total_laps_dataframe.pivot_table(index=MASS_TAG, columns=C_D_TAG, values=TOTAL_LAPS_TAG).T.values

    #         mass_unique = np.sort(np.unique(sa_mass))
    #         c_d_unique = np.sort(np.unique(sa_c_d))

    #         mass_array, c_d_array = np.meshgrid(mass_unique, c_d_unique)

    #         fig = plt.figure()
    #         fig2 = plt.figure()
    #         ax1 = fig.add_subplot(111,projection='3d')
    #         ax1.set_xlabel("Mass (kg)")
    #         ax1.set_ylabel("Coeff of Drag - Cd")
    #         ax1.set_zlabel("Energy per lap (kJ) * ")
    #         ax1.set_title('Energy Per Lap\nvehicle: ' +  solver_opts["vehicle"] + '\ntrack: ' + track_opts["trackname"])
    #         ax1.plot_surface(mass_array, c_d_array, Energy_array)

    #         ax2 = fig2.add_subplot(111,projection='3d')
    #         ax2.set_xlabel('Mass (kg)')
    #         ax2.set_ylabel('Coeff of Drag - Cd')
    #         ax2.set_zlabel('Lap Time (sec)')
    #         ax2.set_title('Lap Times\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         ax2.plot_surface(mass_array, c_d_array, Laptime_array)
 
    #         fig3 = plt.figure()
    #         ax3 = fig3.add_subplot(111,projection='3d')
    #         ax3.set_xlabel('Mass (kg)')
    #         ax3.set_ylabel('Coeff of Drag - Cd')
    #         ax3.set_zlabel('Total Laps')
    #         ax3.set_title('Total Laps\nvehicle: ' + solver_opts["vehicle"] + ' \ntrack: ' + track_opts["trackname"])
    #         ax3.plot_surface(mass_array, c_d_array, total_laps_array)
    #         plt.show()

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