import math
import os
import pickle
import re
import csv
import numpy as np
#import localization as Loc
import scipy as sc
import localization as lx
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
import statistics
import json
from utils import *
from schema import make_default_schema
from scipy.stats import linregress

boards = [
    "FC:90:0F:C9:6E:24",
    "E4:B6:69:C8:DB:9D",
    "FE:5A:0F:0E:29:6F",
    "F7:45:87:CA:E8:06",
    "DA:EB:08:C4:34:32"
]




def clear_errors_from_file(file_name):
    with open(file_name, 'r') as file:
        lines = file.readlines()

    filtered_lines = [line for line in lines if not (line.startswith('E:') or
                                                     line.startswith('I:') or
                                                     line.startswith('KONF:') or
                                                     line.strip() == '' or
                                                     line.startswith('***') or
                                                     line.startswith('Starting') or
                                                     line.startswith('DM'))]

    with open(file_name, 'w') as file:
        file.writelines(filtered_lines)



def read_measurements(file_name):
    meas_list = []

    file = open(file_name, 'r')
    lines = file.readlines()

    counter = 0
    anchor = None
    quality = None
    high_precision = None
    ifft = None
    phase_slope = None
    rssi = None
    best = None

    line_number = 0
    for line in lines:
        line_number = line_number + 1
        try:
            counter += 1

            #addr
            if counter % 4 == 2:
                match = re.search(r'Addr:\s*([0-9A-Fa-f:]+)', line)
                anchor = match.group(1)

            #quality
            elif counter % 4 == 3:
                match = re.search(r'Quality:\s*(\w+)', line)
                quality = match.group(1)
            #dist est
            elif counter % 4 == 0:
                matches = re.findall(r'high_precision=(\d*\.?\d+|nan)|ifft=(\d*\.?\d+)|phase_slope=(\d*\.?\d+)|rssi_openspace=(\d*\.?\d+)|best=(\d*\.?\d+)',line)


                if len(matches) == 5:
                    high_precision = matches[0][0]
                    ifft = matches[1][1]
                    phase_slope = matches[2][2]
                    rssi = matches[3][3]
                    best = matches[4][4]
                elif len(matches) == 4:
                    high_precision = 0
                    ifft = matches[0][1]
                    phase_slope = matches[1][2]
                    rssi = matches[2][3]
                    best = matches[3][4]


                meas = measurement(anchor, quality, float(high_precision),
                                   float(ifft), float(phase_slope), float(rssi), float(best))
                meas_list.append(meas)
        except Exception:
            print(f"Exception file: {file_name}")
            print(f"Exceptionon line: {line_number} ||{line}")


    return meas_list

def create_address_histogram(adrr_dict, save_to_file=True, file_name="Adresy"):
    addresses = list(adrr_dict.keys())
    occurrences = list(adrr_dict.values())

    total_occ = sum(occurrences)

    plt.figure(figsize=(10, 6))
    bars = plt.bar(addresses, occurrences)

    plt.title(f"Liczba wystapien adresu na {total_occ}")
    plt.xlabel("PŁYTKA")
    plt.ylabel("Liczba wystapien")

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval),
                 va='bottom',
                 ha='center',
                 fontweight='bold',
                 color='black')

    if save_to_file:
        plt.savefig(file_name + "_histogram")

    plt.show()

    pass

def save_to_csv(file_name, data):

    field_names = ["Anchor", "Quality", "High_Precision", "IFFT", "Phase_Slope", "RSSI_Openspace", "Best"]

    with open(file_name, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=field_names)
        writer.writeheader()
        for meas in data:
            writer.writerow({"Anchor": meas.anchor,
                             "Quality": meas.quality,
                             "High_Precision": meas.high_precision,
                             "IFFT": meas.ifft,
                             "Phase_Slope": meas.phase_slope,
                             "RSSI_Openspace": meas.rssi_openspace,
                             "Best": meas.best})
    return

def create_anchor_occurannce_histogram(csv_path, save_to_file=True):
    df = pd.read_csv(csv_path)
    column_name = "Anchor"

    value_counts = df[column_name].value_counts()

    plt.figure(figsize=(10,6))
    bars = plt.bar(value_counts.index, value_counts.values, color='skyblue')
    plt.xlabel("Kotwice")
    plt.ylabel("Liczba pomiarów")
    #plt.title(f"Histogram ilosci wystapien kotwic na {len(df)} pomiarów")
    plt.title(f"Liczba wystąpień danej kotwicy w {len(df)} pomiarach")
    plt.xticks(rotation=45)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom')


    plt.tight_layout()

    if save_to_file:
        file_name = csv_path.split('.')[0]

        plt.savefig(file_name + "_histogram")

    plt.show()

#Zwraca slownik w postaci adres: lista_pomiarow
def sort_measurements_for_anchor(data_list):
    anchor_dict = dict()

    for measurement in data_list:
        if measurement.anchor in anchor_dict.keys():
            #dodac do listy
            me_list = anchor_dict[measurement.anchor]
            me_list.append(measurement)

            anchor_dict[measurement.anchor] = me_list
        else:
            me_list = []
            me_list.append(measurement)

            anchor_dict[measurement.anchor] = me_list

    return anchor_dict
#obliczenie sredniej wartosci pomiarów odleglosci
def get_average_distances(measurement_list):
    high_prec_total = 0
    ifft_total = 0
    phase_slope_total = 0
    rssi_openspace_total = 0
    best_total = 0

    nan_hp_number = 0

    # rssi_list = []

    for meas in measurement_list:
        if meas.high_precision != "nan":
            high_prec_total = high_prec_total + float(meas.high_precision)
        else:
            high_prec_total = high_prec_total + 0
            nan_hp_number += 1
        ifft_total = ifft_total + float(meas.ifft)
        phase_slope_total = phase_slope_total + float(meas.phase_slope)
        rssi_openspace_total = rssi_openspace_total + float(meas.rssi_openspace)
        best_total = best_total + float(meas.best)

        # rssi_list.append(float(meas.rssi_openspace))

    # print("MIN RSSI")
    # print(min(rssi_list))

    meas_number = len(measurement_list)



    average_measurement = measurement(measurement_list[0].anchor,
                                      measurement_list[0].quality,
                                      high_prec_total/(meas_number - nan_hp_number),
                                      ifft_total/meas_number,
                                      phase_slope_total/meas_number,
                                      rssi_openspace_total/meas_number,
                                      best_total/meas_number)
    return average_measurement

#oblicznanie srednich odleglosci dla kazdej kotwicy w slowniku
def get_average_measurement_per_anchor(meas_dict):
    avg_measurements = []
    for anchor in meas_dict.keys():
        meas = get_average_distances(meas_dict[anchor])
        avg_measurements.append(meas)

    return avg_measurements

def get_med_measurement(measurement_list):
    ifft_list = []
    phase_list = []
    rssi_list = []
    best_list = []

    for meas in measurement_list:
        ifft_list.append(float(meas.ifft))
        phase_list.append(float(meas.phase_slope))
        rssi_list.append(float(meas.rssi_openspace))
        best_list.append(float(meas.best))

    median_meas = measurement(measurement_list[0].anchor,
                              measurement_list[0].quality,
                              0,
                              statistics.median(ifft_list),
                              statistics.median(phase_list),
                              statistics.median(rssi_list),
                              statistics.median(best_list))

    return median_meas

def get_median_measurement_per_anchor(meas_dict):
    med_measurements = []
    for anchor in meas_dict.keys():
        meas = get_med_measurement(meas_dict[anchor])
        med_measurements.append(meas)
    return med_measurements


def create_boxplot(anchor_measurement_data, Anchor, actual_value=None, save_to_file=False):
    HPlist = []
    ifftlist = []
    phaselist = []
    rssilist = []
    bestlist = []

    for meas in anchor_measurement_data:
        if meas.high_precision != "nan":
            HPlist.append(meas.high_precision)
        ifftlist.append(meas.ifft)
        phaselist.append(meas.phase_slope)
        rssilist.append(meas.rssi_openspace)
        bestlist.append(meas.best)

    data_HP = np.array(HPlist, dtype=float)
    data_ifft = np.array(ifftlist, dtype=float)
    data_phase = np.array(phaselist, dtype=float)
    data_rssi = np.array(rssilist, dtype=float)
    data_best = np.array(bestlist, dtype=float)

    data_sets = [data_HP, data_ifft, data_phase, data_rssi, data_best]

    boxprops = dict(linestyle='-', linewidth=2,color='k')
    medianprops = dict(linestyle='-', linewidth=2, color='firebrick')

    boxplot_elements = plt.boxplot(data_sets, patch_artist=True, boxprops=boxprops,medianprops=medianprops)

    medians = [item.get_ydata()[0] for item in boxplot_elements['medians']]
    for i, median in enumerate(medians):
        plt.text(x=i + 1.3, y=median, s=f'{median}', verticalalignment='center', color='black')

    #prawdziwa wartość
    if actual_value is not None:
        plt.axhline(y=actual_value, color='red', linestyle='--', linewidth=2, label=f'True distance: {actual_value}')


    #plt.boxplot(data_sets)
    plt.title(f"Measurements for {Anchor.name}")
    plt.ylabel('Distance [m]')
    plt.xticks([1,2,3,4,5], ["HP", "IFFT", "Phase", "RSSI", "Best"])

    if save_to_file:
        board_name = Anchor.name
        file_name = board_name + "_boxplot"
        plt.savefig(file_name)

    plt.show()

    pass



def create_boxplots_for_measurement(Anchors, measurements_dict, real_point, remove_outliers=True, save_to_file=False, save_path="reporting"):
    data_sets = []

    for anchor in Anchors:
        for anch_key in measurements_dict.keys():
            if anchor.name == anch_key:
                ifftlist = []
                phaselist = []
                rssilist = []
                bestlist = []
                #Wyciagniecie pomiarow ze slownika i dodanie w odpowiedniej kolejnosci dostosowanej do kotwicy
                for meas in measurements_dict[anch_key]:

                    ifftlist.append(meas.ifft)
                    phaselist.append(meas.phase_slope)
                    rssilist.append(meas.rssi_openspace)
                    bestlist.append(meas.best)


                data_ifft = np.array(ifftlist, dtype=float)
                data_phase = np.array(phaselist, dtype=float)
                data_rssi = np.array(rssilist, dtype=float)
                data_best = np.array(bestlist, dtype=float)

                ds = [data_ifft, data_phase, data_rssi, data_best]

                if remove_outliers:
                    ds_new = []
                    for data_list in ds:
                        Q1 = np.percentile(data_list, 25)
                        Q3 = np.percentile(data_list, 75)

                        IQR = Q3 - Q1
                        lower_bound = Q1 - 1.5 * IQR
                        upper_bound = Q3 + 1.5 * IQR

                        filtered_data = [x for x in data_list if x >= lower_bound and x <= upper_bound]
                        print(f"OG DATA LEN {len(data_list)}")
                        print(f"FILTERED DATA LEN {len(filtered_data)}")

                        ds_new.append(filtered_data)
                    ds = ds_new

                data_sets.append(ds)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
    axes = axes.flatten()

    for id, anchor in enumerate(Anchors):
        boxprops = dict(linestyle='-', linewidth=2, color='k')
        medianprops = dict(linestyle='-', linewidth=2, color='firebrick')
        ####tu poprawic dla e_8_10 gdzie nie ma FE
        if id > len(data_sets) - 1:
            break
        boxplot = axes[id].boxplot(data_sets[id], patch_artist=True, boxprops=boxprops, medianprops=medianprops)
        medians = [item.get_ydata()[0] for item in boxplot['medians']]
        for i, median in enumerate(medians):
            axes[id].text(x=i + 1.3, y=median, s=f'{median}', verticalalignment='center', color='black')

        true_distance = euclidean_distance(anchor, real_point[0], real_point[1], True)
        axes[id].axhline(y=true_distance, color='red', linestyle='--', linewidth=2, label=f'True distance: {true_distance}')

        axes[id].set_title(f"Kotwica: {anchor.name}")
        axes[id].set_ylabel('Odległość [m]')
        axes[id].set_xticks([1,2,3,4], ["IFFT", "PHASE", "RSSI", "BEST"])

    fig.suptitle(f"Porównanie zmierzonych odległości w punkcie ({real_point[0]}, {real_point[1]})")
    plt.tight_layout()

    if save_to_file:
        file_name = save_path + "//" +f"Y{real_point[1]}//{real_point[0]}_{real_point[1]}" +"_boxplot"
        plt.savefig(file_name)
    #plt.show()



def print_avg_measurements(avg_meas):
    for anch in avg_meas:
        print(anch)

def clear_failed_measurements(ml):
    new_list = []
    for meas in ml:
        if meas.quality == "ok":
            new_list.append(meas)

    return new_list



def locate_point(avg_meas, type='3D', og_point=None):
    lp = lx.Project(mode=type, solver='LSE')
    lp.add_anchor(anchors[0].name, (anchors[0].x_cord, anchors[0].y_cord, anchors[0].z_cord))
    lp.add_anchor(anchors[1].name, (anchors[1].x_cord, anchors[1].y_cord, anchors[1].z_cord))
    lp.add_anchor(anchors[2].name, (anchors[2].x_cord, anchors[2].y_cord, anchors[2].z_cord))
    lp.add_anchor(anchors[3].name, (anchors[3].x_cord, anchors[3].y_cord, anchors[3].z_cord))

    ifft, label = lp.add_target()
    phase, phaseLabel = lp.add_target()
    rssi, rssilabel = lp.add_target()
    best, bestlabel = lp.add_target()
    #ifft
    for meas in avg_meas:
        ifft.add_measure(meas.anchor, meas.ifft)
        phase.add_measure(meas.anchor, meas.phase_slope)
        rssi.add_measure(meas.anchor, meas.rssi_openspace)
        best.add_measure(meas.anchor, meas.best)

    lp.solve()

    calc_dist = og_point is not None

    if type == '3D':
        ifft_result = mult_result('IFFT', ifft.loc.x,
                              ifft.loc.y, ifft.loc.z,
                              euclidean_dist_2points((ifft.loc.x, ifft.loc.y, ifft.loc.z),
                                                 (og_point[0], og_point[1], og_point[2]), True))
        phase_result = mult_result('PHASE', phase.loc.x,
                              phase.loc.y, phase.loc.z,
                              euclidean_dist_2points((phase.loc.x, phase.loc.y, phase.loc.z),
                                                 (og_point[0], og_point[1], og_point[2]), True))
        rssi_result = mult_result('RSSI', rssi.loc.x,
                              rssi.loc.y, rssi.loc.z,
                              euclidean_dist_2points((rssi.loc.x, rssi.loc.y, rssi.loc.z),
                                                 (og_point[0], og_point[1], og_point[2]), True))
        best_result = mult_result('BEST', best.loc.x,
                              best.loc.y, best.loc.z,
                              euclidean_dist_2points((best.loc.x, best.loc.y, best.loc.z),
                                                 (og_point[0], og_point[1], og_point[2]), True))
    elif type == '2D':
        ifft_result = mult_result('IFFT', ifft.loc.x,
                                  ifft.loc.y, 1.9,
                                  euclidean_dist_2points((ifft.loc.x, ifft.loc.y),
                                                    (og_point[0], og_point[1]), False))
        phase_result = mult_result('PHASE', phase.loc.x,
                                   phase.loc.y, 1.9,
                                   euclidean_dist_2points((phase.loc.x, phase.loc.y),
                                                    (og_point[0], og_point[1]), False))
        rssi_result = mult_result('RSSI', rssi.loc.x,
                                  rssi.loc.y, 1.9,
                                  euclidean_dist_2points((rssi.loc.x, rssi.loc.y),
                                                    (og_point[0], og_point[1]), False))
        best_result = mult_result('BEST', best.loc.x,
                                  best.loc.y, best.loc.z,
                                  euclidean_dist_2points((best.loc.x, best.loc.y),
                                                    (og_point[0], og_point[1]), False))
    res_dict = {'IFFT': ifft_result,
                'PHASE': phase_result,
                'RSSI': rssi_result,
                'BEST': best_result}

    print(f"IFFF loc: {ifft.loc}: diff:{res_dict['IFFT'].distance_from_point}")
    print(f"PHASE loc: {phase.loc}: diff:{res_dict['PHASE'].distance_from_point}")
    print(f"RSSI loc: {rssi.loc}: diff:{res_dict['IFFT'].distance_from_point}")
    print(f"BEST loc: {best.loc}: diff:{res_dict['BEST'].distance_from_point}")


    return res_dict

def get_sorted_meas_size_string(sorted_meas):
    result = ""
    for key in sorted_meas.keys():
        anchor_name = key.split(":")[0]
        result += f"{anchor_name}: {len(sorted_meas[key])}, "
    return result

def save_loc_res_to_file(path, list):
    with open(path, "wb") as f:
        pickle.dump(list, f)

def load_loc_res_from_file(path):
    with open(path, 'rb') as f:
        list = pickle.load(f)
        return list

def analyze_all_files(base_dir, report_dir, multilateration_type='3D', clear_failed_meas=True, bias=None, create_boxplots=False):
    location_results = []

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                x, y = extract_location_cord_from_file(file)
                ml = read_measurements(file_path)
                if clear_failed_meas:
                    ml = clear_failed_measurements(ml) #usuwanie wyników nie 'ok'
                sorted_meas = sort_measurements_for_anchor(ml)
                if bias is not None:
                    sorted_meas = sub_bias(sorted_meas, bias)

                if create_boxplots:
                    create_boxplots_for_measurement(anchors, sorted_meas, (x, y), False, True)
                avg_meas = get_average_measurement_per_anchor(sorted_meas)
                med_meas = get_median_measurement_per_anchor(sorted_meas)

                # print(f"\n({x},{y}): {get_sorted_meas_size_string(sorted_meas)}")
                # print("AVG")
                # print_avg_measurements(avg_meas)
                avg_mult_res = locate_point(avg_meas, multilateration_type, (x,y,1.6))
                # print("MEDIAN")
                # print_avg_measurements(med_meas)
                med_mult_res = locate_point(med_meas, multilateration_type, (x,y,1.6))

                location_results.append(location_measurements_results(
                    x=x,
                    y=y,
                    meas_dict=sorted_meas,
                    avg_meas=avg_meas,
                    med_meas=med_meas,
                    avg_mult=avg_mult_res,
                    med_mult=med_mult_res
                ))

    #do pliku json
    #dict_list = [loc_res.to_dict() for loc_res in location_results]
    save_loc_res_to_file(f"{report_dir}//results.pkl", location_results)


def sub_bias(sorted_meas, bias):
    # print(sorted_meas)
    # print(bias)

    for anchor in sorted_meas.keys():
        print(anchor)
        for meas in sorted_meas[anchor]:
            meas.ifft = float(meas.ifft) - float(bias[anchor.split(":")[0]]['IFFT'])
            meas.phase_slope = float(meas.phase_slope) - float(bias[anchor.split(":")[0]]['PHASE'])
            meas.rssi_openspace = float(meas.rssi_openspace) - float(bias[anchor.split(":")[0]]['RSSI'])
            meas.best = float(meas.best) - float(bias[anchor.split(":")[0]]['BEST'])

    return sorted_meas


def create_scatterplot_distance(loc_res_list, anchor, mult_type="avg"):
    true_distances = []
    measured_distances = []
    for loc_res in loc_res_list:
        print(f"POINT: ({loc_res.x},{loc_res.y})")
        td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord), False)
        true_distances.append(td)
        print(td)
        if mult_type == "avg":

            for meas in loc_res.avg_meas:
                print(meas.anchor)
                if meas.anchor == anchor.name:
                    print("HIT")
                    md = meas.ifft
            measured_distances.append(md)
        pass


    # Create a scatter plot
    plt.figure(figsize=(10, 5))
    # plt.scatter(true_distance, measured_distance_1, color='red', label='Measured Distance 1')
    # plt.scatter(true_distance, measured_distance_2, color='blue', label='Measured Distance 2')
    #plt.scatter(true_distances, true_distances, color='blue')
    plt.scatter(true_distances, measured_distances, color='red')
    plt.plot(true_distances, true_distances, color='black', linestyle='--')  # Line y=x for reference

    # Labeling the plot
    plt.xlabel('Prawdziwy dystans [m]')
    plt.ylabel('Zmierzony dystans [m]')
    plt.legend()
    plt.title('Comparison of Measured Distance vs True Distance')

    # Display the plot
    plt.grid(True)
    plt.show()


def create_scatter_plots_for_anchor(loc_res_list, anchor, mult_type="avg", save_to_file=True, report_dir='reporting'):
    true_distances = []

    measured_distances_ifft = []
    measured_distances_phase = []
    measured_distances_rssi = []
    measured_distances_best = []

    for loc_res in loc_res_list:
        td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord), False)
        true_distances.append(td)

        if mult_type == "avg":
            for meas in loc_res.avg_meas:
                if meas.anchor == anchor.name:
                    measured_distances_ifft.append(meas.ifft)
                    measured_distances_phase.append(meas.phase_slope)
                    measured_distances_rssi.append(meas.rssi_openspace)
                    measured_distances_best.append(meas.best)
        elif mult_type == 'med':
            for meas in loc_res.med_meas:
                if meas.anchor == anchor.name:
                    measured_distances_ifft.append(meas.ifft)
                    measured_distances_phase.append(meas.phase_slope)
                    measured_distances_rssi.append(meas.rssi_openspace)
                    measured_distances_best.append(meas.best)

        if len(true_distances) != len(measured_distances_ifft):
            true_distances.pop()

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for id, type in enumerate(["IFFT", "PHASE", "RSSI", "BEST"]):
        if type == "IFFT":
            pass
            plot = axes[id].scatter(true_distances, measured_distances_ifft, color='red')
        elif type == "PHASE":
            pass
            plot = axes[id].scatter(true_distances, measured_distances_phase, color='green')
        elif type == "RSSI":
            plot = axes[id].scatter(true_distances, measured_distances_rssi, color='blue')
        elif type == "BEST":
            plot = axes[id].scatter(true_distances, measured_distances_best, color='orange')

        axes[id].plot(true_distances, true_distances, color='black', linestyle='--')

        axes[id].set_xlabel('Prawdziwy dystans [m]')
        axes[id].set_ylabel('Zmierzony dystans [m]')
        axes[id].grid(True)

        axes[id].set_title(f'{type}')

    fig.suptitle(f"Porównanie jakości zmierzonych odległości dla kotwicy {anchor.name} - {mult_type}")
    plt.tight_layout()


    if save_to_file:
        file_name = f".//{report_dir}//"+ f"meas_quality_{anchor.name.split(':')[0]}_{mult_type}"
        plt.savefig(file_name)


    plt.show()

#####
def create_scatter_plots_for_anchor_minmax(loc_res_list, anchor, mult_type="avg", save_to_file=True, report_dir='reporting', regression=True):
    true_distances = []

    measured_distances_ifft = []
    measured_distances_ifft_min = []
    measured_distances_ifft_max = []

    measured_distances_phase = []
    measured_distances_phase_min = []
    measured_distances_phase_max = []

    measured_distances_rssi = []
    measured_distances_rssi_min = []
    measured_distances_rssi_max = []

    measured_distances_best = []
    measured_distances_best_min = []
    measured_distances_best_max = []

    for loc_res in loc_res_list:
        td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord), False)
        true_distances.append(td)

        if mult_type == "avg":
            for meas in loc_res.avg_meas:
                if meas.anchor == anchor.name:
                    measured_distances_ifft.append(meas.ifft)
                    measured_distances_phase.append(meas.phase_slope)
                    measured_distances_rssi.append(meas.rssi_openspace)
                    measured_distances_best.append(meas.best)
        elif mult_type == 'med':
            for meas in loc_res.med_meas:
                if meas.anchor == anchor.name:
                    measured_distances_ifft.append(meas.ifft)
                    measured_distances_phase.append(meas.phase_slope)
                    measured_distances_rssi.append(meas.rssi_openspace)
                    measured_distances_best.append(meas.best)

        min_values = loc_res.get_min_measurements_per_anchor(anchor.name)
        max_values = loc_res.get_max_measurements_per_anchor(anchor.name)
        measured_distances_ifft_min.append(min_values["IFFT"])
        measured_distances_ifft_max.append(max_values["IFFT"])
        measured_distances_phase_min.append(min_values["PHASE"])
        measured_distances_phase_max.append(max_values["PHASE"])
        measured_distances_rssi_min.append(min_values["RSSI"])
        measured_distances_rssi_max.append(max_values["RSSI"])
        measured_distances_best_min.append(min_values["BEST"])
        measured_distances_best_max.append(max_values["BEST"])

        if len(true_distances) != len(measured_distances_ifft):
            true_distances.pop()



    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    axes = axes.flatten()

    for id, type in enumerate(["IFFT", "PHASE", "RSSI", "BEST"]):
        if type == "IFFT":
            plot = axes[id].scatter(true_distances, measured_distances_ifft, color='red')
            axes[id].scatter(true_distances, measured_distances_ifft_min, color='black', marker='2', s=11)
            axes[id].scatter(true_distances, measured_distances_ifft_max, color='black', marker='1', s=11)
            #linie łączące
            for i in range(len(true_distances)):
                axes[id].plot([true_distances[i], true_distances[i]],
                              [measured_distances_ifft_min[i], measured_distances_ifft_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            #regresja
            if regression:
                slope, intercept, _,_, _ = linregress(true_distances, measured_distances_ifft)
                x_vals = np.array(true_distances)
                y_vals = x_vals + intercept
                axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
                print(f"INTERCEPT {mult_type} IFFT {anchor.name}: y=x+{intercept:.2f}")

        elif type == "PHASE":
            plot = axes[id].scatter(true_distances, measured_distances_phase, color='green')
            axes[id].scatter(true_distances, measured_distances_phase_min, color='black', marker='2', s=11)
            axes[id].scatter(true_distances, measured_distances_phase_max, color='black', marker='1', s=11)
            for i in range(len(true_distances)):
                axes[id].plot([true_distances[i], true_distances[i]],
                              [measured_distances_phase_min[i], measured_distances_phase_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            axes[id].set_ylim(0,30)
            #regresja
            if regression:
                slope, intercept, _, _, _ = linregress(true_distances, measured_distances_phase)
                x_vals = np.array(true_distances)
                y_vals = x_vals + intercept
                axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
                print(f"INTERCEPT {mult_type} PHASE {anchor.name}: y=x+{intercept:.2f}")

        elif type == "RSSI":
            plot = axes[id].scatter(true_distances, measured_distances_rssi, color='blue')
            axes[id].scatter(true_distances, measured_distances_rssi_min, color='black', marker='2', s=11)
            axes[id].scatter(true_distances, measured_distances_rssi_max, color='black', marker='1', s=11)
            for i in range(len(true_distances)):
                axes[id].plot([true_distances[i], true_distances[i]],
                              [measured_distances_rssi_min[i], measured_distances_rssi_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            #regresja
            # if regression:
            #     slope, intercept, _, _, _ = linregress(true_distances, measured_distances_rssi)
            #     x_vals = np.array(true_distances)
            #     y_vals = x_vals + intercept
            #     axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
            #     print(f"INTERCEPT {mult_type} RSSI {anchor.name}: y=x+{intercept:.2f}")

            axes[id].set_ylim(0,40)
        elif type == "BEST":
            plot = axes[id].scatter(true_distances, measured_distances_best, color='orange')
            axes[id].scatter(true_distances, measured_distances_best_min, color='black', marker='2', s=11)
            axes[id].scatter(true_distances, measured_distances_best_max, color='black', marker='1', s=11)
            for i in range(len(true_distances)):
                axes[id].plot([true_distances[i], true_distances[i]],
                              [measured_distances_best_min[i], measured_distances_best_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            # if regression:
            #     slope, intercept, _, _, _ = linregress(true_distances, measured_distances_best)
            #     x_vals = np.array(true_distances)
            #     y_vals = x_vals + intercept
            #     axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
            #     print(f"INTERCEPT {mult_type} BEST {anchor.name}: y=x+{intercept:.2f}")

        axes[id].plot(true_distances, true_distances, color='black', linestyle='--')

        axes[id].set_xlabel('Prawdziwy dystans [m]')
        axes[id].set_ylabel('Zmierzony dystans [m]')
        axes[id].grid(True)

        axes[id].set_title(f'{type}')
        #axes[id].legend()

    fig.suptitle(f"Porównanie jakości zmierzonych odległości dla kotwicy {anchor.name} - {mult_type}")
    plt.tight_layout()


    if save_to_file:
        file_name = f".//{report_dir}//"+ f"meas_quality_{anchor.name.split(':')[0]}_{mult_type}_minmax"
        plt.savefig(file_name)


    plt.show()


def add_measurements_to_schema(ax, measured_points, real_points, color):
    for (mx, my), (rx, ry) in zip(measured_points, real_points):
        ax.scatter(mx, my, color=color, s=50)  # zielone punkty pomiarowe
        ax.plot([mx, rx], [my, ry], color='black', linestyle='--', linewidth=1)
def create_schema_with_results(loc_res_list, mult_type='avg', all_types=True,save_to_file=True, report_dir='reporting'):
    localized_points_ifft = []
    localized_points_phase = []
    localized_points_rssi = []
    localized_points_best = []
    real_points = []

    for loc_res in loc_res_list:
        rp = (loc_res.x, loc_res.y)
        real_points.append(rp)
        if mult_type == 'avg':
            localized_points_ifft.append((loc_res.avg_mult['IFFT'].x, loc_res.avg_mult['IFFT'].y))
            localized_points_phase.append((loc_res.avg_mult['PHASE'].x, loc_res.avg_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.avg_mult['RSSI'].x, loc_res.avg_mult['RSSI'].y))
            localized_points_best.append((loc_res.avg_mult['BEST'].x, loc_res.avg_mult['BEST'].y))

        elif mult_type == 'med':
            localized_points_ifft.append((loc_res.med_mult['IFFT'].x, loc_res.med_mult['IFFT'].y))
            localized_points_phase.append((loc_res.med_mult['PHASE'].x, loc_res.med_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.med_mult['RSSI'].x, loc_res.med_mult['RSSI'].y))
            localized_points_best.append((loc_res.med_mult['BEST'].x, loc_res.med_mult['BEST'].y))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    make_default_schema(axes[0,0])
    make_default_schema(axes[0,1])
    make_default_schema(axes[1,0])
    make_default_schema(axes[1,1])

    add_measurements_to_schema(axes[0,0], localized_points_ifft, real_points, color='red')
    axes[0,0].set_title("IFFT")
    add_measurements_to_schema(axes[0, 1], localized_points_phase, real_points, color='green')
    axes[0, 1].set_title("PHASE")
    add_measurements_to_schema(axes[1, 0], localized_points_rssi, real_points, color='purple')
    axes[1, 0].set_title("RSSI")
    add_measurements_to_schema(axes[1, 1], localized_points_best, real_points, color='orange')
    axes[1, 1].set_title("BEST")

    fig.suptitle(f'Porównanie punktów zlokalizowanych z prawdziwymi - {mult_type}')

    plt.tight_layout()

    if save_to_file:
        file_name = f".//{report_dir}//" + f"loc_quality_{mult_type}"
        plt.savefig(file_name)

    plt.show()

def calculate_distances_heatmap(localized_points, real_points):
    distances=[]
    for lp, rp in zip(localized_points, real_points):
        distance = np.linalg.norm(np.array(lp) - np.array(rp))
        distances.append(distance)
    return distances

def add_heatmap_to_schema(ax, localized_points, real_points, cmap='viridis', vmin=None, vmax=None):
    distances = calculate_distances_heatmap(localized_points, real_points)
    sc = ax.scatter(*zip(*real_points), c=distances, cmap=cmap, s=50, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="Różnica dystansu")

def create_heat_map_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting'):
    localized_points_ifft = []
    localized_points_phase = []
    localized_points_rssi = []
    localized_points_best = []
    real_points = []

    for loc_res in loc_res_list:
        rp = (loc_res.x, loc_res.y)
        real_points.append(rp)
        if mult_type == 'avg':
            localized_points_ifft.append((loc_res.avg_mult['IFFT'].x, loc_res.avg_mult['IFFT'].y))
            localized_points_phase.append((loc_res.avg_mult['PHASE'].x, loc_res.avg_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.avg_mult['RSSI'].x, loc_res.avg_mult['RSSI'].y))
            localized_points_best.append((loc_res.avg_mult['BEST'].x, loc_res.avg_mult['BEST'].y))

        elif mult_type == 'med':
            localized_points_ifft.append((loc_res.med_mult['IFFT'].x, loc_res.med_mult['IFFT'].y))
            localized_points_phase.append((loc_res.med_mult['PHASE'].x, loc_res.med_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.med_mult['RSSI'].x, loc_res.med_mult['RSSI'].y))
            localized_points_best.append((loc_res.med_mult['BEST'].x, loc_res.med_mult['BEST'].y))

    fig, axes = plt.subplots(2, 2, figsize=(15, 15))

    make_default_schema(axes[0, 0])
    make_default_schema(axes[0, 1])
    make_default_schema(axes[1, 0])
    make_default_schema(axes[1, 1])

    distances_ifft = calculate_distances_heatmap(localized_points_ifft, real_points)
    distances_phase = calculate_distances_heatmap(localized_points_phase, real_points)
    distances_rssi = calculate_distances_heatmap(localized_points_rssi, real_points)
    distances_best = calculate_distances_heatmap(localized_points_best, real_points)

    all_distances = distances_ifft + distances_phase + distances_rssi + distances_best
    vmin = min(all_distances)
    vmax = max(all_distances)

    add_heatmap_to_schema(axes[0, 0], localized_points_ifft, real_points, cmap='Reds',vmin=vmin, vmax=vmax)
    axes[0, 0].set_title("IFFT")
    add_heatmap_to_schema(axes[0, 1], localized_points_phase, real_points, cmap='Greens',vmin=vmin, vmax=vmax)
    axes[0, 1].set_title("PHASE")
    add_heatmap_to_schema(axes[1, 0], localized_points_rssi, real_points, cmap='Purples',vmin=vmin, vmax=vmax)
    axes[1, 0].set_title("RSSI")
    add_heatmap_to_schema(axes[1, 1], localized_points_best, real_points, cmap='Oranges',vmin=vmin, vmax=vmax)
    axes[1, 1].set_title("BEST")

    fig.suptitle(f'Mapa Cieplna Lokalizacji - {mult_type}')

    plt.tight_layout()

    if save_to_file:
        file_name = f".//{report_dir}//" + f"loc_quality_{mult_type}"
        plt.savefig(file_name)

    plt.show()

    pass

def get_total_number_of_measurements(loc_res_list):
    total = 0
    for loc_res in loc_res_list:
        for anchor in loc_res.meas_dict.keys():
            total = total + len(loc_res.meas_dict[anchor])


    return total

def get_bias(loc_re_list):
    result_dic = {"FE": {}, "E4": {}, "F7": {}, "FC": {}}

    for loc_res in loc_re_list:
        if (loc_res.x, loc_res.y) in [(1,0), (9,0), (1,10), (9,10)]:
            if (loc_res.x, loc_res.y) == (1, 0):
                for meas in loc_res.avg_meas:
                    if meas.anchor.split(":")[0] == "FE":
                        result_dic["FE"]["IFFT"] = meas.ifft - 1
                        result_dic["FE"]["PHASE"] = meas.phase_slope - 1
                        result_dic["FE"]["RSSI"] = meas.rssi_openspace - 1
                        result_dic["FE"]["BEST"] = meas.best - 1
            if (loc_res.x, loc_res.y) == (9, 0):
                for meas in loc_res.avg_meas:
                    if meas.anchor.split(":")[0] == "E4":
                        result_dic["E4"]["IFFT"] = meas.ifft - 1
                        result_dic["E4"]["PHASE"] = meas.phase_slope - 1
                        result_dic["E4"]["RSSI"] = meas.rssi_openspace - 1
                        result_dic["E4"]["BEST"] = meas.best - 1
            if (loc_res.x, loc_res.y) == (1, 10):
                for meas in loc_res.avg_meas:
                    if meas.anchor.split(":")[0] == "F7":
                        result_dic["F7"]["IFFT"] = meas.ifft - 1
                        result_dic["F7"]["PHASE"] = meas.phase_slope - 1
                        result_dic["F7"]["RSSI"] = meas.rssi_openspace - 1
                        result_dic["F7"]["BEST"] = meas.best - 1
            if (loc_res.x, loc_res.y) == (9, 10):
                for meas in loc_res.avg_meas:
                    if meas.anchor.split(":")[0] == "FC":
                        result_dic["FC"]["IFFT"] = meas.ifft - 1
                        result_dic["FC"]["PHASE"] = meas.phase_slope - 1
                        result_dic["FC"]["RSSI"] = meas.rssi_openspace - 1
                        result_dic["FC"]["BEST"] = meas.best - 1
    return result_dic

def get_avg_error_distance_meas_per_anchor(loc_res_list, anchor):
    true_distances = []
    ifft_differences = []
    phase_differences = []
    rssi_differences = []
    best_differences = []
    #print(loc_res_list[0].avg_meas)

    for loc_res in loc_res_list:
        for meas in loc_res.avg_meas:
            if meas.anchor == anchor.name:
                td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))
                true_distances.append(td)
                ifft_diff = abs(td - meas.ifft)
                ifft_differences.append(ifft_diff)
                phase_diff = abs(td - meas.phase_slope)
                phase_differences.append(phase_diff)
                rssi_diff = abs(td - meas.rssi_openspace)
                rssi_differences.append(rssi_diff)
                best_diff = abs(td - meas.best)
                best_differences.append(best_diff)

    return {
        "IFFT": statistics.mean(ifft_differences),
        "PHASE": statistics.mean(phase_differences),
        "RSSI": statistics.mean(rssi_differences),
        "BEST": statistics.mean(best_differences)
    }


anchors = [Anchor(boards[2], 0, 0, 1.9), #FE
           Anchor(boards[1], 10, 0, 1.9), #E4
           Anchor(boards[3], 0, 10, 1.9), #F7
           Anchor(boards[0], 10, 10, 1.9)] #FC


if __name__ == '__main__':
    print("START")


    analyze_all_files("wyniki_pomiarów", "reporting", multilateration_type="2D", clear_failed_meas=True, )
    loc_res_list = load_loc_res_from_file("reporting/results.pkl") #lista obiektów klasy location_measurement_results



    for anchor in anchors:
        create_scatter_plots_for_anchor_minmax(loc_res_list, anchor=anchor, mult_type="avg", save_to_file=True,
                                               report_dir='reporting//distance')
        create_scatter_plots_for_anchor_minmax(loc_res_list, anchor=anchor, mult_type="med", save_to_file=True,
                                               report_dir='reporting//distance')
    create_schema_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting//localisation')
    create_schema_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir='reporting//localisation')
    create_heat_map_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting//heatmap')
    create_heat_map_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir='reporting//heatmap')
    # for loc_res in loc_res_list:
    #     print(f'POINT: ({loc_res.x}, {loc_res.y})')
    #     print(f'MIN: {loc_res.get_min_measurements_per_anchor("FE:5A:0F:0E:29:6F")}')
    #     print(f'MAX: {loc_res.get_max_measurements_per_anchor("FE:5A:0F:0E:29:6F")}')
    #     for avg in loc_res.avg_meas:
    #         if avg.anchor.split(":")[0] == "FE":
    #             print(f'AVG: {avg}')

    bias_dict_reg = {'FE': {'IFFT': 0.97, 'PHASE': 6.50, 'RSSI': 0, 'BEST': 0.97},
                     'E4': {'IFFT': 1.12, 'PHASE': 3.07, 'RSSI': 0, 'BEST': 1.12},
                     'F7': {'IFFT': 0.09, 'PHASE': 2.11, 'RSSI': 0, 'BEST': 0.09},
                     'FC': {'IFFT': 1.44, 'PHASE': 4.16, 'RSSI': 0, 'BEST': 1.44}}
    #
    analyze_all_files("wyniki_pomiarów", "testowe", multilateration_type='2D', clear_failed_meas=True, bias=bias_dict_reg)
    loc_res_unbiased = load_loc_res_from_file("testowe/results.pkl")

    for anchor in anchors:
        create_scatter_plots_for_anchor_minmax(loc_res_unbiased, anchor=anchor, mult_type="avg", save_to_file=True,
                                               report_dir='reporting//distance_bias')
        create_scatter_plots_for_anchor_minmax(loc_res_unbiased, anchor=anchor, mult_type="med", save_to_file=True,
                                               report_dir='reporting//distance_bias')


    create_schema_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True, report_dir='reporting//localisation_bias')
    create_schema_with_results(loc_res_unbiased, mult_type='med', save_to_file=True, report_dir='reporting//localisation_bias')
    create_heat_map_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True, report_dir='reporting//heatmap_bias')
    create_heat_map_with_results(loc_res_unbiased, mult_type='med', save_to_file=True, report_dir='reporting//heatmap_bias')

