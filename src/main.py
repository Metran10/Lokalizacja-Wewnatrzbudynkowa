import math
import os
import pickle
import random
import re
import csv
import numpy as np
import pandas
#import localization as Loc
import scipy as sc
import localization as lx
import pandas as pd
import matplotlib.pyplot as plt
from conda_libmamba_solver import solver
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares
import statistics
import json
from utils import *
from schema import make_default_schema, create_default_schema
from scipy.stats import linregress
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import curve_fit

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

                data_sets.append(ds)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12,10))
    axes = axes.flatten()

    for id, anchor in enumerate(Anchors):
        boxprops = dict(linestyle='-', linewidth=2, color='k')
        medianprops = dict(linestyle='-', linewidth=2, color='firebrick')

        if id > len(data_sets) - 1:
            break
        boxplot = axes[id].boxplot(data_sets[id], patch_artist=True, boxprops=boxprops, medianprops=medianprops)
        medians = [item.get_ydata()[0] for item in boxplot['medians']]
        for i, median in enumerate(medians):
            axes[id].text(x=i + 1.3, y=median, s=f'{str(round(median,2))}', verticalalignment='center', color='black')

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


def create_boxplots_for_loc_res_list(loc_res_list, Anchors, save_to_file=False, save_path='reporting//box_plots_bias'):
    for loc_res in loc_res_list:
        create_boxplots_for_measurement(Anchors, loc_res.meas_dict, (loc_res.x, loc_res.y), remove_outliers=False, save_to_file=save_to_file, save_path=save_path)


    pass
def print_avg_measurements(avg_meas):
    for anch in avg_meas:
        print(anch)

def clear_failed_measurements(ml):
    new_list = []
    for meas in ml:
        if meas.quality == "ok":
            new_list.append(meas)

    return new_list



def locate_point(avg_meas, type='3D', og_point=None, solver='LSE'):
    lp = lx.Project(mode=type, solver=solver)
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

def analyze_all_files(base_dir, report_dir, multilateration_type='2D', clear_failed_meas=True, bias=None, create_boxplots=False, solver='LSE'):
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
                avg_mult_res = locate_point(avg_meas, multilateration_type, (x,y,1.6), solver=solver)
                # print("MEDIAN")
                # print_avg_measurements(med_meas)
                # if x == 1 and y ==0:
                #     print(avg_meas)
                #     for me in avg_meas:
                #         print(f'({x}, {y}) AVG: {me.anchor}: {me.ifft}')


                med_mult_res = locate_point(med_meas, multilateration_type, (x,y,1.6), solver=solver)

                location_results.append(location_measurements_results(
                    x=x,
                    y=y,
                    meas_dict=sorted_meas,
                    avg_meas=avg_meas,
                    med_meas=med_meas,
                    avg_mult=avg_mult_res,
                    med_mult=med_mult_res,
                    avg_trilat=None
                ))

    #do pliku json
    #dict_list = [loc_res.to_dict() for loc_res in location_results]
    save_loc_res_to_file(f"{report_dir}//results.pkl", location_results)


def analyze_all_files_improved(base_dir, report_dir, file_name, mult_type='2D', clear_failed_meas=True, bias=None, trilaterations=False, data_batch=None):
    location_results = []

    trilateration_combinations = [
        'FE_E4_FC', 'FE_FC_F7', 'FE_E4_F7', 'F7_FC_E4'
    ]

    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)
                x, y = extract_location_cord_from_file(file)
                ml = read_measurements(file_path)
                if clear_failed_meas:
                    ml = clear_failed_measurements(ml)
                sorted_meas = sort_measurements_for_anchor(ml)
                if bias is not None:
                    sorted_meas = sub_bias(sorted_meas, bias)

                if data_batch is not None:
                    #wybranie x danych, albo maks dostepne jesli nie  ma x pomiarow
                    avg_meas = get_average_measurement_per_anchor_from_batch(sorted_meas, data_batch)
                    med_meas = get_median_measurement_per_anchor_from_batch(sorted_meas, data_batch)
                else:
                    avg_meas = get_average_measurement_per_anchor(sorted_meas)
                    med_meas = get_median_measurement_per_anchor(sorted_meas)

                if trilaterations:
                    avg_trilat_res = {}

                    for comb in trilateration_combinations:
                        comb_res = locate_point_trilateration(avg_meas, comb, '2D', (x,y,1.6), solver='LSE')

                        avg_trilat_res[comb] = comb_res

                    location_results.append(location_measurements_results(
                        x=x,
                        y=y,
                        meas_dict=sorted_meas,
                        avg_meas=avg_meas,
                        med_meas=med_meas,
                        avg_mult=None,
                        med_mult=None,
                        avg_trilat=avg_trilat_res
                    ))
                else:
                    print(avg_meas)
                    for me in avg_meas:
                        print(f'({x}, {y}) AVG: {me.anchor}: {me.ifft}')

                    avg_mult_res = locate_point(avg_meas, mult_type, (x, y, 1.6), solver='LSE')
                    #avg_mult_res = locate_point(avg_meas, mult_type, (x,y,1.6), solver=solver)
                    med_mult_res = locate_point(med_meas, mult_type, (x,y,1.6), solver='LSE')

                    location_results.append(location_measurements_results(
                        x=x,
                        y=y,
                        meas_dict=sorted_meas,
                        avg_meas=avg_meas,
                        med_meas=med_meas,
                        avg_mult=avg_mult_res,
                        med_mult=med_mult_res,
                        avg_trilat=None
                    ))
    save_loc_res_to_file(f"{report_dir}//{file_name}", location_results)


def locate_point_trilateration(avg_meas, anchor_name_list, mult_type='2D',og_point=None, solver='LSE'):
    lp = lx.Project(mode=mult_type, solver=solver)

    anchor_name_list = anchor_name_list.split('_')
    for anchor in anchors:
        if anchor.name.split(':')[0] in anchor_name_list:
            lp.add_anchor(anchor.name, (anchor.x_cord, anchor.y_cord, anchor.z_cord))


    ifft, ifftlabel = lp.add_target()
    phase, phaselabel = lp.add_target()
    rssi, rssilabel = lp.add_target()
    best, bestlabel = lp.add_target()

    for meas in avg_meas:
        if meas.anchor.split(":")[0] in anchor_name_list:
            ifft.add_measure(meas.anchor, meas.ifft)
            phase.add_measure(meas.anchor, meas.phase_slope)
            rssi.add_measure(meas.anchor, meas.rssi_openspace)
            best.add_measure(meas.anchor, meas.best)

    lp.solve()

    ifft_result = mult_result('IFFT', ifft.loc.x, ifft.loc.y, 1.9,
                              euclidean_dist_2points((ifft.loc.x, ifft.loc.y), (og_point[0], og_point[1]), False))
    phase_result = mult_result('PHASE', phase.loc.x, phase.loc.y, 1.9,
                              euclidean_dist_2points((phase.loc.x, phase.loc.y), (og_point[0], og_point[1]), False))
    rssi_result = mult_result('IFFT', rssi.loc.x, rssi.loc.y, 1.9,
                              euclidean_dist_2points((rssi.loc.x, rssi.loc.y), (og_point[0], og_point[1]), False))
    best_result = mult_result('IFFT', best.loc.x, best.loc.y, 1.9,
                              euclidean_dist_2points((best.loc.x, best.loc.y), (og_point[0], og_point[1]), False))

    res_dict = {'IFFT': ifft_result,
                'PHASE': phase_result,
                'RSSI': rssi_result,
                'BEST': best_result}
    print(f"TRILATERATION {anchor_name_list} for ({og_point[0]},{og_point[1]})")
    print(f"IFFF loc: {ifft.loc}: diff:{res_dict['IFFT'].distance_from_point}")
    print(f"PHASE loc: {phase.loc}: diff:{res_dict['PHASE'].distance_from_point}")
    print(f"RSSI loc: {rssi.loc}: diff:{res_dict['IFFT'].distance_from_point}")
    print(f"BEST loc: {best.loc}: diff:{res_dict['BEST'].distance_from_point}")

    return res_dict

def get_average_measurement_per_anchor_from_batch(sorted_meas, data_batch):
    avg_meas = []

    for anchor in sorted_meas.keys():
        #wybranie x danych z listy, jesli nie ma to cała
        meas_sample = data_batch
        if meas_sample > len(sorted_meas[anchor]):
            meas_sample = len(sorted_meas[anchor])
        sorted_meas[anchor] = random.sample(sorted_meas[anchor], meas_sample)

        meas = get_average_distances(sorted_meas[anchor])
        avg_meas.append(meas)

    return avg_meas

def get_median_measurement_per_anchor_from_batch(sorted_meas, data_batch):
    med_meas = []

    for anchor in sorted_meas.keys():
        #wybranie x danych z listy
        meas_sample = data_batch
        if meas_sample > len(sorted_meas[anchor]):
            meas_sample = len(sorted_meas[anchor])
        sorted_meas[anchor] = random.sample(sorted_meas[anchor], meas_sample)

        meas = get_med_measurement(sorted_meas[anchor])
        med_meas.append(meas)

    return med_meas


def sub_bias(sorted_meas, bias):
    # print(sorted_meas)
    # print(bias)

    for anchor in sorted_meas.keys():
        print(anchor)
        for meas in sorted_meas[anchor]:
            meas.ifft = float(meas.ifft) - float(bias[anchor.split(":")[0]]['IFFT'])
            meas.phase_slope = float(meas.phase_slope) - float(bias[anchor.split(":")[0]]['PHASE'])
            meas.rssi_openspace = (float(meas.rssi_openspace) - float(bias[anchor.split(":")[0]]['RSSI']))/1.35
            if meas.rssi_openspace < 0:
                meas.rssi_openspace = 0.01
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

    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    # axes = axes.flatten()
    fig = plt.figure(figsize=(12, 10))
    gs = GridSpec(4,4,figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax3 = fig.add_subplot(gs[2:4, 1:3])


    for id, type in enumerate(["IFFT", "PHASE", "RSSI", "BEST"]):
        if type == "IFFT":
            # plot = axes[id].scatter(true_distances, measured_distances_ifft, color='red')
            # axes[id].set_ylim(0,25)
            ax1.scatter(true_distances, measured_distances_ifft, color='red')
            #ax1.set_ylim(0, 25)
            ax1.plot(true_distances, true_distances, color='black', linestyle='--')
            ax1.set_xlabel('Prawdziwy dystans [m]')
            ax1.set_ylabel('Zmierzony dystans [m]')
            ax1.grid(True)
            ax1.set_title(f'{type}')
        elif type == "PHASE":
            # plot = axes[id].scatter(true_distances, measured_distances_phase, color='green')
            # axes[id].set_ylim(0, 25)
            ax2.scatter(true_distances, measured_distances_phase, color='green')
            #ax2.set_ylim(0, 25)
            ax2.plot(true_distances, true_distances, color='black', linestyle='--')
            ax2.set_xlabel('Prawdziwy dystans [m]')
            ax2.set_ylabel('Zmierzony dystans [m]')
            ax2.grid(True)
            ax2.set_title(f'{type}')
        elif type == "RSSI":
            # plot = axes[id].scatter(true_distances, measured_distances_rssi, color='blue')
            # axes[id].set_ylim(0, 25)
            ax3.scatter(true_distances, measured_distances_rssi, color='blue')
            #ax3.set_ylim(0, 25)
            ax3.plot(true_distances, true_distances, color='black', linestyle='--')
            ax3.set_xlabel('Prawdziwy dystans [m]')
            ax3.set_ylabel('Zmierzony dystans [m]')
            ax3.grid(True)
            ax3.set_title(f'{type}')
        # elif type == "BEST":
        #     plot = axes[id].scatter(true_distances, measured_distances_best, color='orange')
        #     axes[id].set_ylim(0, 25)

        # axes[id].plot(true_distances, true_distances, color='black', linestyle='--')
        #
        # axes[id].set_xlabel('Prawdziwy dystans [m]')
        # axes[id].set_ylabel('Zmierzony dystans [m]')
        # axes[id].grid(True)
        #
        # axes[id].set_title(f'{type}')

    fig.suptitle(f"Porównanie jakości {'średnich' if mult_type == 'avg' else 'mediany'} zmierzonych odległości dla kotwicy {anchor.name}")
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


    # fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 10))
    # axes = axes.flatten()

    # for id, type in enumerate(["IFFT", "PHASE", "RSSI", "BEST"]):
    #     if type == "IFFT":
    #         plot = axes[id].scatter(true_distances, measured_distances_ifft, color='red')
    #         axes[id].scatter(true_distances, measured_distances_ifft_min, color='black', marker='2', s=11)
    #         axes[id].scatter(true_distances, measured_distances_ifft_max, color='black', marker='1', s=11)
    #         #linie łączące
    #         for i in range(len(true_distances)):
    #             axes[id].plot([true_distances[i], true_distances[i]],
    #                           [measured_distances_ifft_min[i], measured_distances_ifft_max[i]], color='black',
    #                           linestyle='-', linewidth=0.2)
    #         #regresja
    #         if regression:
    #             slope, intercept, _,_, _ = linregress(true_distances, measured_distances_ifft)
    #             x_vals = np.array(true_distances)
    #             y_vals = x_vals + intercept
    #             axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
    #             print(f"INTERCEPT {mult_type} IFFT {anchor.name}: y=x+{intercept:.2f}")
    #
    #     elif type == "PHASE":
    #         plot = axes[id].scatter(true_distances, measured_distances_phase, color='green')
    #         axes[id].scatter(true_distances, measured_distances_phase_min, color='black', marker='2', s=11)
    #         axes[id].scatter(true_distances, measured_distances_phase_max, color='black', marker='1', s=11)
    #         for i in range(len(true_distances)):
    #             axes[id].plot([true_distances[i], true_distances[i]],
    #                           [measured_distances_phase_min[i], measured_distances_phase_max[i]], color='black',
    #                           linestyle='-', linewidth=0.2)
    #         axes[id].set_ylim(0,30)
    #         #regresja
    #         if regression:
    #             slope, intercept, _, _, _ = linregress(true_distances, measured_distances_phase)
    #             x_vals = np.array(true_distances)
    #             y_vals = x_vals + intercept
    #             axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
    #             print(f"INTERCEPT {mult_type} PHASE {anchor.name}: y=x+{intercept:.2f}")
    #
    #     elif type == "RSSI":
    #         plot = axes[id].scatter(true_distances, measured_distances_rssi, color='blue')
    #         axes[id].scatter(true_distances, measured_distances_rssi_min, color='black', marker='2', s=11)
    #         axes[id].scatter(true_distances, measured_distances_rssi_max, color='black', marker='1', s=11)
    #         for i in range(len(true_distances)):
    #             axes[id].plot([true_distances[i], true_distances[i]],
    #                           [measured_distances_rssi_min[i], measured_distances_rssi_max[i]], color='black',
    #                           linestyle='-', linewidth=0.2)
    #         #regresja
    #         # if regression:
    #         #     slope, intercept, _, _, _ = linregress(true_distances, measured_distances_rssi)
    #         #     x_vals = np.array(true_distances)
    #         #     y_vals = x_vals + intercept
    #         #     axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
    #         #     print(f"INTERCEPT {mult_type} RSSI {anchor.name}: y=x+{intercept:.2f}")
    #
    #         axes[id].set_ylim(0,40)
    #     elif type == "BEST":
    #         plot = axes[id].scatter(true_distances, measured_distances_best, color='orange')
    #         axes[id].scatter(true_distances, measured_distances_best_min, color='black', marker='2', s=11)
    #         axes[id].scatter(true_distances, measured_distances_best_max, color='black', marker='1', s=11)
    #         for i in range(len(true_distances)):
    #             axes[id].plot([true_distances[i], true_distances[i]],
    #                           [measured_distances_best_min[i], measured_distances_best_max[i]], color='black',
    #                           linestyle='-', linewidth=0.2)
    #         # if regression:
    #         #     slope, intercept, _, _, _ = linregress(true_distances, measured_distances_best)
    #         #     x_vals = np.array(true_distances)
    #         #     y_vals = x_vals + intercept
    #         #     axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
    #         #     print(f"INTERCEPT {mult_type} BEST {anchor.name}: y=x+{intercept:.2f}")
    #
    #     axes[id].plot(true_distances, true_distances, color='black', linestyle='--')
    #
    #     axes[id].set_xlabel('Prawdziwy dystans [m]')
    #     axes[id].set_ylabel('Zmierzony dystans [m]')
    #     axes[id].grid(True)
    #
    #     axes[id].set_title(f'{type}')
    #     #axes[id].legend()

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax3 = fig.add_subplot(gs[2:4, 1:3])

    for id, type in enumerate(["IFFT", "PHASE", "RSSI", "BEST"]):
        if type == "IFFT":
            plot = ax1.scatter(true_distances, measured_distances_ifft, color='red')
            ax1.scatter(true_distances, measured_distances_ifft_min, color='black', marker='2', s=11)
            ax1.scatter(true_distances, measured_distances_ifft_max, color='black', marker='1', s=11)
            #linie łączące
            for i in range(len(true_distances)):
                ax1.plot([true_distances[i], true_distances[i]],
                              [measured_distances_ifft_min[i], measured_distances_ifft_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            #regresja
            if regression:
                slope, intercept, r_value,_, _ = linregress(true_distances, measured_distances_ifft)
                x_vals = np.array(true_distances)
                y_vals = x_vals + intercept
                ax1.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
                print(f"INTERCEPT {mult_type} IFFT {anchor.name}: y=x+{intercept:.2f}")
                print(f'R^2: {r_value**2}')

            ax1.plot(true_distances, true_distances, color='black', linestyle='--')

            ax1.set_xlabel('Prawdziwy dystans [m]')
            ax1.set_ylabel('Zmierzony dystans [m]')
            ax1.grid(True)

            ax1.set_title(f'{type}')

        elif type == "PHASE":
            plot = ax2.scatter(true_distances, measured_distances_phase, color='green')
            ax2.scatter(true_distances, measured_distances_phase_min, color='black', marker='2', s=11)
            ax2.scatter(true_distances, measured_distances_phase_max, color='black', marker='1', s=11)
            for i in range(len(true_distances)):
                ax2.plot([true_distances[i], true_distances[i]],
                              [measured_distances_phase_min[i], measured_distances_phase_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            ax2.set_ylim(0,30)
            #regresja
            if regression:
                slope, intercept, r_value, _, _ = linregress(true_distances, measured_distances_phase)
                x_vals = np.array(true_distances)
                y_vals = x_vals + intercept
                ax2.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
                print(f"INTERCEPT {mult_type} PHASE {anchor.name}: y=x+{intercept:.2f}")
                print(f'R^2: {r_value ** 2}')




            ax2.plot(true_distances, true_distances, color='black', linestyle='--')

            ax2.set_xlabel('Prawdziwy dystans [m]')
            ax2.set_ylabel('Zmierzony dystans [m]')
            ax2.grid(True)

            ax2.set_title(f'{type}')
            #ax2.legend()
        elif type == "RSSI":
            plot = ax3.scatter(true_distances, measured_distances_rssi, color='blue')
            ax3.scatter(true_distances, measured_distances_rssi_min, color='black', marker='2', s=11)
            ax3.scatter(true_distances, measured_distances_rssi_max, color='black', marker='1', s=11)
            for i in range(len(true_distances)):
                ax3.plot([true_distances[i], true_distances[i]],
                              [measured_distances_rssi_min[i], measured_distances_rssi_max[i]], color='black',
                              linestyle='-', linewidth=0.2)
            #regresja
            if regression:
                slope, intercept, r_value, _, _ = linregress(true_distances, measured_distances_rssi)
                x_vals = np.array(true_distances)
                y_vals = slope * x_vals + intercept
                ax3.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
                print(f"INTERCEPT {mult_type} RSSI {anchor.name}: y=x+{intercept:.2f}")
                print(f'SLOPE: {slope}')
                print(f'R^2: {r_value ** 2}')

            ax3.set_ylim(0,40)
            ax3.plot(true_distances, true_distances, color='black', linestyle='--')

            ax3.set_xlabel('Prawdziwy dystans [m]')
            ax3.set_ylabel('Zmierzony dystans [m]')
            ax3.grid(True)

            ax3.set_title(f'{type}')

        # elif type == "BEST":
        #     plot = axes[id].scatter(true_distances, measured_distances_best, color='orange')
        #     axes[id].scatter(true_distances, measured_distances_best_min, color='black', marker='2', s=11)
        #     axes[id].scatter(true_distances, measured_distances_best_max, color='black', marker='1', s=11)
        #     for i in range(len(true_distances)):
        #         axes[id].plot([true_distances[i], true_distances[i]],
        #                       [measured_distances_best_min[i], measured_distances_best_max[i]], color='black',
        #                       linestyle='-', linewidth=0.2)
            # if regression:
            #     slope, intercept, _, _, _ = linregress(true_distances, measured_distances_best)
            #     x_vals = np.array(true_distances)
            #     y_vals = x_vals + intercept
            #     axes[id].plot(x_vals, y_vals, color='black', linestyle='--', linewidth=0.5)
            #     print(f"INTERCEPT {mult_type} BEST {anchor.name}: y=x+{intercept:.2f}")

        # axes[id].plot(true_distances, true_distances, color='black', linestyle='--')
        #
        # axes[id].set_xlabel('Prawdziwy dystans [m]')
        # axes[id].set_ylabel('Zmierzony dystans [m]')
        # axes[id].grid(True)
        #
        # axes[id].set_title(f'{type}')
        # axes[id].legend()

    fig.suptitle(f"Porównanie jakości zmierzonych odległości dla kotwicy {anchor.name}")
    plt.tight_layout()


    if save_to_file:
        file_name = f".//{report_dir}//"+ f"meas_quality_{anchor.name.split(':')[0]}_{mult_type}_minmax"
        plt.savefig(file_name)


    plt.show()


def create_scatter_plots_for_anchor_all_methods(loc_res_list, anchor,anchor_name='Anchor 1', correction=False):
    true_distances=[]

    measured_dstiances_ifft=[]
    measured_distances_phase=[]
    measured_distances_rssi=[]

    for loc_res in loc_res_list:
        td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))
        true_distances.append(td)

        for meas in loc_res.avg_meas:
            if meas.anchor == anchor.name:
                measured_dstiances_ifft.append(meas.ifft)
                measured_distances_phase.append(meas.phase_slope)
                measured_distances_rssi.append(meas.rssi_openspace)

        #if len(tru)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot()

    ax.scatter(true_distances, measured_dstiances_ifft, color='red')
    ax.scatter(true_distances, measured_distances_phase, color='green')
    ax.scatter(true_distances, measured_distances_rssi, color='blue')

    ax.plot(true_distances, true_distances, color='black', linestyle='--')
    ax.set_xlabel('Real Distance [m]', fontsize=16)
    ax.set_ylabel('Measured Distance [m]', fontsize=16)
    ax.grid(True)
    ax.set_title(f"Distance Measurements Comparison for {anchor_name}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    ax.set_ylim(0,30)
    ax.set_xlim(0,15)

    plt.savefig(f"dist_meas_comp_{anchor_name}{'_improved' if correction else ''}.png")

    plt.show()

    pass

def create_scatter_plot_for_anchor_every_measurement(loc_res_list, anchor, anchor_name="Anchor 1", correction=False):
    true_distances=[]

    ifft_lists= []
    phase_lists = []
    rssi_lists = []

    for loc_res in loc_res_list:
        td = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))
        true_distances.append(td)

        for anch in loc_res.meas_dict.keys():
            if anch == anchor.name:
                ifft_curr_list = []
                phase_curr_list = []
                rssi_curr_list = []
                for meas in loc_res.meas_dict[anch]:
                    ifft_curr_list.append(float(meas.ifft))
                    phase_curr_list.append(float(meas.phase_slope))
                    rssi_curr_list.append(float(meas.rssi_openspace))

        ifft_lists.append(ifft_curr_list)
        phase_lists.append(phase_curr_list)
        rssi_lists.append(rssi_curr_list)

    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot()

    for id, td in enumerate(true_distances):
        print(id)
        td_supp = [td] * len(ifft_lists[id])

        # print(len(td_supp))
        # print(len(true_distances))
        # print(len(ifft_lists[id]))
        ax.scatter(td_supp, ifft_lists[id], color='red')
        ax.scatter(td_supp, phase_lists[id], color='green')
        ax.scatter(td_supp, rssi_lists[id], color='blue')

    print("SCATTTERED")
    ax.plot(true_distances, true_distances, color='black', linestyle='--')
    ax.set_xlabel('Real Distance [m]', fontsize=16)
    ax.set_ylabel('Measured Distance [m]', fontsize=16)
    ax.grid(True)
    ax.set_title(f"Distance Measurements Comparison for {anchor_name}", fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=16)
    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.1)

    ax.set_ylim(0, 30)
    ax.set_xlim(0, 15)

    plt.savefig(f"all_dist_meas_comp_{anchor_name}{'_improved' if correction else ''}.png")

    plt.show()

def get_distance_errors_clear(loc_res_list, anchor):
    ifft_errors = []
    phase_errors = []
    rssi_errors = []

    for loc_res in loc_res_list:
        dist_to_point = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))

        for anch in loc_res.meas_dict.keys():
            if anch == anchor.name:
                for meas in loc_res.meas_dict[anch]:
                    ifft_errors.append(meas.ifft - dist_to_point)
                    phase_errors.append(meas.phase_slope - dist_to_point)
                    rssi_errors.append(meas.rssi_openspace - dist_to_point)

    return {
        'IFFT': ifft_errors,
        'PHASE': phase_errors,
        'RSSI': rssi_errors
    }


def create_cdf_all_errors(loc_res_list, loc_res_list_unbiased, anchor, anchor_name='Anchor 1'):

    error_list = get_distance_errors_clear(loc_res_list, anchor)
    error_list_bias = get_distance_errors_clear(loc_res_list_unbiased, anchor)

    error_dict = {
        'IFFT': error_list['IFFT'],
        'PHASE': error_list['PHASE'],
        'RSSI': error_list['RSSI'],
        'IFFT - improved': error_list_bias['IFFT'],
        'PHASE - improved': error_list_bias['PHASE'],
        'RSSI - improved': error_list_bias['RSSI']
    }
    plt.figure(figsize=(10, 10))
    colors = {
        'IFFT': 'red',
        'PHASE': 'green',
        'RSSI': 'blue'
    }

    for alg, errors in error_dict.items():
        sorted_data, cdf = compute_cdf(errors)
        color = colors[alg.replace(' - improved', '')]
        linestyle = '--' if ' - improved' in alg else '-'
        plt.plot(sorted_data, cdf, linestyle=linestyle, color=color, linewidth=3, label=alg)

    plt.xlabel('Distance measurement error [m]')
    plt.xlim(-5,15)
    plt.ylim(0,1)
    plt.title(f"Distance measurements error for {anchor_name}", fontsize=16)
    plt.grid(True)
    plt.legend()

    plt.savefig(f"all_errors_cdf_{anchor_name}.png")

    plt.show()


    pass



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

    # fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    #
    # make_default_schema(axes[0,0])
    # make_default_schema(axes[0,1])
    # make_default_schema(axes[1,0])
    # make_default_schema(axes[1,1])
    #
    # add_measurements_to_schema(axes[0,0], localized_points_ifft, real_points, color='red')
    # axes[0,0].set_title("IFFT")
    # add_measurements_to_schema(axes[0, 1], localized_points_phase, real_points, color='green')
    # axes[0, 1].set_title("PHASE")
    # add_measurements_to_schema(axes[1, 0], localized_points_rssi, real_points, color='purple')
    # axes[1, 0].set_title("RSSI")
    # add_measurements_to_schema(axes[1, 1], localized_points_best, real_points, color='orange')
    # axes[1, 1].set_title("BEST")

    fig = plt.figure(figsize=(12, 12))
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax3 = fig.add_subplot(gs[2:4, 1:3])

    make_default_schema(ax1)
    make_default_schema(ax2)
    make_default_schema(ax3)

    add_measurements_to_schema(ax1, localized_points_ifft, real_points, color='red')
    ax1.set_title("IFFT")
    add_measurements_to_schema(ax2, localized_points_phase, real_points, color='green')
    ax2.set_title("PHASE")
    add_measurements_to_schema(ax3, localized_points_rssi, real_points, color='purple')
    ax3.set_title("RSSI")


    fig.suptitle(f'Porównanie punktów zlokalizowanych z prawdziwymi w oparciu o {"średnią z" if mult_type == "avg" else "medianę z"} pomiarów')

    plt.tight_layout()

    if save_to_file:
        file_name = f".//{report_dir}//" + f"loc_quality_{mult_type}"
        plt.savefig(file_name)

    plt.show()

def create_single_schema_results(loc_res_list, type='IFFT',color="red", correction=False):
    localized_points = []
    real_points = []
    for loc_res in loc_res_list:
        rp = (loc_res.x, loc_res.y)
        real_points.append(rp)

        localized_points.append((loc_res.avg_mult[type].x, loc_res.avg_mult[type].y))

    fig = plt.figure(figsize=(12, 12))

    ax = fig.add_subplot()
    make_default_schema(ax)
    add_measurements_to_schema(ax, localized_points, real_points, color=color)
    ax.set_title(type, fontsize=16)

    file_name = f"{type}_loc{'_improved' if correction else ''}"
    plt.savefig(file_name)

    plt.show()

    pass

def calculate_distances_heatmap(localized_points, real_points):
    distances=[]
    for lp, rp in zip(localized_points, real_points):
        distance = np.linalg.norm(np.array(lp) - np.array(rp))
        distances.append(distance)
    return distances

my_cmaps = {
    'RED': LinearSegmentedColormap.from_list("my_red", [(0,'lightcoral'), (0.5, 'red'), (1, 'darkred')], N=256),
    'BLUE': LinearSegmentedColormap.from_list("my_blue", [(0,'lightblue'), (0.5, 'blue'), (1, 'darkblue')], N=256),
    'GREEN': LinearSegmentedColormap.from_list("my_green", [(0,'lightgreen'), (0.5, 'green'), (1, 'darkgreen')], N=256),
    'PURPLE': LinearSegmentedColormap.from_list("my_green", [(0,'lavender'), (0.5, 'purple'), (1, 'indigo')], N=256)
}


def add_heatmap_to_schema(ax, localized_points, real_points, cmap='viridis', vmin=None, vmax=None):
    distances = calculate_distances_heatmap(localized_points, real_points)
    sc = ax.scatter(*zip(*real_points), c=distances, cmap=cmap, s=50, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label="Różnica dystansu")

def create_heat_map_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting'):
    localized_points_ifft = []
    localized_points_phase = []
    localized_points_rssi = []
    real_points = []

    for loc_res in loc_res_list:
        rp = (loc_res.x, loc_res.y)
        real_points.append(rp)
        if mult_type == 'avg':
            localized_points_ifft.append((loc_res.avg_mult['IFFT'].x, loc_res.avg_mult['IFFT'].y))
            localized_points_phase.append((loc_res.avg_mult['PHASE'].x, loc_res.avg_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.avg_mult['RSSI'].x, loc_res.avg_mult['RSSI'].y))
        elif mult_type == 'med':
            localized_points_ifft.append((loc_res.med_mult['IFFT'].x, loc_res.med_mult['IFFT'].y))
            localized_points_phase.append((loc_res.med_mult['PHASE'].x, loc_res.med_mult['PHASE'].y))
            localized_points_rssi.append((loc_res.med_mult['RSSI'].x, loc_res.med_mult['RSSI'].y))

    fig = plt.figure(figsize=(15, 15))
    gs = GridSpec(4, 4, figure=fig)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    ax2 = fig.add_subplot(gs[0:2, 2:4])
    ax3 = fig.add_subplot(gs[2:4, 1:3])

    make_default_schema(ax1)
    make_default_schema(ax2)
    make_default_schema(ax3)

    distances_ifft = calculate_distances_heatmap(localized_points_ifft, real_points)
    distances_phase = calculate_distances_heatmap(localized_points_phase, real_points)
    distances_rssi = calculate_distances_heatmap(localized_points_rssi, real_points)

    all_distances = distances_ifft + distances_phase + distances_rssi
    vmin = min(all_distances)
    vmax = max(all_distances)

    add_heatmap_to_schema(ax1, localized_points_ifft, real_points, cmap=my_cmaps['RED'], vmin=0.5, vmax=15)
    ax1.set_title("IFFT")
    add_heatmap_to_schema(ax2, localized_points_phase, real_points, cmap=my_cmaps['GREEN'], vmin=0.5, vmax=15)
    ax2.set_title("PHASE")
    add_heatmap_to_schema(ax3, localized_points_rssi, real_points, cmap=my_cmaps['PURPLE'], vmin=0.5, vmax=15)
    ax3.set_title("RSSI")


    fig.suptitle(f'Mapa Cieplna Błędów Lokalizacji opartych o {"średnią" if mult_type=="avg" else "medianę"} z pomiarów')

    plt.tight_layout()

    if save_to_file:
        file_name = f".//{report_dir}//" + f"loc_quality_{mult_type}"
        plt.savefig(file_name)

    plt.show()

    pass

def get_measurement_number_for_loc(loc_res):
    meas_number = 0
    for anch in loc_res.meas_dict.keys():
        anch_num = len(loc_res.meas_dict[anch])
        meas_number += anch_num

    return meas_number

def calc_heat_meas_num(meas_num_list, points):
    values =[]
    for mn, p in zip(meas_num_list, points):
        val = np.linalg.norm(np.array(mn) - np.array(p))
        values.append(val)
    return values

def add_heatmap_meas_num(ax, meas_num_list, points, vmin=None, vmax=None):
    vals = calc_heat_meas_num(meas_num_list, points)
    sc = ax.scatter(*zip(*points), c=vals, cmap='Blues', s=50, vmin=vmin, vmax=vmax)
    plt.colorbar(sc, ax=ax, label='Liczba pomiarów')

def create_heatmap_number_of_measurements(loc_res_list, save_to_file=True, report_dir='reporting'):
    real_points = []
    measurements_number_points = []

    for loc_res in loc_res_list:
        #print(f'({loc_res.x}, {loc_res.y}): {get_measurement_number_for_loc(loc_res)}')
        rp = (loc_res.x, loc_res.y)
        real_points.append(rp)
        measurements_number_points.append(get_measurement_number_for_loc(loc_res))

    fig, ax = plt.subplots(figsize=(12, 10))

    make_default_schema(ax)
    add_heatmap_meas_num(ax, measurements_number_points, real_points, vmax=1000)

    if save_to_file:
        file_name = f".//{report_dir}//" + f"meas_num_heatmap.png"
        plt.savefig(file_name)

    plt.show()







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

def get_avg_error_distance_meas_per_anchor(loc_res_list, anchor, type='avg'):
    true_distances = []
    ifft_differences = []
    phase_differences = []
    rssi_differences = []
    best_differences = []
    #print(loc_res_list[0].avg_meas)

    if type=='avg':
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
    elif type=='med':
        for loc_res in loc_res_list:
            for meas in loc_res.med_meas:
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

def get_locations_error(loc_res_list, type='avg'):
    error_dict = {'IFFT': [], 'PHASE': [], "RSSI": []}

    for loc_res in loc_res_list:
        if type == 'avg':
            error_dict['IFFT'].append(loc_res.avg_mult['IFFT'].distance_from_point)
            error_dict['PHASE'].append(loc_res.avg_mult['PHASE'].distance_from_point)
            error_dict['RSSI'].append(loc_res.avg_mult['RSSI'].distance_from_point)
        if type == 'med':
            error_dict['IFFT'].append(loc_res.med_mult['IFFT'].distance_from_point)
            error_dict['PHASE'].append(loc_res.med_mult['PHASE'].distance_from_point)
            error_dict['RSSI'].append(loc_res.med_mult['RSSI'].distance_from_point)

    return error_dict

def create_loc_error_histogram(error_list, bins=20, alg_type='IFFT', mult_type='avg', save_to_file=True, report_dir='reporting', isBias=False):
    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(error_list, bins=bins, color='skyblue', edgecolor='black')

    quantiles = np.percentile(error_list, [25, 50, 75, 90, 95])
    quantiles_labels = ['25%', '50%', '75%', '90%', '95%']

    for quantile, label in zip(quantiles, quantiles_labels):
        plt.axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        plt.text(quantile, max(counts) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')



    plt.title(f"Histogram Błędów Lokalizacji {alg_type} z wykorzystaniem {'średniej z' if mult_type=='avg' else 'mediany z' } pomiarów {'po korekcie' if isBias else ''}")
    plt.xlabel("Błąd [m]")
    plt.ylabel("Wystąpienia")
    plt.grid(axis='y', alpha=0.75)

    if save_to_file:
        file_name = f".//{report_dir}//" + f"error_hist_{mult_type}_{alg_type}_{'bias' if isBias else ''}.png"
        plt.savefig(file_name)

    plt.show()

def create_loc_error_hist_comparision(error_list, error_list_bias, bins=20, alg_type='IFFT', mult_type='avg', save_to_file=True, report_dir='reporting', isBias=False):
    color='skyblue'

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))

    min_error = min(min(error_list), min(error_list_bias))
    max_error = max(max(error_list), max(error_list_bias))
    common_bins = np.linspace(min_error, max_error, bins)

    #przed korekta
    counts, bins, patches = axes[0].hist(error_list, bins=common_bins, color='skyblue', edgecolor='black')

    quantiles = np.percentile(error_list, [25, 50, 75, 90, 95])
    quantiles_labels = ['25%', '50%', '75%', '90%', '95%']

    for quantile, label in zip(quantiles, quantiles_labels):
        axes[0].axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        axes[0].text(quantile, max(counts) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')

    axes[0].set_title(
        f"Histogram Błędów Lokalizacji {alg_type} z wykorzystaniem {'średniej z' if mult_type == 'avg' else 'mediany z'} pomiarów")
    axes[0].set_xlabel("Błąd [m]")
    axes[0].set_ylabel("Wystąpienia")
    axes[0].grid(axis='y', alpha=0.75)

    if alg_type == 'PHASE':
        print("PRE")
        print(quantiles)
        print(len(error_list))
    #po korekcie
    counts_bias, bins_bias, patches_bias = axes[1].hist(error_list_bias, bins=common_bins, color='skyblue', edgecolor='black')




    quantiles_bias = np.percentile(error_list_bias, [25, 50, 75, 90, 95])
    quantiles_labels_bias = ['25%', '50%', '75%', '90%', '95%']

    for quantile, label in zip(quantiles_bias, quantiles_labels_bias):
        axes[1].axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        axes[1].text(quantile, max(counts_bias) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')

    axes[1].set_title(
        f"Histogram Błędów Lokalizacji {alg_type} z wykorzystaniem {'średniej z' if mult_type == 'avg' else 'mediany z'} pomiarów  po korekcie")
    axes[1].set_xlabel("Błąd [m]")
    axes[1].set_ylabel("Wystąpienia")
    axes[1].grid(axis='y', alpha=0.75)

    if alg_type == 'PHASE':
        print("POST")
        print(quantiles_bias)
        print(len(error_list_bias))
        print(error_list_bias)

    # max_bin=max(max(bins), max(bins_bias))
    # axes[0].set_xlim([0, 20])
    # axes[1].set_xlim([-5, 20])

    if save_to_file:
        file_name = f".//{report_dir}//" + f"error_hist_{mult_type}_{alg_type}_comparision.png"
        plt.savefig(file_name)

    plt.show()

    pass

def create_loc_error_histogram_all_algorithms(error_dict, bins=30, mult_type='avg', save_to_file=True, reporting_dir='reporting'):
    plt.figure(figsize=(10, 6))
    plt.xlim(0, 25)

    colors = {
        'IFFT': 'red',
        'PHASE': 'green',
        'RSSI': 'blue'
    }

    for alg in error_dict.keys():
        data = error_dict[alg]
        counts, bin_edges = np.histogram(data, bins=bins, range=(0,25))
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        plt.bar(bin_centers, counts, width=0.85, color=colors[alg], edgecolor='black', alpha=0.5, label=alg)


    plt.title("Porównanie błędów lokalizacji algorytmów")
    plt.xlabel("Błąd [m]")
    plt.ylabel("Wystąpienia")
    plt.grid(axis='y', alpha=0.75)
    plt.legend()

    plt.show()
    pass

def get_distance_errors(loc_res_list, anchor,type='avg'):
    ifft_errors = []
    phase_errors = []
    rssi_errors = []

    for loc_res in loc_res_list:
        dist_to_point = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))


        if type == 'avg':
            for avg_meas in loc_res.avg_meas:
                if avg_meas.anchor == anchor.name:
                    ifft_errors.append(abs(dist_to_point - avg_meas.ifft))
                    phase_errors.append(abs(dist_to_point - avg_meas.phase_slope))
                    rssi_errors.append(abs(dist_to_point - avg_meas.rssi_openspace))
    return {
        'IFFT': ifft_errors,
        'PHASE': phase_errors,
        'RSSI': rssi_errors
    }

def create_dist_error_histogram_for_anchor(loc_res_list, anchor, bins, type, algorithm, save_to_file=True, report_dir='reporting//dist_error_histogram', x_lim=(0,6)):
    # print(anchor.name)
    # print(get_avg_error_distance_meas_per_anchor(loc_res_list, anchor, type))
    color = 'skyblue'
    # if algorithm == 'IFFT':
    #     color = 'red'
    # elif algorithm == 'PHASE':
    #     color = 'green'
    # elif algorithm == 'RSSI':
    #     color = 'blue'



    error_list = get_distance_errors(loc_res_list, anchor, type=type)
    print(anchor.name)
    print(f'{algorithm} srednio: {np.mean(error_list[algorithm])}')



    plt.figure(figsize=(10, 6))
    counts, bins, patches = plt.hist(error_list[algorithm], bins=bins, color=color, edgecolor='black', alpha=0.75)
    quantiles = np.percentile(error_list[algorithm], [25, 50, 75, 90, 95])
    quantiles_labels = ['25%', '50%', '75%', '90%', '95%']
    for quantile, label in zip(quantiles, quantiles_labels):
        plt.axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        plt.text(quantile, max(counts) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')
    plt.title(f"Histogram Błędów Pomiarów Dystansu {algorithm} dla kotwicy {anchor.name.split(':')[0]}")
    plt.xlabel("Błąd [m]")
    plt.ylabel("Wystąpienia")
    plt.grid(axis='y', alpha=0.75)
    plt.xlim(x_lim)

    if save_to_file:
        file_name = f".//{report_dir}//" + f"dist_error_hist_{anchor.name.split(':')[0]}_{type}_{algorithm}.png"
        plt.savefig(file_name)

    plt.show()

def create_dist_error_hist_comparision_for_anchor(loc_res_list, loc_res_list_bias, anchor, bins, type, algorithm, save_to_file=True, report_dir='reporting//dist_error_histogram', x_lim=(0,6)):
    error_list = get_distance_errors(loc_res_list, anchor, type=type)
    error_list_bias = get_distance_errors(loc_res_list_bias, anchor, type=type)
    print(anchor.name)
    color='skyblue'

    print("COMP")
    print(error_list)
    print(error_list_bias)

    min_error = min(min(error_list[algorithm]), min(error_list_bias[algorithm]))
    max_error = max(max(error_list[algorithm]), max(error_list_bias[algorithm]))
    common_bins = np.linspace(min_error, max_error, bins)

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 10))
    #normal
    counts, bins, patches = axes[0].hist(error_list[algorithm], bins=common_bins, color=color, edgecolor='black', alpha=0.75)
    quantiles = np.percentile(error_list[algorithm], [25, 50, 75, 90, 95])
    quantiles_labels = ['25%', '50%', '75%', '90%', '95%']
    print(f"SREDNI BLAD pre: {np.mean(error_list[algorithm])}")

    for quantile, label in zip(quantiles, quantiles_labels):
        axes[0].axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        axes[0].text(quantile, max(counts) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')
    axes[0].set_title(f"Histogram Błędów Pomiarów Dystansu {algorithm} dla kotwicy {anchor.name.split(':')[0]}")
    axes[0].set_xlabel("Błąd [m]")
    axes[0].set_ylabel("Wystąpienia")
    axes[0].grid(axis='y', alpha=0.75)
    axes[0].set_xlim(x_lim)

    counts, bins, patches = axes[1].hist(error_list_bias[algorithm], bins=common_bins, color=color, edgecolor='black', alpha=0.75)
    quantiles = np.percentile(error_list_bias[algorithm], [25, 50, 75, 90, 95])
    quantiles_labels = ['25%', '50%', '75%', '90%', '95%']

    print(f"SREDNI BLAD post: {np.mean(error_list_bias[algorithm])}")
    for quantile, label in zip(quantiles, quantiles_labels):
        axes[1].axvline(quantile, color='black', linestyle='dashed', linewidth=2)
        axes[1].text(quantile, max(counts) * 0.9, f'{label}\n{quantile:.2f}', color='black', ha='right', va='center')
    axes[1].set_title(f"Histogram Błędów Pomiarów Dystansu {algorithm} dla kotwicy {anchor.name.split(':')[0]} po korekcji pomiarów")
    axes[1].set_xlabel("Błąd [m]")
    axes[1].set_ylabel("Wystąpienia")
    axes[1].grid(axis='y', alpha=0.75)
    axes[1].set_xlim(x_lim)

    if save_to_file:
        file_name = f".//{report_dir}//" + f"dist_error_hist_{anchor.name.split(':')[0]}_{type}_{algorithm}_comp.png"
        plt.savefig(file_name)

    plt.show()
    pass


def create_cdf_dist_plot_anchor(loc_res_list, loc_res_list_bias, anchor):
    print(anchor.name)
    error_list = get_distance_errors(loc_res_list, anchor, type='avg')
    error_list_bias = get_distance_errors(loc_res_list_bias, anchor, type='avg')

    error_dict ={
        'IFFT': error_list['IFFT'],
        'PHASE': error_list['PHASE'],
        'RSSI': error_list['RSSI'],
        'IFFT - korekcja': error_list_bias['IFFT'],
        'PHASE - korekcja': error_list_bias['PHASE'],
        'RSSI - korekcja': error_list_bias['RSSI']
    }
    plt.figure(figsize=(10,10))
    colors = {
        'IFFT': 'red',
        'PHASE': 'green',
        'RSSI': 'blue'
    }

    for alg, errors in error_dict.items():
        sorted_data, cdf = compute_cdf(errors)
        color = colors[alg.replace(' - korekcja', '')]
        linestyle = '--' if '- korekcja' in alg else '-'

        plt.plot(sorted_data, cdf, label=alg, linestyle=linestyle, color=color, linewidth=3)

    plt.xlabel('Błąd pomiaru odległości [m]')
    plt.xlim(0, 15)
    #plt.ylabel('Skumulowane Prawdopodobieństwo')
    plt.ylim(0, 1)
    #plt.title('CDF Błędów lokalizacji')
    plt.grid(True)
    plt.legend()

    plt.show()





def __get_rssi_distances_from_anchro(loc_res_list, anchor):
    rssi_values = []

    for loc_res in loc_res_list:
        for anch_meas in loc_res.avg_meas:
            if anch_meas.anchor == anchor.name:
                rssi_values.append(anch_meas.rssi_openspace)

    return rssi_values

def create_rssi_free_space_loss_chart(rssi_distances):
    # Sprawdzenie, czy odległości są większe od zera
    #rssi_distances = np.where(rssi_distances == 0, 1e-6, rssi_distances)
    rssi_distances.sort()
    # Parametry modelu FSPL
    frequency = 2.4e9  # częstotliwość dla BLE (2.4 GHz)
    c = 3e8  # prędkość światła w m/s

    # Obliczanie wartości RSSI na podstawie odległości
    rssi_values = 20 * np.log10(rssi_distances) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
    rssi_values = -rssi_values
    # Obliczanie FSPL (Free Space Path Loss)
    fspl_values = 20 * np.log10(rssi_distances) + 20 * np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
    fspl_values = -fspl_values  # Konwersja na wartości dBm (ujemne)

    # Tworzenie wykresu
    plt.figure(figsize=(10, 8))
    plt.plot(rssi_distances, fspl_values, label='Model FSPL', color='blue')  # Jedna linia dla modelu FSPL
    plt.scatter(rssi_distances, rssi_values, label='Pomiary RSSI', color='red', s=15)
    plt.title('Wpływ odległości na zmierzone RSSI', fontsize=20)
    plt.xlabel('Zmierzona odległość [m]', fontsize=18)
    plt.ylabel('RSSI [dBm]', fontsize=18)
    plt.grid(True)
    plt.legend()
    plt.show()


def log_func(x, a, b):
    return a * np.log10(x) + b
def create_rssi_free_space_loss_real(loc_res_list, anchor):
    rssi_values = []
    real_distances = []

    for loc_res in loc_res_list:
        for anch_meas in loc_res.avg_meas:
            if anch_meas.anchor == anchor.name:
                rssi_values.append(anch_meas.rssi_openspace)

        rd = euclidean_dist_2points((loc_res.x, loc_res.y), (anchor.x_cord, anchor.y_cord))
        real_distances.append(rd)

    sorted_pairs = sorted(zip(real_distances, rssi_values))
    real_distances, rssi_values = zip(*sorted_pairs)

    real_distances = list(real_distances)
    rssi_values = list(rssi_values)

    plt.figure(figsize=(10, 10))

    frequency = 2.4e9
    c= 3e8
    fspl_values= 20 * np.log10(rssi_values) + 20 *  np.log10(frequency) + 20 * np.log10(4 * np.pi / c)
    fspl_values = - fspl_values
    plt.scatter(real_distances, fspl_values, label='Pomiary RSSI', color='red')

    popt, pcov = curve_fit(log_func, real_distances, fspl_values)


    fitted_rssi = log_func(np.array(real_distances), *popt)


    plt.plot(real_distances, fitted_rssi, label='Model FSPL', color='blue')


    plt.xlabel("Zmierzona odległość [m]")
    plt.ylabel('RSSI [dBm]')
    plt.grid(True)
    plt.title("Przybliżony wpływ odległości na zmierzone RSSI")
    plt.legend()
    plt.show()

def load_aoa_errors(path):
    df = pandas.read_excel(path)
    data_list = df.iloc[:,0].tolist()
    return data_list

def compute_cdf(data):
    sorted_data = np.sort(data)
    cdf = np.arange(1, len(sorted_data)+1) / len(sorted_data)
    return sorted_data, cdf

def create_cdf_loc_plot(error_dict, include_aoa=True, aoa_path='AoA_PDDA_100samples.xlsx'):
    plt.figure(figsize=(10, 10))
    #print(error_dict)

    colors = {
        'IFFT': 'red',
        'PHASE': 'green',
        'RSSI': 'blue',
        'AOA': 'purple'
    }

    if include_aoa:
        error_dict['IFFT'] = error_dict['IFFT - korekcja']
        del(error_dict['IFFT - korekcja'])
        error_dict['PHASE'] = error_dict['PHASE - korekcja']
        del(error_dict['PHASE - korekcja'])
        error_dict['RSSI'] = error_dict['RSSI - korekcja']
        del(error_dict['RSSI - korekcja'])

        #wczytanie aoa
        error_dict['AOA'] = load_aoa_errors(aoa_path)



    for alg, errors in error_dict.items():
        sorted_data, cdf = compute_cdf(errors)
        color = colors[alg.replace(' - korekcja', '')]
        linestyle = '--' if '- korekcja' in alg else '-'

        plt.plot(sorted_data, cdf, label=alg, linestyle=linestyle, color=color, linewidth=3)

    plt.xlabel('Localization Error [m]', fontsize=16)
    plt.xlim(0,10)
    plt.ylabel(' Cumulated Probability', fontsize=16)
    plt.ylim(0,1)
    plt.title('Localization Error Distribution', fontsize=16)
    plt.grid(True)
    plt.tick_params(axis='both', which='major', labelsize=16)
    plt.legend(fontsize=16)

    plt.savefig("loc_cdf.png")

    plt.show()


def get_total_avg_error(loc_res_list,isBias=False):
    all_ifft_errors = []
    all_phase_errors = []
    all_RSSI_errors = []

    for loc_res in loc_res_list:
        point = (loc_res.x, loc_res.y)
        for anch in loc_res.meas_dict.keys():
            coord = ()
            for an in anchors:
                if an.name == anch:
                    coord = (an.x_cord, an.y_cord)
            for meas in loc_res.meas_dict[anch]:
                all_ifft_errors.append(abs(euclidean_dist_2points(point, coord) - meas.ifft))
                all_phase_errors.append(abs(euclidean_dist_2points(point, coord) - meas.phase_slope))
                all_RSSI_errors.append(abs(euclidean_dist_2points(point, coord) - meas.rssi_openspace))

    ret_dict = {
        'IFFT': np.mean(all_ifft_errors),
        'PHASE': np.mean(all_phase_errors),
        'RSSI': np.mean(all_RSSI_errors)
    }
    if isBias:
        print("SREDNIE WARTOSCI BŁEDÓW PO KOREKCIE")
    else:
        print("SREDNIE WARTOSCI BLEDOW PRZED KOREKTA")

    print(ret_dict)

    return ret_dict



def print_loc_results(loc_res_list):

    for loc_res in loc_res_list:
        print(f"({loc_res.x}, {loc_res.y})\n AVG: IFFT: ({loc_res.avg_mult['IFFT'].x}, {loc_res.avg_mult['IFFT'].y}) "
              f"PHASE: ({loc_res.avg_mult['PHASE'].x}, {loc_res.avg_mult['PHASE'].y}) "
              f"RSSI: ({loc_res.avg_mult['RSSI'].x}, {loc_res.avg_mult['RSSI'].y}) \n"
              f"MED: IFFT: ({loc_res.med_mult['IFFT'].x}, {loc_res.med_mult['IFFT'].y}) "
              f"PHASE: ({loc_res.avg_mult['PHASE'].x}, {loc_res.avg_mult['PHASE'].y})"
              f"RSSI: ({loc_res.avg_mult['RSSI'].x}, {loc_res.avg_mult['RSSI'].y})")


def save_clear_meas_to_excel(meas_file_name, output_file_name):
    ml = read_measurements(meas_file_name)
    print(ml)

    sorted_list = sort_measurements_for_anchor(ml)
    print(sorted_list)

    with pd.ExcelWriter(output_file_name, engine='openpyxl') as writer:
        for sheet_name, data in sorted_list.items():
            sheet_name = sheet_name.split(':')[0]

            dict_list = [meas.to_dict() for meas in data]
            df = pd.DataFrame(dict_list)
            df.to_excel(writer, sheet_name=sheet_name, index=False)


    pass


anchors = [Anchor(boards[2], 0, 0, 1.9), #FE
           Anchor(boards[1], 10, 0, 1.9), #E4
           Anchor(boards[3], 0, 10, 1.9), #F7
           Anchor(boards[0], 10, 10, 1.9)] #FC


if __name__ == '__main__':
    print("START")

    #analyze_all_files("wyniki_pomiarów", "reporting", multilateration_type='2D', clear_failed_meas=True)
    print("NEW TYPE ANALYZING")
    #analyze_all_files_improved("wyniki_pomiarów", "test_tril", "lrl_b20.pkl", data_batch=20)

    # loc_res_ist_b20 = load_loc_res_from_file("test_tril/lrl_b20.pkl")
    #
    # mul1= loc_res_ist_b20[20].avg_mult['IFFT']
    # print(f'IFFT: ({mul1.x},{mul1.y}): {mul1.distance_from_point}')
    #
    loc_res_list = load_loc_res_from_file("reporting/results.pkl") #lista obiektów klasy location_measurement_results
    # mul2= loc_res_list[20].avg_mult['IFFT']
    # print(f'IFFT: ({mul2.x},{mul2.y}): {mul2.distance_from_point}')
    print("TRILATERACJA")
    analyze_all_files_improved("wyniki_pomiarów", "test_tril", 'trilat_full.pkl', trilaterations=True)
    loc_res_list_trilateration = load_loc_res_from_file("test_tril/trilat_full.pkl")
    print(loc_res_list_trilateration[0].avg_trilat)












    # create_scatter_plot_for_anchor_every_measurement(loc_res_list, anchors[0], anchor_name="Anchor 1", correction=False)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_list, anchors[1], anchor_name="Anchor 2", correction=False)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_list, anchors[2], anchor_name="Anchor 4", correction=False)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_list, anchors[3], anchor_name="Anchor 3", correction=False)


    # create_scatter_plots_for_anchor_all_methods(loc_res_list, anchors[0], anchor_name="Anchor 1")
    # create_scatter_plots_for_anchor_all_methods(loc_res_list, anchors[1], anchor_name="Anchor 2")
    # create_scatter_plots_for_anchor_all_methods(loc_res_list, anchors[2], anchor_name="Anchor 4")
    # create_scatter_plots_for_anchor_all_methods(loc_res_list, anchors[3], anchor_name="Anchor 3")


    # create_single_schema_results(loc_res_list, type='PHASE', color='green')
    # print_loc_results(loc_res_list)
    #create_rssi_free_space_loss_real(loc_res_list, anchors[3])

    # for loc_res in loc_res_list:
    #     print(f'({loc_res.x}, {loc_res.y})')
    #     for anchor in loc_res.meas_dict.keys():
    #         print(f'({anchor}): {len(loc_res.meas_dict[anchor])}')








    # error_dict = get_locations_error(loc_res_list, 'avg')
    # error_dict_med = get_locations_error(loc_res_list, 'med')
    # #print(get_avg_error_distance_meas_per_anchor(loc_res_list,anchors[0]))

    #create_loc_error_histogram_all_algorithms(error_dict)

    #create_heatmap_number_of_measurements(loc_res_list)
    # for alg in ['IFFT', 'PHASE', 'RSSI']:
    #     create_loc_error_histogram(error_dict[alg], bins=30, mult_type='avg', alg_type=alg, report_dir='reporting//error_histogram')
    #     create_loc_error_histogram(error_dict_med[alg], bins=30, mult_type='med', alg_type=alg, report_dir='reporting//error_histogram')



    # for anchor in anchors:
    #     print(anchor.name)


    #     # print(get_avg_error_distance_meas_per_anchor(loc_res_list, anchor, type='med'))
    #     # create_dist_error_histogram_for_anchor(loc_res_list, anchor, bins=30, type='avg', algorithm='IFFT', save_to_file=True)
    #     create_scatter_plots_for_anchor_minmax(loc_res_list, anchor=anchor, mult_type="avg", save_to_file=True,
    #                                             report_dir='reporting//distance')
        # create_scatter_plots_for_anchor_minmax(loc_res_list, anchor=anchor, mult_type="med", save_to_file=True,
        #                                        report_dir='reporting//distance')
        #create_scatter_plots_for_anchor(loc_res_list, anchor, mult_type='avg', save_to_file=True, report_dir='reporting//distance_clear//Triple')
    #
    #


    # create_schema_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting//localisation')
    # create_schema_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir='reporting//localisation')
    # create_heat_map_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir='reporting//heatmap')
    # create_heat_map_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir='reporting//heatmap')
    # # for loc_res in loc_res_list:
    # #     print(f'POINT: ({loc_res.x}, {loc_res.y})')
    # #     print(f'MIN: {loc_res.get_min_measurements_per_anchor("FE:5A:0F:0E:29:6F")}')
    # #     print(f'MAX: {loc_res.get_max_measurements_per_anchor("FE:5A:0F:0E:29:6F")}')
    # #     for avg in loc_res.avg_meas:
    # #         if avg.anchor.split(":")[0] == "FE":
    # #             print(f'AVG: {avg}')
    #
    bias_dict_reg = {'FE': {'IFFT': 0.97, 'PHASE': 6.50, 'RSSI': 6.45, 'BEST': 0.97},
                     'E4': {'IFFT': 1.12, 'PHASE': 3.07, 'RSSI': 1.92, 'BEST': 1.12},
                     'F7': {'IFFT': 0.09, 'PHASE': 2.11, 'RSSI': -1.11, 'BEST': 0.09},
                     'FC': {'IFFT': 1.44, 'PHASE': 4.16, 'RSSI': -0.29, 'BEST': 1.44}}
    #a =1.35
    bias_dict_reg_med = {
        'FE': {'IFFT': 0.81, 'PHASE': 6.53, 'RSSI': 1.07, 'BEST': 0.81},
        'E4': {'IFFT': 1.07, 'PHASE': 3.00, 'RSSI': 1.97, 'BEST': 1.07},
        'F7': {'IFFT': 0.15, 'PHASE': 2.10, 'RSSI': -0.46, 'BEST': 0.15},
        'FC': {'IFFT': 1.48, 'PHASE': 4.20, 'RSSI': -0.82, 'BEST': 1.48}
    }
    #a=


    #analyze_all_files("wyniki_pomiarów", "testowe", multilateration_type='2D', clear_failed_meas=True, bias=bias_dict_reg)
    loc_res_unbiased = load_loc_res_from_file("testowe/results.pkl")

    error_dict_bias = get_locations_error(loc_res_unbiased, 'avg')
    error_dict_med_bias = get_locations_error(loc_res_unbiased, 'med')

    error_dict = get_locations_error(loc_res_list, 'avg')

    error_dict['IFFT - korekcja'] = error_dict_bias['IFFT']
    error_dict['PHASE - korekcja'] = error_dict_bias['PHASE']
    error_dict['RSSI - korekcja'] = error_dict_bias['RSSI']
    create_cdf_loc_plot(error_dict)





    # create_cdf_all_errors(loc_res_list, loc_res_unbiased, anchors[0], anchor_name='Anchor 1')
    # create_cdf_all_errors(loc_res_list, loc_res_unbiased, anchors[1], anchor_name='Anchor 2')
    # create_cdf_all_errors(loc_res_list, loc_res_unbiased, anchors[2], anchor_name='Anchor 4')
    # create_cdf_all_errors(loc_res_list, loc_res_unbiased, anchors[3], anchor_name='Anchor 3')

    # create_scatter_plot_for_anchor_every_measurement(loc_res_unbiased, anchors[0], anchor_name="Anchor 1", correction=True)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_unbiased, anchors[1], anchor_name="Anchor 2", correction=True)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_unbiased, anchors[2], anchor_name="Anchor 4", correction=True)
    # create_scatter_plot_for_anchor_every_measurement(loc_res_unbiased, anchors[3], anchor_name="Anchor 3", correction=True)

    # create_scatter_plots_for_anchor_all_methods(loc_res_unbiased, anchors[0], anchor_name="Anchor 1", correction=True)
    # create_scatter_plots_for_anchor_all_methods(loc_res_unbiased, anchors[1], anchor_name="Anchor 2", correction=True)
    # create_scatter_plots_for_anchor_all_methods(loc_res_unbiased, anchors[2], anchor_name="Anchor 4", correction=True)
    # create_scatter_plots_for_anchor_all_methods(loc_res_unbiased, anchors[3], anchor_name="Anchor 3", correction=True)
    #create_single_schema_results(loc_res_unbiased, type='IFFT', color='red', correction=True)
    #print_loc_results(loc_res_unbiased)

    #create_boxplots_for_loc_res_list(loc_res_unbiased, anchors, True)

    #
    # error_dict_unbiased = get_locations_error(loc_res_unbiased, 'avg')
    # error_dict_unbiased_med = get_locations_error(loc_res_unbiased, 'med')
    #
    # for alg in ['IFFT', 'PHASE', 'RSSI']:
    #     create_loc_error_histogram(error_dict_unbiased[alg], bins=30, mult_type='avg', alg_type=alg, report_dir='reporting//error_histogram_bias')
    #     create_loc_error_histogram(error_dict_unbiased_med[alg], bins=30, mult_type='med', alg_type=alg, report_dir='reporting//error_histogram_bias')


    # get_total_avg_error(loc_res_list, False)
    # get_total_avg_error(loc_res_unbiased, True)


    # for anchor in anchors:
    #     print("HEEEEERREEEEEEE")
    #     create_dist_error_hist_comparision_for_anchor(loc_res_list, loc_res_unbiased,anchor, bins=30, type='avg', algorithm='RSSI',
    #                                             save_to_file=True, report_dir='reporting//dist_error_comparision', x_lim=(0,15))
    #     create_cdf_dist_plot_anchor(loc_res_list, loc_res_unbiased, anchor)
    # #     create_dist_error_histogram_for_anchor(loc_res_unbiased, anchor, bins=30, type='avg', algorithm='RSSI',
    # #                                            save_to_file=True, report_dir='reporting//dist_error_histogram_bias', x_lim=(0,30))
    #     create_scatter_plots_for_anchor_minmax(loc_res_unbiased, anchor=anchor, mult_type="avg", save_to_file=True,
    #                                            report_dir='reporting//distance_bias')
    #     create_scatter_plots_for_anchor_minmax(loc_res_unbiased, anchor=anchor, mult_type="med", save_to_file=True,
    #                                            report_dir='reporting//distance_bias')

    #
    # create_schema_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True, report_dir='reporting//localisation_bias')
    # create_schema_with_results(loc_res_unbiased, mult_type='med', save_to_file=True, report_dir='reporting//localisation_bias')
    # create_heat_map_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True, report_dir='reporting//heatmap_bias')
    # create_heat_map_with_results(loc_res_unbiased, mult_type='med', save_to_file=True, report_dir='reporting//heatmap_bias')

    #ANALIZA LOKALIZACJI
    # for solver in ['LSE']:
    #     print('re')
    #     # analyze_all_files("wyniki_pomiarów", f"reporting//loc//{solver}", multilateration_type="2D", clear_failed_meas=True,
    #     #                    create_boxplots=False, solver=solver)
    #     loc_res_list = load_loc_res_from_file(f"reporting//loc//{solver}//results.pkl")
    #
    #     # analyze_all_files("wyniki_pomiarów", f"reporting//loc//{solver}//bias", multilateration_type='2D', clear_failed_meas=True, create_boxplots=False,
    #     #                    bias=bias_dict_reg, solver=solver)
    #     loc_res_unbiased = load_loc_res_from_file(f"reporting//loc//{solver}//bias//results.pkl")
    #
    #     for anchor in anchors:
    #         create_scatter_plots_for_anchor_minmax(loc_res_list, anchor, mult_type='avg')
    #         # create_dist_error_hist_comparision_for_anchor(loc_res_list, loc_res_unbiased, anchor, bins=30, type='avg',
    #         #                                               algorithm='RSSI',
    #         #                                               save_to_file=True, report_dir='reporting//dist_error_comparision', x_lim=(0,15))
    #         create_scatter_plots_for_anchor_minmax(loc_res_unbiased, anchor, mult_type='avg', regression=False)
    #         #create_cdf_dist_plot_anchor(loc_res_list, loc_res_unbiased, anchor)
    #
    #         # print(np.mean(get_distance_errors(loc_res_unbiased,anchor, 'avg')['RSSI']))
    #
    #
    #
    #
    #     create_schema_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir=f'reporting//loc//{solver}//loc_clear')
    #     create_schema_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir=f'reporting//loc//{solver}//loc_clear')
    #     create_heat_map_with_results(loc_res_list, mult_type='avg', save_to_file=True, report_dir=f'reporting//loc//{solver}//heatmap_clear')
    #     create_heat_map_with_results(loc_res_list, mult_type='med', save_to_file=True, report_dir=f'reporting//loc//{solver}//heatmap_clear')
    #
    #     create_schema_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True,
    #                                report_dir=f'reporting//loc//{solver}//loc_bias')
    #     create_schema_with_results(loc_res_unbiased, mult_type='med', save_to_file=True,
    #                                report_dir=f'reporting//loc//{solver}//loc_bias')
    #     create_heat_map_with_results(loc_res_unbiased, mult_type='avg', save_to_file=True,
    #                                  report_dir=f'reporting//loc//{solver}//heatmap_bias')
    #     create_heat_map_with_results(loc_res_unbiased, mult_type='med', save_to_file=True,
    #                                  report_dir=f'reporting//loc//{solver}//heatmap_bias')
    #
    #     error_dict = get_locations_error(loc_res_list, 'avg')
    #
    #     error_dict_med = get_locations_error(loc_res_list, 'med')
    #     # for alg in error_dict.keys():
    #     #     print(f"AVG {alg}")
    #     #     print(np.mean(error_dict[alg]))
    #     #     print(f"MED {alg}")
    #     #     print(np.mean(error_dict_med[alg]))
    #
    #
    #
    #     # for alg in ['IFFT', 'PHASE', 'RSSI']:
    #     #     create_loc_error_histogram(error_dict[alg], bins=30, mult_type='avg', alg_type=alg,
    #     #                                report_dir=f'reporting//loc//{solver}')
    #     #     create_loc_error_histogram(error_dict_med[alg], bins=30, mult_type='med', alg_type=alg,
    #     #                                report_dir=f'reporting//loc//{solver}')
    #
        # error_dict_bias = get_locations_error(loc_res_unbiased, 'avg')
        # error_dict_med_bias = get_locations_error(loc_res_unbiased, 'med')
        #
        #
        # error_dict['IFFT - korekcja'] = error_dict_bias['IFFT']
        # error_dict['PHASE - korekcja'] = error_dict_bias['PHASE']
        # error_dict['RSSI - korekcja'] = error_dict_bias['RSSI']
        # create_cdf_loc_plot(error_dict)
    #
    #     for alg in error_dict.keys():
    #         print(f"AVG {alg}")
    #         print(round(np.mean(error_dict[alg]),2))
    #         print(f"MED {alg}")
    #         print(round(np.mean(error_dict[alg]),2))
    #
    #     create_loc_error_hist_comparision(error_dict['RSSI'], error_dict_bias['RSSI'], bins=30, mult_type='avg', alg_type='RSSI',
    #                                       report_dir=f'reporting//loc//{solver}', isBias=True)


        # for alg in ['IFFT', 'PHASE', 'RSSI']:
        # #     # create_loc_error_histogram(error_dict_bias[alg], bins=30, mult_type='avg', alg_type=alg,
        # #     #                            report_dir=f'reporting//loc//{solver}', isBias=True)
        # #     # create_loc_error_histogram(error_dict_med_bias[alg], bins=30, mult_type='med', alg_type=alg,
        # #     #                            report_dir=f'reporting//loc//{solver}', isBias=True)
        #     create_loc_error_hist_comparision(error_dict[alg],error_dict_bias[alg] , bins=30, mult_type='avg', alg_type=alg,
        #                                report_dir=f'reporting//loc//{solver}', isBias=True)
        #     create_loc_error_hist_comparision(error_dict_med[alg], error_dict_med_bias[alg], bins=30, mult_type='med',
        # #                                       alg_type=alg,
        # #                                       report_dir=f'reporting//loc//{solver}', isBias=True)