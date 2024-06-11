import math
import re



class measurement:
    def __init__(self, anchor, quality, high_precision, ifft, phase_slope, rssi_openspace, best):
        self.anchor = anchor
        self.quality = quality
        self.high_precision = high_precision
        self.ifft = ifft
        self.phase_slope = phase_slope
        self.rssi_openspace = rssi_openspace
        self.best = best

    def __str__(self):
        return f"{self.anchor}: [HP: {self.high_precision}, IFFT: {self.ifft}, PS: {self.phase_slope}, RSSI: {self.rssi_openspace}, B: {self.best} "


class Anchor:
    def __init__(self, name, x, y, z):
        self.name = name
        self.x_cord = x
        self.y_cord = y
        self.z_cord = z

    def __str__(self):
        return f"{self.name}:  X: {self.x_cord}, Y: {self.y_cord}, Z: {self.z_cord}"

class location_measurements_results:

    def __init__(self, x, y, meas_dict, avg_meas, med_meas, avg_mult, med_mult):
        self.x = x
        self.y = y
        self.meas_dict = meas_dict #wszystkie pomiary danego punktu w slowniku kotwic
        self.avg_meas = avg_meas #lista obiektów measurement
        self.med_meas = med_meas
        self.avg_mult = avg_mult
        self.med_mult = med_mult
        pass

    def __str__(self):
        return f"({self.x},{self.y},{self.meas_dict},{self.avg_mult},{self.med_mult}.)"

    def to_dict(self):
        dict = {
            'x': self.x,
            'y': self.y,
            'meas_dict': self.meas_dict,
            'avg_meas': self.avg_meas,
            'med_meas': self.med_meas,
            'avg_mult': self.avg_mult,
            'med_mult': self.med_mult
        }
        return dict

    def get_min_measurements_per_anchor(self, anchor_name):
        # min_ifft = self.meas_dict[anchor_name][0].ifft
        # min_phase_slope = self.meas_dict[anchor_name][0].phase_slope
        # min_rssi_openspace = self.meas_dict[anchor_name][0].rssi_openspace
        # min_best = self.meas_dict[anchor_name][0].best
        # #cos tu jest nie tak
        # for meas in self.meas_dict[anchor_name]:
        #     if meas.ifft < min_ifft:
        #         min_ifft = meas.ifft
        #     if meas.phase_slope < min_phase_slope:
        #         min_phase_slope = meas.phase_slope
        #     if meas.rssi_openspace < min_rssi_openspace:
        #         min_rssi_openspace = meas.rssi_openspace
        #     if meas.best < min_best:
        #         min_best = meas.best
        ifft_values =[]
        phase_values = []
        rssi_values = []
        best_values = []

        for meas in self.meas_dict[anchor_name]:
            ifft_values.append(meas.ifft)
            phase_values.append(meas.phase_slope)
            rssi_values.append(meas.rssi_openspace)
            best_values.append(meas.best)

        min_ifft = min(ifft_values)
        min_phase_slope = min(phase_values)
        min_rssi_openspace = min(rssi_values)
        min_best = min(best_values)


        return {"IFFT": min_ifft, "PHASE": min_phase_slope, "RSSI": min_rssi_openspace, "BEST": min_best}

    def get_max_measurements_per_anchor(self, anchor_name):
        max_ifft = self.meas_dict[anchor_name][0].ifft
        max_phase_slope = self.meas_dict[anchor_name][0].phase_slope
        max_rssi_openspace = self.meas_dict[anchor_name][0].rssi_openspace
        max_best = self.meas_dict[anchor_name][0].best

        for meas in self.meas_dict[anchor_name]:
            if meas.ifft > max_ifft:
                max_ifft = meas.ifft
            if meas.phase_slope > max_phase_slope:
                max_phase_slope = meas.phase_slope
            if meas.rssi_openspace > max_rssi_openspace:
                max_rssi_openspace = meas.rssi_openspace
            if meas.best > max_best:
                max_best = meas.best

        return {"IFFT": max_ifft, "PHASE": max_phase_slope, "RSSI": max_rssi_openspace, "BEST": max_best}


class mult_result:
    def __init__(self, type, x, y, z, distance_from_point):
        self.type = type
        self.x = x
        self.y = y
        self.z = z
        self.distance_from_point = distance_from_point



def euclidean_distance(anchor, meas_x, meas_y, include_height=False, meas_height=1.6):
    ax = anchor.x_cord
    ay = anchor.y_cord

    if include_height:
        az = anchor.z_cord ##jesli bierzemy wysokosc
        return math.dist([ax, ay, az], [meas_x, meas_y, meas_height])
    else:
        return math.dist([ax, ay], [meas_x, meas_y])

def euclidean_dist_2points(point1, point2, height=False):
    #print(point1.x)
    if height:
        return math.dist([point1[0], point1[1], point1[2]], [point2[0], point2[1], point2[2]])
    else:
        return math.dist([point1[0], point1[1]], [point2[0], point2[1]])


def extract_location_cord_from_file(file_name):
    match = re.match(r'e(\d+)_(\d+)\.txt$', file_name)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    return None, None