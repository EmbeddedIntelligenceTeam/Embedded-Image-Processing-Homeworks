import numpy as np
from collections import defaultdict

def calculate_moments(rect_image):
    moments =  defaultdict(np.longfloat)
    rect_image = rect_image > 0 
    for x in range(rect_image.shape[0]):
        for y in range(rect_image.shape[1]):
            for i in range(4):
                for j in range(4 - i):
                    order = "m" + str(i) + str(j)
                    moments[order] += (x**i) * (y**j) * rect_image[y,x]

    centroid_x = moments["m10"] / moments["m00"]
    centroid_y = moments["m01"] / moments["m00"]
    moments["mu11"] = max(moments["m11"] - centroid_x * moments["m01"],0)
    moments["mu20"] = max(moments["m20"] - centroid_x * moments["m10"],0)
    moments["mu02"] = max(moments["m02"] - centroid_y * moments["m01"],0)
    moments["mu30"] = max(moments["m30"] - 3 * centroid_x * moments["m20"] + 2 * (centroid_x ** 2 ) * moments["m10"],0)
    moments["mu21"] = max(moments["m21"] - 2 * centroid_x * moments["m11"] - centroid_y * moments["m20"] + 2 * (centroid_x ** 2) * moments["m01"],0)
    moments["mu12"] = max(moments["m12"] - 2 * centroid_y * moments["m11"] - centroid_x * moments["m02"] + 2 * (centroid_y ** 2) * moments["m10"],0)
    moments["mu03"] = max(moments["m03"] - 3 * centroid_y * moments["m02"] + 2 * (centroid_y ** 2 ) * moments["m01"],0)
    moments["nu20"] = moments["mu20"] / (moments["m00"] ** 2) 
    moments["nu11"] = moments["mu11"] / (moments["m00"] ** 2)
    moments["nu02"] = moments["mu02"] / (moments["m00"] ** 2)
    moments["nu30"] = moments["mu30"] / (moments["m00"] ** 2.5)
    moments["nu21"] = moments["mu21"] / (moments["m00"] ** 2.5)
    moments["nu12"] = moments["mu12"] / (moments["m00"] ** 2.5)
    moments["nu03"] = moments["mu03"] / (moments["m00"] ** 2.5)
    print("MOMENTS")
    print(moments)


def calculate_hu_moments(moments):
    hu_moments = [0] * 7
    hu_moments[0] = moments["nu20"]  +moments["nu02"]
    hu_moments[1] = (moments["nu20"] - moments["nu02"]) ** 2 + 4 * moments["nu11"] ** 2
    hu_moments[2] = (moments["nu30"] -3 * moments["nu12"]) ** 2+(3 * moments["nu21"] -moments["nu03"]) ** 2
    hu_moments[3] = (moments["nu30"] + moments["nu12"]) ** 2 + (moments["nu21"] + moments["nu03"]) ** 2
    hu_moments[4] = (moments["nu30"] - 3 * moments["nu12"]) *\
                    (moments["nu30"] + moments["nu12"]) *\
                    ((moments["nu30"] + moments["nu12"]) ** 2-3 * (moments["nu21"] + moments["nu03"]) ** 2) +\
                    (3 * moments["nu21"] - moments["nu03"]) *\
                    (moments["nu21"]+moments["nu03"]) *\
                    (3 * (moments["nu30"] + moments["nu12"]) ** 2 - (moments["nu21"] + moments["nu03"]) ** 2)
    hu_moments[5] = (moments["nu20"]-moments["nu02"]) *\
                    ((moments["nu30"]+moments["nu12"]) ** 2 - (moments["nu21"] + moments["nu03"]) ** 2) +\
                    4 * moments["nu11"] * (moments["nu30"] + moments["nu12"]) * (moments["nu21"]+moments["nu03"])
    hu_moments[6] = (3 * moments["nu21"] - moments["nu03"]) *\
                    (moments["nu30"] + moments["nu12"]) *\
                    ((moments["nu30"]+moments["nu12"]) ** 2 - 3 * (moments["nu21"]+moments["nu03"]) ** 2) -\
                    (moments["nu30"]-3 * moments["nu12"]) * (moments["nu21"]+moments["nu03"]) * (3 * (moments["nu30"]+moments["nu12"]) ** 2-(moments["nu21"]+moments["nu03"]) ** 2)
    
    return hu_moments