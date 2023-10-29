import numpy as np
from . import projectedKnown3dBoxPoints, cat_points

def is_points_image_bound(Pc):
    if (np.any(Pc[:,0]<=0) or np.any(Pc[:,0]>=640) or np.any(Pc[:,1]<=0) or np.any(Pc[:,1]>=480)):
        return False
    else:
        return True

def is_object_size_good(Pc,Thr_x, Thr_y):
    if np.max(Pc[:,0])-np.min(Pc[:,0]) < Thr_x or np.max(Pc[:,1])-np.min(Pc[:,1]) < Thr_y :
        return False
    else:
        return True

def is_object_visible(Pc, Thr_x, Thr_y):
    if is_points_image_bound(Pc):
        if is_object_size_good(Pc,Thr_x, Thr_y):
            return True
        else:
            return False
    else:
        return False

def categorize_objects(n, Hc, K, Thr_x, Thr_y,points_only):
    cat_ids = []
    Pcs = []
    for cat_id in range(1,n+1):
        P = cat_points(cat_id=cat_id)
        Pc, _, _, _, _= projectedKnown3dBoxPoints(P, Hc,K,points_only=points_only)
        if is_object_visible(Pc, Thr_x, Thr_y):
            cat_ids.append(cat_id)
            Pcs.append(Pc)

    return cat_ids, Pcs
