import numpy as np
from mlib.Cat import *
from config import *
import pandas as pd
from pd.CSV import CSV

def keypointUpdate(idx_t, pc_t, pc_added, csv):
    '''
    idx_t : previous tracked index (initial keypoint indexes used for co-tracker)
            (indexes before modifiying pw.xlsx)
    pc_t : co-trcker output keypoint predictions 

    idx_curnt : All modified indexes for current frame want to be tracked for future
                (indexes after modifiying pw.xlsx)
    pc_added : Newly added/ modified keypoints
    idx_added : indexes of newly added/ modified keypoints

    '''
    idx_curnt = csv.idx
    idx_added = csv.nidx
    Pw = csv.Pw
    idx_common = np.intersect1d(idx_t,idx_curnt) # common idx current and previos
    pc_curr_comm = pc_t[np.in1d(idx_t,idx_curnt)]
    Pw = Pw[np.in1d(idx_curnt,idx_t)]
    pc_all_t = np.concatenate((np.expand_dims(idx_common, axis=1),pc_curr_comm), axis=1) # all keypoints which is tracked
    try:
        pc_all_added = np.concatenate((np.expand_dims(idx_added, axis=1),pc_added), axis=1)

        pc_all_added_sort = pc_all_added[pc_all_added[:, 0].argsort()]
        modified_tracked = np.in1d(pc_all_added_sort[:, 0], idx_common)
        pc_all_new = pc_all_added_sort[np.invert(modified_tracked)]
        
        modified_tracked_comm = np.in1d(idx_common, pc_all_added_sort[:, 0])
        '''copy modified common keypoints'''
        pc_all_t[modified_tracked_comm]=pc_all_added_sort[modified_tracked]

        new_pc = np.concatenate((pc_all_t,pc_all_new), axis=0).astype(np.int32)
        new_pc_organized  = new_pc[new_pc[:, 0].argsort()] # idx sorting
    except:
        new_pc_organized = pc_all_t.astype(np.int32)
    return new_pc_organized, Pw

def observeKeysDict(pc_prev, pc_newly_added):
    df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')
    csv = CSV(df)

    try:
        pc_updated, Pw = keypointUpdate(pc_prev[:,0], pc_prev[:,1:3], pc_newly_added, csv)
    except:
        try:
            pc_updated = np.concatenate((np.expand_dims(csv.idx, axis=1),pc_newly_added), axis=1)
            Pw = csv.Pw
        except:
            print("Selected keypoints are not matched with assignment")
    return Pw, pc_updated.astype(np.int32)