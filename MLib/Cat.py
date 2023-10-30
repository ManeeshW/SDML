import numpy as np

def cat(pc, p=[0,0]):
    try:
        p.shape[1]
    except:
        p = np.expand_dims(p, axis=0)

    try:
        if pc.shape[0] == 0:
            if p.shape[0]==0:
                pc = np.append(pc,p, axis=0).reshape(-1,2)
            else:
                pc = np.append(pc,p[0], axis=0).reshape(-1,2)
        else:
            pc = np.concatenate((pc, p), axis=0)
    except:
        pc = p
        pass

    return pc.astype(np.int32)