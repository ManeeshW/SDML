import numpy as np

class CSV:
    def __init__(self, df):
        
        self.idxPw = self.get_4x1(df, 'Pw')
        
        self.keep = self.get(df, 'Keep')
        nKeysU = self.get_newK(df)[np.where(self.get_newK(df)[:, 1]>0)]
        p5d = self.get_5x1(df,'Pw')[np.where(self.get_5x1(df,'Pw')[:, 4]>0)]
        self.Pw = p5d[:,1:4]
        self.idx = p5d[:,0]
        self.nidx= nKeysU[nKeysU[:, 1].argsort()][:, 0]
        self.requried_keys = self.get(df, 'Total')
        self.modified_keys = self.get(df, 'Modified')

    def get_5x1(self, df, label):
        return np.vstack(
            (
                df['{}'.format("idx")].values,
                df['{}_x'.format(label)].values, 
                df['{}_y'.format(label)].values, 
                df['{}_z'.format(label)].values, 
                df['{}'.format("Keep")].values
            )
        ).T

    def get_4x1(self, df, label):
        return np.vstack(
            (
                df['{}'.format("idx")].values,
                df['{}_x'.format(label)].values, 
                df['{}_y'.format(label)].values, 
                df['{}_z'.format(label)].values
            )
        ).T
    
    def get_3x1(self, df, label):
        return np.vstack(
            (
                df['{}_x'.format(label)].values, 
                df['{}_y'.format(label)].values, 
                df['{}_z'.format(label)].values
            )
        ).T
    
    def get_newK(self, df):
        return np.vstack(
            (
                df['{}'.format("idx")].values,
                df['{}'.format("New_Keys_Order*")].values, 
            )
        ).T

    def get(self, df, label):
        try:
            return df[label].values
        except:
            return 0