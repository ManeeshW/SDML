import numpy as np
from numpy import linalg as L2
import pandas as pd

from pd.CSV import CSV


# df = pd.read_csv("pw_initial.csv", header=0, skipinitialspace=True)
# cols = ['idx', 'Pw_x', 'Pw_y', 'Pw_z', 'keep']
# df.to_csv("pw.csv")

# csv = CSV(df)
# C = np.array([csv])
# np.savez("CSV.npz", C)
# print(csv.idx)
# print(csv.Pw)
# print(csv.keep)
# print(np.array([csv.Pw[i] for i in range(10) if csv.keep[i]]))

# # function
# def dfs_tabs(df_list, sheet_list, file_name):
#     writer = pd.ExcelWriter(file_name,engine='xlsxwriter')   
#     for df, sheet in zip(df_list, sheet_list):
#         df.to_excel(writer, sheet_name=sheet, startrow=0 , startcol=0)   
#     writer.close()

# # list of dataframes and sheet names
# dfs = [df for i in range(10)]
# sheets = ["{}".format(i+1) for i in range(10)]    

# # run function
# dfs_tabs(dfs, sheets, 'multi.xlsx')

df = pd.read_excel('pw.xlsx',header=0, sheet_name='All')

csv = CSV(df)
print(csv.idx)
print(csv.Pw)
# print(csv.keep)
#print(np.array([csv.Pw[i] for i in range(csv.Pw.shape[0]) if csv.keep[i]]))

print(csv.Nidx)
#print(np.array([csv.NewKeys[0] for i in range(csv.Pw.shape[0]) if csv.NewKeys[:,i] > 0]))
#df.to_csv("pw_u.xlsx")