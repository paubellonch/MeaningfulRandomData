import numpy as np
import pandas as pd

 #Esta prepadado para los datos Churn, aqui se hace el prepocesado que lo hace el cliente
def read_csv(csv_filename):
   
    data = pd.read_csv(csv_filename)
    data.drop("RowNumber", axis=1, inplace=True)
    data.drop("CustomerId", axis=1, inplace=True)
    data.drop("Surname", axis=1, inplace=True)
    return data
