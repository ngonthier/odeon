# Create a csv file for the HiUCD dataset to be able to read it 

import os
import pandas as pd

# Specify the folder path
folder_path = r'\\store\store-DAI\datasrc\dchan\hiucd_mini' #\train\image\2017\9"
if not os.path.exists(folder_path):
    folder_path = r"C:\Users\NGonthier\Documents\Detection_changement\data\hiucd_mini"

years_per_split = {"train": ['2017', '2018'], "val":['2017', '2018'] , "test":['2018', '2019']}

for split in ['train','val','test']: 

    output_name = split + '.csv'

    df = pd.DataFrame(columns=["id", "id_zone","millesime1","millesime2","change_pat", "T0_path", "T1_path", "change_pat", "T0", "T1", "change"])

    year_0, year_1 = years_per_split[split]
    split_folder_T0 = os.path.join(folder_path,split,"image",year_0,"9")
    split_folder_T1 = os.path.join(folder_path,split,"image",year_1,"9")
    split_folder_change = os.path.join(folder_path,split,"mask_merge",year_0+'_'+year_1,"9")

    files = os.listdir(split_folder_T0)

    for file in files:
        #print(file)
        file_parts = file.split(".")
        
        id_value = file_parts[0] 
        id_zone = id_value.split('_')[0] 
        T0_path_value = os.path.join(split,"image",year_0,"9", file)
        T1_path_value = os.path.join(split,"image",year_1,"9", file)
        change_value = os.path.join(split,"change_mask",year_0+'_'+year_1,"9", file)
        
        # Append a new row to the DataFrame
        df.loc[len(df)] = {"id": id_value,"id_zone": id_zone, "millesime1": year_0, "millesime2": year_1, "T0_path": T0_path_value, "T1_path": T1_path_value, "change_pat": change_value,
                           "T0": T0_path_value, "T1": T1_path_value, "change": change_value}
        # Faut il que T0 T1 et change soit égale aux valeurs path ? 

    # Display the DataFrame
    print(df.head(5))

    path_or_buf = os.path.join(folder_path, output_name)
    print('path_or_buf',path_or_buf)
    df.to_csv(path_or_buf=path_or_buf, sep=',', index=False)