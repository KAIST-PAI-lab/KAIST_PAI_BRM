import os
import pandas as pd

def to_months(birthdate, experimentdate):
    year_bd=int(birthdate[:4])
    month_bd=int(birthdate[5:7])
    day_bd=int(birthdate[-2:])
    year_ed=int(experimentdate[:4])
    month_ed=int(experimentdate[5:7])
    day_ed=int(birthdate[8:10])
    
    months=(year_ed-year_bd)*12+(month_ed-month_bd)+int(day_ed>=day_bd)-1
    return months

def to_mf(genderText):
    if genderText=="남성":
        return "Male"
    if genderText=="여성":
        return "Female"

path=f"experiment_results"
#path=os.path.join("experiment_results", "participant_25081110")
for dir in os.listdir(path):
    files=os.listdir(os.path.join(path, dir))
    for file in files:
        df=pd.read_csv(os.path.join(path, dir, file))
        df['Months']=[to_months(df.iloc[0]['Birthdate'], df.iloc[0]['timestamp']) for _ in range(df.shape[0])]
        df['Gender']=[to_mf(df.iloc[i]['Gender']) for i in range(df.shape[-0])]
        df.to_csv(os.path.join(path,dir, file), index=False)

