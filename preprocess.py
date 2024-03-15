import pandas as pd

# Load the data
data_bbca = pd.read_excel("C:\Users\Lenovo\Downloads\archive\daily\BBCA.xlsx")
data_bmri = pd.read_excel("C:\Users\Lenovo\Downloads\archive\daily\BMRI.xlsx")
data_bbri = pd.read_excel("C:\Users\Lenovo\Downloads\archive\daily\BBRI.xlsx")

data_bbca = data_bbca[data_bbca['timestamp'].dt.year.isin([2013, 2014, 2015, 2016, 2017,2018,2019,2020,2021,2022])]
data_bmri = data_bmri[data_bmri['timestamp'].dt.year.isin([2013, 2014, 2015, 2016, 2017,2018,2019,2020,2021,2022])]
data_bbri = data_bbri[data_bbri['timestamp'].dt.year.isin([2013, 2014, 2015, 2016, 2017,2018,2019,2020,2021,2022])]

# Save the processed data
data_bbca.to_pickle("processed_data_bbca.pkl")
data_bbca.to_pickle("processed_data_bbca.pkl")