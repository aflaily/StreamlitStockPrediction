import numpy as np, pandas as pd
import joblib 
import streamlit as st
import plotly.express as px
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu 
from scipy.stats import kurtosis, skew

bbca_model = joblib.load('E:\Streamlit\model\model_bbca')
bmri_model = joblib.load('E:\Streamlit\model\model_bmri')
bbri_model = joblib.load('E:\Streamlit\model\model_bbri')

# Load the data
data_bbca = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BBCA.xlsx')
data_bmri = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BMRI.xlsx')
data_bbri = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BBRI.xlsx')

#Sidebar for Navigation
with st.sidebar:
    selected = option_menu('Stock Dashboard', options=["Stock Dashboard", "Stock Price Prediction System"])

def stats_features(input_data):
    inp2=list(input_data)
    min=float(np.min(input_data))
    max=float(np.max(input_data))
    diff=(max-min)
    std=float(np.std(input_data))
    mean=float(np.mean(input_data))
    median=float(np.median(input_data))
    kurt=float(kurtosis(input_data))
    sk=float(skew(input_data))
    inp2=np.append(inp2,min)
    inp2=np.append(inp2,max)
    inp2=np.append(inp2,diff)
    inp2=np.append(inp2,std)
    inp2=np.append(inp2,mean)
    inp2=np.append(inp2,median)
    inp2=np.append(inp2,kurt)
    inp2=np.append(inp2,sk)
    inp2 = inp2.reshape(-1, len(inp2))

    #print(inp)
    return inp2

#Creating a function for prediction
def bbca_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_data = input_data_as_numpy_array.astype(float)
    std_data = stats_features(input_data)
    prediction = bbca_model.predict(std_data) 
    return prediction[0]

def bmri_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_data = input_data_as_numpy_array.astype(float)
    std_data = stats_features(input_data)
    prediction = bmri_model.predict(std_data) 
    return prediction[0]

def bbri_prediction(input_data):
    # changing the input data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)
    input_data = input_data_as_numpy_array.astype(float)
    std_data = stats_features(input_data)
    prediction = bbri_model.predict(std_data) 
    return prediction[0]

if (selected == "Stock Price Prediction System"):
    #def main():
    #Giving Title
        st.title('Stock Price Prediction Web App')
        st.sidebar.header('Apply Filter')
        option = st.sidebar.selectbox(
        'Pilih Bank',
        ('Bank BCA', 'Bank Mandiri', 'Bank BRI'),
        index=0 ) # Indeks 0 (Pilihan 1) akan dipilih secara default
 
        # Getting input data from the user
        price1 = st.text_input('Price 4 days ago')
        price2 = st.text_input('Price 3 days ago')
        price3 = st.text_input('Price 2 days ago')
        price4 = st.text_input('Price 1 days ago')
 
        # Code for prediction
        price_prediction = ''
 
        # Creating button for Prediction
        if st.button('Price Prediction Result'):
            if option == 'Bank BCA':
                price_prediction = bbca_prediction([price1, price2, price3, price4])
            elif option == 'Bank Mandiri':
                price_prediction = bmri_prediction([price1, price2, price3, price4])
            elif option == 'Bank BRI':
                price_prediction = bbri_prediction([price1, price2, price3, price4])
 
        st.success(price_prediction)
        #if __name__ == '__main__':
         #   main()

if (selected == "Stock Dashboard"):

    st.sidebar.header('Apply Filter')
    option = st.sidebar.selectbox(
        'Pilih Bank',
        ('Bank BCA', 'Bank Mandiri', 'Bank BRI'),
        index=0 )
    start_date = st.sidebar.date_input('Start Date')
    end_date = st.sidebar.date_input('End Date')

    def create_chart(start_date, end_date):
        if option == 'Bank BCA':
            data_filtered = data_bbca.copy()
        elif option == 'Bank Mandiri':
            data_filtered = data_bmri.copy()
        elif option == 'Bank BRI':
            data_filtered = data_bbri.copy()
            
        data_filtered['timestamp'] = data_filtered['timestamp'].dt.date
        data_filtered = data_filtered[(data_filtered['timestamp'] >= start_date) & (data_filtered['timestamp'] <= end_date)]
        fig = px.line(data_filtered, x = 'timestamp', y = 'close', title = 'BBCA Stock')
        return fig

    st.title("Stock Dashboard")
    # Generate chart based on selection
    if start_date and end_date:  # Check if both dates are selected
        fig = create_chart(start_date, end_date)
        if not fig:
            st.write("No data found for the selected date range")
        else:
            st.plotly_chart(fig)
    else:
        st.write("Please select both Start and End Dates")
    
    pricing_data = st.tabs(["Pricing Data"])

    with pricing_data[0]:
        st.header("Price Movements")
        if option == 'Bank BCA':
            data_filtered = data_bbca.copy()
            data_filtered['timestamp'] = data_filtered['timestamp'].dt.date
            data_filtered = data_filtered[(data_filtered['timestamp'] >= start_date) & (data_filtered['timestamp'] <= end_date)]
            st.write(data_filtered)
        elif option == 'Bank Mandiri':
            st.write(data_bmri)
        if option == 'Bank BRI':
            st.write(data_bbri)

    