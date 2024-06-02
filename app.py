import numpy as np, pandas as pd
import joblib 
import pickle
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go 
from pathlib import Path
import streamlit_authenticator as stauth
from datetime import datetime, date
from sklearn.preprocessing import StandardScaler
from streamlit_option_menu import option_menu 
from scipy.stats import kurtosis, skew
from datetime import datetime, date

#load model stat features
bbca_model = joblib.load('E:\Streamlit\model\model_bbca')
bmri_model = joblib.load('E:\Streamlit\model\model_bmri')
bbri_model = joblib.load('E:\Streamlit\model\model_bbri')

#load model without stat features
bbca_model2 = joblib.load('E:\Streamlit\model\model_bbca2')
bmri_model2 = joblib.load('E:\Streamlit\model\model_bmri2')
bbri_model2 = joblib.load('E:\Streamlit\model\model_bbri2')


# Load the data
data_bbca = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BBCA.xlsx')
data_bmri = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BMRI.xlsx')
data_bbri = pd.read_excel(r'C:\Users\Lenovo\Downloads\archive\daily\BBRI.xlsx')


st.set_page_config(page_title="Stock Dashboard", layout="wide")
#---USER AUTHENTICATION---
names = ["User 1"]
usernames = ["user1"]

#load hashed passwords
file_path = Path(__file__).parent / "hashed_pw.pkl"
with file_path.open("rb") as file:
    hashed_passwords = pickle.load(file)

credential = {
    "usernames":{
        usernames[0]:{
        "name":names[0],
        "password":hashed_passwords[0]
     }
    }
}
authenticator = stauth.Authenticate(credential, "stock_dashboard", "123", cookie_expiry_days=30)

name, authentication_status, username = authenticator.login("main")

if authentication_status == False:
    st.error("Username/Password is incorrect")

if authentication_status == None:
    st.warning("Please enter your username and password")

if authentication_status:

    #Sidebar for Navigation
    authenticator.logout("Logout", "sidebar")
    st.sidebar.title(f"Welcome {name}")
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
        return inp2

    bank_info = {
        'Bank BCA': [bbca_model, bbca_model2, data_bbca],
        'Bank Mandiri': [bmri_model, bmri_model2, data_bmri],
        'Bank BRI': [bbri_model, bbri_model2, data_bbri]
    }

    def bank_prediction1(input_data, bank_name):
        # changing the input data to numpy array
        input_data_as_numpy_array = np.asarray(input_data)
        #print(input_data_as_numpy_array)
        input_data = input_data_as_numpy_array.astype(float)
        std_data = stats_features(input_data)
        prediction = bank_info[bank_name][0].predict(std_data) 
        return prediction[0]
    
    def bank_prediction2(input_data, bank_name):
        input_data_as_numpy_array = np.asarray(input_data)
        input_data = input_data_as_numpy_array.astype(float)
        input_data = input_data.reshape(-1, len(input_data))
        prediction = bank_info[bank_name][1].predict(input_data) 
        return prediction[0]
    
    #def main():
    if (selected == "Stock Price Prediction System"):
            st.title('Stock Price Prediction Web App')
            st.sidebar.header('Apply Filter')
            bank_name = st.sidebar.selectbox(
            'Choose Bank',
            ('Bank BCA', 'Bank Mandiri', 'Bank BRI'),
            index=0 ) # Indeks 0 (Pilihan 1) akan dipilih secara default
            
            def is_number(s):
                try:
                    int(s)
                    return True
                except ValueError:
                    return False

            # Getting input data from the user
            price1 = st.text_input('Price 4 days ago')
            price2 = st.text_input('Price 3 days ago')
            price3 = st.text_input('Price 2 days ago')
            price4 = st.text_input('Price 1 days ago')

            all_inputs_valids = True
            all_inputs = [price1, price2, price3, price4]
            for inp in all_inputs:
                if inp and not is_number(inp):
                    all_inputs_valids = False
                    st.error("All values must be numeric.")
                    break
            # Creating button for Prediction
            if st.button('Stock Prediction Result', disabled=not all_inputs_valids):
                if (None in [price1,price2,price3,price4]) or (0 in (len(price1), len(price2), len(price3), len(price4))):
                    st.error("Please enter some data.")
                elif price1 == price2 == price3 == price4:
                    price_prediction = bank_prediction2([price1, price2, price3, price4], bank_name=bank_name)
                    st.success(round(price_prediction))
                else:
                    price_prediction = bank_prediction1([price1, price2, price3, price4], bank_name=bank_name)
                    st.success(round(price_prediction))
            #if __name__ == '__main__':
            #   main()

    def filter_data(start_date, end_date):
        data_filtered = bank_info[bank_name][2].copy() 
        data_filtered['timestamp'] = data_filtered['timestamp'].dt.date
        #print(start_date>data_filtered['timestamp'].min() , end_date>data_filtered['timestamp'].max())
        #if (start_date>data_filtered['timestamp'].min()) or (end_date>=data_filtered['timestamp'].max()):
        #    print('Out of range')
        #    st.write("No data found for the selected date range")
        #else:
        #    print('In Range')
        data_filtered = data_filtered[(data_filtered['timestamp'] >= start_date) & (data_filtered['timestamp'] <= end_date)]
            
        return data_filtered

    if (selected == "Stock Dashboard"):
        st.sidebar.header('Apply Filter')
        bank_name = st.sidebar.selectbox(
            'Choose Bank',
            ('Bank BCA', 'Bank Mandiri', 'Bank BRI'),
            index=0 )
        
        # Set minimum date to 20 years before today
        #min_date = (st.session_state.get('selected_date') or date.today()) - pd.DateOffset(years=20)
        # Set maximum date to 10 years after today
        #max_date = date.today() + pd.DateOffset(years=10)
        # Define your desired default date (replace with your preferred date)
        default_startdate = datetime(year=2003, month=11, day=10)
        default_enddate = datetime(year=2023, month=1, day=6)
        #selected_date = st.sidebar.date_input("Select Date", min_date=min_date, max_date=max_date)
        min_date = datetime(year=2003, month=11, day=10)
        max_date = datetime(year=2023, month=1, day=6)
        start_date = st.sidebar.date_input('Start Date', default_startdate, min_date, max_date)
        end_date = st.sidebar.date_input('End Date', default_enddate, min_date, max_date)
        if start_date > end_date or start_date == end_date:
            st.error("Start date must be before end date.")
        else:
            st.title("Stock Dashboard")
         
            if start_date and end_date:  # Check if both dates are selected
                fig1 = px.line(filter_data(start_date, end_date), x = 'timestamp', y = 'close', title =f"Stock Closing Price for {bank_name}")

                st.plotly_chart(fig1, use_container_width=True)
            else:
                st.write("Please select both Start and End Dates")
            
            if start_date and end_date:
                data = filter_data(start_date, end_date)
                fig2 = go.Figure(data=[go.Candlestick(
                    x=data['timestamp'],  # X-axis: Dates
                    open=data['open'],  # Opening price
                    high=data['high'],  # Highest price
                    low=data['low'],  # Lowest price
                    close=data['close']  # Closing price
                )])
                fig2.update_layout(
                    title='Candlestick Chart',
                    xaxis_title='Date',
                    yaxis_title='Price'
                )
            # Display the chart
                st.plotly_chart(fig2, use_container_width=True) 

            def bank_filter(bank_name):
                datadf = bank_info[bank_name][2]
                datadf['% Change'] = datadf['close'] / datadf['close'].shift(1) -1 
                datadf.dropna(inplace = True)
                return datadf
        
            pricing_data = st.tabs(["Pricing Data"])
            with pricing_data[0]:
                st.header("Price Movements")
               
                datadf = bank_filter(bank_name=bank_name)
                datadf_2022 = datadf[datadf['timestamp'].dt.year == 2022]
                annual_return = datadf_2022['% Change'].mean()*252*100
                stdev = np.std(datadf_2022['% Change'])*np.sqrt(len(datadf_2022))

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info('Annual Return')
                    st.metric(label="Return", value="{:.2f}%".format(annual_return))
                with col2:
                    st.info("Standard Deviation")
                    st.metric(label="STDV", value="{:.2f}%".format(stdev*100))
                with col3:
                    st.info("Risk Close Return")
                    st.metric(label="Risk", value="{:.2f}%".format(annual_return/(stdev*100)))

                dframe = filter_data(start_date, end_date)
                st.dataframe(dframe, hide_index=True)