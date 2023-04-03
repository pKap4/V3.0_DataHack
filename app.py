import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
from sklearn.preprocessing import MinMaxScaler
import datetime
# import pickle
import streamlit as st
import model_building as m
import technical_analysis as t
import correlation_analysis as c
import mpld3
import streamlit.components.v1 as components
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volatility import BollingerBands
from sqlalchemy import create_engine
from sqlalchemy.dialects import registry
# from snowflake.connector import connect
import pandas as pd
# import snowflake.connector
# from snowflake.sqlalchemy import URL
# from sqlalchemy import create_engine
import pickle

@st.cache_resource
def predict_model(df):
    plotdf, future_predicted_values,predicted,actual,dates = m.create_model(df)
    return plotdf, future_predicted_values,predicted,actual,dates


        
with st.sidebar:
    st.markdown("# Stock Analysis & Forecasting")
    user_input = st.selectbox(
    '### Please select the stock for forecasting and technical analysis ',
    ('ADANIENT.NS','TATASTEEL.NS','PAGEIND.NS','EICHERMOT.NS','INFY.NS','YESBANK.NS','ADANIPORTS.NS'))
    # user_input = st.text_input('Enter Stock Name', "ADANIENT.NS")
    st.write("### Choose Date for your anaylsis")
    date_from = st.date_input("From",datetime.date(2000, 1, 1))
    date_to = st.date_input("To",datetime.date(2023, 3, 20))
    st.markdown("### Choose Companies for your Diversified Portfolio anaylsis")
    options = st.multiselect(
        '### Select stocks for diversification analysis',
            ['ADANIENT.NS','TATASTEEL.NS','PAGEIND.NS','EICHERMOT.NS','INFY.NS','YESBANK.NS','ADANIPORTS.NS'],
        ['ADANIENT.NS','TATASTEEL.NS']
    )
    st.write("### Detailed Information")
    actions = st.radio('### Display Additional Information',['Home','Trend Analysis','Technical Analysis','Forecasting','Case-Study','Make Your Own Data!'])
    # st.write('You selected:', options[0])

st.header("Welcome to Financial Analysis of Data !")
st.subheader(" Please Select a Company from the left slider.")



       
# st.write('session state is ',st.session_state)
if actions=="Home":
    st.write("")
    
    
if actions=="Trend Analysis":
    df = yf.download(user_input, start=date_from, end=date_to)
    st.markdown("### Trend Analysis")
    fig= plt.figure(figsize=(20,10))
    t.trend_pie_chart(df)
    st.pyplot(fig)
        
    # st.markdown("### Original vs predicted close price")
    # fig= plt.figure(figsize=(20,10))
    # sns.lineplot(data=plotdf)
    # st.pyplot(fig)
    
    st.markdown("### Daily Percentage Changes")
    fig= plt.figure(figsize=(20,10))
    t.daily_percent_change_plot(df)
    st.pyplot(fig)
    
    st.markdown("### Volatility Plot")
    fig= plt.figure(figsize=(20,10))
    t.volatility_plot(df)
    st.pyplot(fig)
    
    st.markdown("### Adj Close Price")
    fig= plt.figure(figsize=(20,10))
    t.last_2_years_price_plot(df)
    st.pyplot(fig)
    
    st.markdown("### Volume Plot")
    fig= plt.figure(figsize=(20,10))
    t.volume_plot(df)
    st.pyplot(fig)
    
if actions == "Technical Analysis":
#adding a button

        # st.markdown("### Next 10 days forecast")
        # list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]

        # for i,j in zip(st.tabs(list_of_days),range(10)):
        #     with i:
        #         st.write(future_predicted_values.iloc[j:j+1])
        
        
        # st.markdown("### Daily Percentage Changes Histogram")
        # fig= plt.figure(figsize=(20,10))
        # t.daily_percent_change_histogram(df)
        # st.pyplot(fig)
        df = yf.download(user_input, start=date_from, end=date_to)

        st.markdown("## Technical Analysis")

        st.markdown("### MACD Indicator")
        
        fig= plt.figure(figsize=(20,10))
        t.plot_price_and_signals(t.get_macd(df),'MACD')
        st.pyplot(fig)

        fig= plt.figure(figsize=(20,10))
        t.plot_macd(df)
        st.pyplot(fig)

        st.write(" ***:blue[Strategy:]:***")
        st.write(":red[Sell  Signal:] The cross over: When the MACD line is below the signal line.")
        st.write(":green[Buy Signal:] The cross over: When the MACD line is above the signal line.")

        st.markdown("### RSI Indicator")

        fig= plt.figure(figsize=(20,10))
        t.plot_price_and_signals(t.get_rsi(df),'RSI')
        st.pyplot(fig)

        fig= plt.figure(figsize=(20,10))
        t.plot_rsi(df)
        st.pyplot(fig)

        st.write(" ***:blue[Strategy:]:***")
        st.write(":red[Sell  Signal:] When RSI increases above 70%")
        st.write(":green[Buy Signal:] When RSI decreases below 30%.")


        st.markdown("### Bollinger Indicator")

        fig= plt.figure(figsize=(20,10))
        t.plot_price_and_signals(t.get_bollinger_bands(df),'Bollinger_Bands')
        st.pyplot(fig)

        fig= plt.figure(figsize=(20,10))
        t.plot_bollinger_bands(df)
        st.pyplot(fig)

        st.write(" ***:blue[Strategy:]:***")
        st.write(":red[Sell  Signal:] As soon as the market price touches the upper Bollinger band")
        st.write(":green[Buy Signal:] As soon as the market price touches the lower Bollinger band")

        st.markdown("### SMA Indicator")
    
        fig= plt.figure(figsize=(20,10))
        t.sma_plot(df)
        st.pyplot(fig)
        st.write(" ***:blue[Strategy:]:***")
        st.write(":red[Sell  Signal:] When the 50-day SMA crosses below the 200-day SMA.")
        st.write(":green[Buy Signal:] When the 50-day SMA crosses above the 200-day SMA.")

        st.markdown("### EMA Indicator")
    
        fig= plt.figure(figsize=(20,10))
        t.ema_plot(df)
        st.pyplot(fig)
        st.write(" ***:blue[Strategy:]:***")
        st.write(":red[Sell  Signal:] When the 50-day EMA crosses below the 200-day EMA.")
        st.write(":green[Buy Signal:] When the 50-day EMA crosses above the 200-day EMA.")
    
        st.markdown("### Diversified Portfolio Analysis")
        combined_df = yf.download(options, start=date_from, end=date_to)['Adj Close']
        combined_df = combined_df.round(2)
        
        fig= plt.figure(figsize=(20,10))
        c.corr_plot(combined_df)
        st.pyplot(fig)

        st.write(" ***:blue[Strategy:]:*** All the stocks which do not show significant correlation can be included in a portfolio.")
        
@st.cache_data
def show_data(df):
    st.write(df)
    
def print_data(df):
    st.write(df)

    
if actions == "Forecasting":
    df = yf.download(user_input, start=date_from, end=date_to)
    st.write(f"#### The selected company is {user_input}")
    show_data(df)
    
    st.write("#### Run the model to check the forecasts of selected Company")
    btn = st.button('Run Model')

    if btn:
        df = yf.download(user_input, start=date_from, end=date_to)
        plotdf,future_predicted_values,predicted,actual,dates = predict_model(df)

    
    
        # engine = create_engine(URL(
        # account = '<account>',
        # user = '<username>',
        # password = '<password>',
        # database = '<Database name in Snowflake',
        # schema = '<Schema>',
        # warehouse = '<Warehouse_Name>',
        # role='ACCOUNTADMIN',
        # ))
        
        #if type(future_predicted_values) == int
        # # df_new = pd.DataFrame([comp, future_predicted_values[0]], columns = ['name', 'next_closing'])
        # registry.register('snowflake', 'snowflake.sqlalchemy', 'dialect')

        # connection = engine.connect()
        
        # #df_new.to_sql('price_table', con=engine, index=False) #make sure index is False, Snowflake doesnt accept indexes
        
        # connection.close()
        # engine.dispose()
        st.markdown("#### Next 10 days forecast")
        list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]
        
        for i,j in zip(st.tabs(list_of_days),range(10)):
                with i:
                    st.write(future_predicted_values.iloc[j:j+1])

        st.markdown("#### Graph of Next 100 days forecast")
        st.line_chart(future_predicted_values)
        
        st.markdown("#### Graph of previous days to check accuracy")
        
        data = pd.DataFrame(list(zip(list(actual.ravel()), list(predicted.ravel()))),columns=['actual','predicted'])
        st.line_chart(data,height=500)

        
if actions == "Case-Study":
    # comp = st.selectbox('Choose the Company: ', ('ADANIENT.NS','TATASTEEL.NS','PAGEIND.NS','EICHERMOT.NS','INFY.NS','YESBANK.NS','ADANIPORTS.NS'))
    # df = yf.download(comp, start=date_from, end=date_to)

    # edited_df = pd.DataFrame(df)
    # edited_df.reset_index(inplace = True)
    # edited_df = st.experimental_data_editor(df)
    
    # if st.button("Edited"):
    #     st.write(edited_df)
    st.header("Case Study : Yes Bank")
    st.write("Introduction A bank plays an important role in maintaining the economical condition of the country. The Yes Bank crisis created panic among the people in March 2020 as it was the biggest private sector bank. The failure of a bank, be it a public sector bank or a private sector bank, can affect everyone. In March 2020, news came out that there is a high chance of a Yes bank collapse which caused panic among the depositors. The Reserve Bank of India decided to solve this issue. Reserve Bank of India superseded Yes Bank board of directors for a period of 30 days. ")
    st.write("Yes Bank provides banking and financial services. There are around 1050 branches all over India. Yes Bank Ltd. was incorporated on November 21, 2003 by Rana Kapoor and late Ashok Kapoor.In September 2014, Yes Bank announced it had received a ratings upgrade from credit rating agency ICRA and CARE for its various long-term debt programs. On December 18 2017, Yes Bank made its entry in the 30-share S&P BSE Sensex. In 2017, RBI noticed bad loans of Yes Bank. In 2018 RBI ordered Rana Kapoor to vacate the chair of CEO. In November 2018 Chairman and two independent directors resigned from the post. This was the time when Yes bank credit rates decreased. In 2019, November Rana Kapoor’s house sold away all his shares of Yes Bank at a total value of 142 crores.Since the bank was lending money at high risk, the loan increased in such a way that led to the bank crisis")
    a = st.selectbox('What led to the downfall of the Yes Bank?',['A large amount of loan','Loan to firms and companies and NPA’s','A large number of withdrawals','Financial Position of Yes Bank','Huge Liabilities','Improper governance'])
    if a == 'A large amount of loan':
        st.write("On 31st March 2014, Yes Bank book of accounts reflected loan as 55,633 crores, and the deposit book data was 74,192 crores. Since then loan growth increased highly and went to 2.25 trillion as of Sept 30 of 2019. The Asset quality of the bank also worsened. According to global financial company UBS, it has been pointed out that Yes bank is giving stressed loans to such companies which cannot repay the loan in time. It was a high risk taken by Yes Bank. Such loans are called bad loans. ")
    if a == 'Loan to firms and companies and NPA’s':
        st.write("In the 2015 UBS report, a global financial service company made an analysis and found that assets quality of Yes Bank had loans more than its net worth to such companies that are unlikely to repay the loan amount. Also, Yes Bank continued to give loan amounts to several big firms and companies that resulted in the start of the crisis. Such companies were DHFL, CCD, Essel group, Reliance group of industries etc. because such corporate companies cannot repay the loan amount in a limited period. Around 25% of loans were given to the non-banking financial companies, real estate firms, and construction sector. These sectors are the more struggling sectors of India during the past few years. The bad loans of Yes Bank are estimated to be around Rs.40000 crores (Gross NPA). While the Gross NPA was around 19% of advances, Net NPA was around 6% of loans at the end of December 2019. And it is the time when performing assets (NPA’s) started rising in Yes Bank. ")
    if a == 'A large number of withdrawals':
        st.write("While the loan amount was increasing at a high rate, on the other hand, withdrawals were also increasing. The bank showed steady withdrawals due to this burden on the balance sheet and slowly the bank collapsed.Bank run situations have been raised as people panicked seeing news relating to the Yes Bank crisis. Bank run situations arise when a large number of depositors withdraw their money during the same period or within a specific period. A bank run typically results due to panic rather than true insolvency.")
    if a == 'Financial Position of Yes Bank':
        st.write("The share price of the Yes Bank was Rs.400 in 2018 which is now standing at just 16.60 as of 6 March 2020. The financial condition deteriorated due to its inability to raise capital to address potential loan losses.  ")
    if a == 'Huge Liabilities':
        st.write("The Yes Bank has a total liability of 24 thousand crore dollars. The bank has a balance sheet of about $40 billion (2.85 lakh crore rupees). It had to pay $ 2 billion to increase the capital base. (As per 2019 data) These were some reasons behind the pitiable condition of the yes bank. ")
    if a == 'Improper governance':
        st.write("Yes bank is a private sector bank. It has faced several governance issues that led to this crisis. In January 2020, Uttam Prakash Agarwal independent director quit Yes Bank, citing governance of Yes Bank due to its degradation.")
      
    dfyb = yf.download('YESBANK.NS', start=date_from, end=date_to)
    dfnf = yf.download('ADANIENT.NS', start=date_from, end=date_to)
    
    st.markdown("## Predicted value Graph")
    fig= plt.figure(figsize=(20,10))
    plt.plot(dfyb['Close'],label="YESBANK")
    plt.plot(dfnf['Close'],label="NIFTYFIFTY")
    plt.legend()
    st.pyplot(fig)
        
        
if actions == "Make Your Own Data!":
    st.write(f"#### The selected company is {user_input}")
    df = yf.download(user_input, start=date_from, end=date_to)

    edited_df = pd.DataFrame(df)
    edited_df.reset_index(inplace = True)

    if 'exp_data_frame' not in st.session_state:
        st.session_state.exp_data_frame = st.experimental_data_editor(edited_df)
        output_df = st.session_state.exp_data_frame

    else:
        output_df = st.experimental_data_editor(st.session_state.exp_data_frame)
    
    if st.button("Show Data And Prediction On Edited Data"):
        st.write(output_df)
        
        plotdf,future_predicted_values,predicted,actual,dates = predict_model(edited_df)
        st.markdown("### Next 10 days forecast")
        list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]
        
        for i,j in zip(st.tabs(list_of_days),range(10)):
                with i:
                    st.write(future_predicted_values.iloc[j:j+1])
        
        st.markdown("## Predicted value Graph")
        #fig = plt.figure(figsize=(20,10))
        #plt.plot(list_of_days,future_predicted_values)
        #st.pyplot(fig)
        st.line_chart(future_predicted_values)
        # edited_df.set_index('Date')
        # plotdf,future_predicted_values,predicted,actual,dates = predict_model(edited_df)
        # st.markdown("### Next 10 days forecast")
        # list_of_days = ["Day 1", "Day 2", "Day 3", "Day 4", "Day 5", "Day 6", "Day 7","Day 8", "Day 9", "Day 10"]
    
        # for i,j in zip(st.tabs(list_of_days),range(10)):
        #     with i:
        #         st.write(future_predicted_values.iloc[j:j+1])
                
        # st.markdown("## Predicted value Graph")
        # fig= plt.figure(figsize=(20,10))
        # plt.plot(list_of_days,future_predicted_values)
        # st.pyplot(fig)
        # st.write(edited_df)
        
    # st.markdown("## Comparison value Graph")
    # fig= plt.figure(figsize=(20,10))
    # plt.plot(dates,actual,label = "actual")
    # plt.plot(dates,predicted,label = "predicted")
    # # plt.plot(list_of_days,future_predicted_values)
    # st.pyplot(fig)