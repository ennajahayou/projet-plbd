import streamlit as st
import pandas as pd 
import datetime
def app():
 st.title('projet learning by doing group 15')
 today = datetime.date.today()
 df=pd.read_csv('base_od_donne.csv')
 st.write(df)
 start_date = st.date_input('Start date', today)
 st.write('la valeur minimal de ET est',(min(df['ET'][df['Year']==str(start_date)])))
 st.write('l heure ou on va irriger est',df['hour1'][df['ET']==(min(df['ET'][df['Year']==str(start_date)]))].to_list()[0])