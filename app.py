import streamlit as st
from data_processing import load_data, preprocess_data, split_data
from model import train_model, evaluate_model, forecast_prices

def main():
    st.title('Gold Price Forecasting')

  
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
     
        df = load_data(uploaded_file)
        df = preprocess_data(df)
        
       
        #st.write("Data Overview:")
        #st.write(df.head())
        
      
        X_train, X_test, y_train, y_test = split_data(df)
        
       
        model = train_model(X_train, y_train)
        
        
        #metrics = evaluate_model(model, X_test, y_test)
        #st.write("Model Evaluation Metrics:")
        #st.write(metrics)
        
     
        forecast_df = forecast_prices(model, df)
        st.write("60-Day Forecasted Prices:")
        st.write(forecast_df)

if __name__ == "__main__":
    main()
