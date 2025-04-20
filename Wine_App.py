# import streamlit as st
# import numpy as np
# import pandas as pd
# import joblib

# model=joblib.load('knn_model.pkl')
# scaler=joblib.load('scaler_model.pkl')

# st.title("Wine Quality Prediction App")
# st.write("Predict Quality of Wine Using KNN")

# # st.sidebar.header("Wine Data")

# # def user_input_features():
# Alcohol=st.sidebar.slider("Alcohol",min_value=11.03,max_value=14.83,value=13.0,step=0.01)
# Malic_sacid = st.sidebar.slider("Malic Acid", min_value=0.74, max_value=5.8, step=0.01)
# Ash = st.sidebar.slider("Ash", min_value=1.36, max_value=3.23,step=0.01)
# Alcalinity_of_ash = st.sidebar.slider("Alcalinity of ash", min_value=10.6, max_value=30.0,value=19.0, step=0.01)
# Magnesium = st.sidebar.slider("Magnesium", min_value=70, max_value=162, value=100,step=1)
# Total_phenols = st.sidebar.slider("Total phenols", min_value=0.98, max_value=3.88,  value=2.5,step=0.01)
# Flavanoids = st.sidebar.slider("Flavanoids", min_value=0.34, max_value=5.08, value=2.5,step=0.01)
# Nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid phenols", min_value=0.13, max_value=0.66,step=0.01 )
# Proanthocyanins = st.sidebar.slider("Proanthocyanins", min_value=0.41, max_value=3.58, step=0.01)
# Color_intensity = st.sidebar.slider("Color intensity", min_value=1.28, max_value=13.0, value=5.0,step=0.01)
# Hue = st.sidebar.slider("Hue", min_value=0.48, max_value=1.71,value=1.0 ,step=0.01)
# diluted_wines = st.sidebar.slider("OD280/OD315 of diluted wines", min_value=1.27, max_value=4.0,value=2.0 ,step=0.01)
# Proline = st.sidebar.slider("Proline", min_value=278, max_value=1680, step=5)

# wine_classes = {
#     1: "🍇 Class 1: Wines made from Cultivar 1 (Grape A)",
#     2: "🍷 Class 2: Wines made from Cultivar 2 (Grape B)",
#     3: "🥂 Class 3: Wines made from Cultivar 3 (Grape C)"
# }


# # Predict button
# if st.button('Predict Wine Class'):
#     input_data = np.array([[Alcohol,Malic_sacid, Ash,Alcalinity_of_ash, Magnesium, Total_phenols,
#                             Flavanoids, Nonflavanoid_phenols, Proanthocyanins,Color_intensity,
#                             Hue, diluted_wines,Proline]])
    
#     scaled_data = scaler.transform(input_data)
#     prediction = model.predict(scaled_data)[0]

#     st.success(f"Prediction: {wine_classes[int(prediction)]}")


import streamlit as st
import numpy as np
import pandas as pd
import joblib

model = joblib.load('knn_model.pkl')
scaler = joblib.load('scaler_model.pkl')

st.title("Wine Quality Prediction App")
st.write("Predict Quality of Wine Using KNN")

# Sidebar sliders with values from your provided data
Alcohol = st.sidebar.slider("Alcohol", min_value=11.03, max_value=14.83, value=12.86, step=0.01)
Malic_sacid = st.sidebar.slider("Malic Acid", min_value=0.74, max_value=5.8, value=1.35, step=0.01)
Ash = st.sidebar.slider("Ash", min_value=1.36, max_value=3.23, value=2.32, step=0.01)
Alcalinity_of_ash = st.sidebar.slider("Alcalinity of ash", min_value=10.6, max_value=30.0, value=18.0, step=0.01)
Magnesium = st.sidebar.slider("Magnesium", min_value=70, max_value=162, value=122, step=1)
Total_phenols = st.sidebar.slider("Total phenols", min_value=0.98, max_value=3.88, value=1.51, step=0.01)
Flavanoids = st.sidebar.slider("Flavanoids", min_value=0.34, max_value=5.08, value=1.25, step=0.01)
Nonflavanoid_phenols = st.sidebar.slider("Nonflavanoid phenols", min_value=0.13, max_value=0.66, value=0.21, step=0.01)
Proanthocyanins = st.sidebar.slider("Proanthocyanins", min_value=0.41, max_value=3.58, value=0.94, step=0.01)
Color_intensity = st.sidebar.slider("Color intensity", min_value=1.28, max_value=13.0, value=4.10, step=0.01)
Hue = st.sidebar.slider("Hue", min_value=0.48, max_value=1.71, value=0.76, step=0.01)
diluted_wines = st.sidebar.slider("OD280/OD315 of diluted wines", min_value=1.27, max_value=4.0, value=1.29, step=0.01)
Proline = st.sidebar.slider("Proline", min_value=63, max_value=1680, value=63, step=5)

wine_classes = {
    1: "🍇 Class 1: Wines made from Cultivar 1 (Grape A)",
    2: "🍷 Class 2: Wines made from Cultivar 2 (Grape B)",
    3: "🥂 Class 3: Wines made from Cultivar 3 (Grape C)"
}

# Predict button
if st.button('Predict Wine Class'):
    input_data = np.array([[Alcohol, Malic_sacid, Ash, Alcalinity_of_ash, Magnesium, Total_phenols,
                            Flavanoids, Nonflavanoid_phenols, Proanthocyanins, Color_intensity,
                            Hue, diluted_wines, Proline]])
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]

    if int(prediction) in wine_classes:
        st.success(f"Prediction: {wine_classes[int(prediction)]}")
    else:
        st.error(f"Prediction Error: Unknown class {prediction}")

