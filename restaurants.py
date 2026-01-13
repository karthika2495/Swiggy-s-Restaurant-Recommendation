import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64

encoded_data = pd.read_csv("encoded_data.csv", index_col=0)
cleaned_data = pd.read_csv("cleaned_data.csv", index_col=0)

scaler = StandardScaler()
scaled_data = scaler.fit_transform(encoded_data)

kmeans = KMeans(n_clusters=6, random_state=42, n_init=10)
cleaned_data["cluster"] = kmeans.fit_predict(scaled_data)

st.title("🍽️ SWIGGY RESTAURANT RECOMMENDATION")

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-image: url("https://dummyimage.com/1920x1080/FC8019/ffffff&text=Swiggy+Food");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
""", unsafe_allow_html=True)




city = st.selectbox(
        "City",
        sorted(cleaned_data["city"].unique())
    )

cuisines = st.multiselect(
        "Cuisine",
        sorted({c.strip() for x in cleaned_data["cuisine"] for c in x.split(",")})
    )



max_cost = st.slider("Max Cost", 50, 1000, 300)
min_rating = st.slider("Minimum Rating", 1.0, 5.0, 4.0)

min_rating_count = st.slider(
    "Minimum Rating Count",
    min_value=0,
    max_value=int(cleaned_data["rating_count"].max()),
    value=50
)


if st.button("Recommend"):

    # Create user vector
    user_vector = encoded_data.mean() * 0

    city_col = f"city_{city}"
    if city_col in user_vector.index:
        user_vector[city_col] = 1

    for cuisine in cuisines:
        if cuisine in user_vector.index:
            user_vector[cuisine] = 1

    user_vector["cost"] = max_cost
    user_vector["rating"] = min_rating

    # Scale & predict cluster
    user_scaled = scaler.transform(user_vector.values.reshape(1, -1))
    cluster_id = kmeans.predict(user_scaled)[0]

    st.subheader("⭐ Recommended Restaurants")

    
    recommended = cleaned_data[
        (cleaned_data["cluster"] == cluster_id) &
        (cleaned_data["city"].str.lower() == city.lower()) &
        (cleaned_data["cost"] <= max_cost) &
        (cleaned_data["rating"] >= min_rating) &
        (cleaned_data["rating_count"] >= min_rating_count)
    ]


 
    recommended = (
        recommended
        .drop(columns=["cluster"])
        .sort_values(by=["rating", "rating_count"], ascending=False)
        .head(5)
    )

   
    if recommended.empty:
        st.warning("No restaurants match your filters. Try lowering rating count or cost.")
    else:
        st.dataframe(recommended)
