#### Importing dependencies
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

### Function to perform clustering and visualization for spending score analysis
def perform_clustering(filtered_df, attributes, selected_category, spending_min, spending_moderate, spending_max):
    if filtered_df.empty:
        st.warning(f"No data available for the selected age range: 0 - {children_and_teens_max_age}.")
        return

    # Prepare data for clustering
    if attributes == 'Age':
        X = filtered_df[['Age', 'Spending Score (1-100)']].dropna().values
    elif attributes == 'Annual Income ($)':
        X = filtered_df[['Annual Income ($)', 'Spending Score (1-100)']].dropna().values
    else:
        st.warning("Invalid attribute for clustering.")
        return

    # Perform K-means clustering
    wcss = []
    for i in range(1, min(11, len(X) + 1)):  # Ensure the loop runs up to the number of samples if less than 10
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    # Finding optimal number of clusters
    if len(wcss) > 2:  # Ensure there are enough points to calculate second differences
        differences = np.diff(wcss)
        second_differences = np.diff(differences)
        if len(second_differences) > 0:  # Ensure second differences are not empty
            knee_index = np.argmax(second_differences) + 1
            optimal_clusters = knee_index + 1
        else:
            optimal_clusters = 3  # Default to 3 clusters if no elbow is found
    else:
        optimal_clusters = 3  # Default to 3 clusters if not enough data points

    # Ensure the number of clusters is not greater than the number of samples
    if optimal_clusters > len(X):
        optimal_clusters = len(X)

    # Training K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
    Y = kmeans.fit_predict(X)

    # Add cluster labels to the filtered dataframe
    filtered_df['Cluster'] = Y

    # Define spending score categories
    spending_bins = [0, spending_min, spending_moderate, spending_max]
    spending_labels = ['Low Spenders', 'Moderate Spenders', 'High Spenders']
    filtered_df['Spending Category'] = pd.cut(filtered_df['Spending Score (1-100)'], bins=spending_bins, labels=spending_labels, include_lowest=True)

    # Create pie chart for spending score distribution
    category_counts = filtered_df['Spending Category'].value_counts().sort_index()

    # Plotting spending score distribution
    st.markdown(f'<h2 style="text-align:center; color:#ff6347;">Spending Score Distribution for {selected_category}</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(category_counts, labels=[f'{label} ({category_counts[label]})' for label in category_counts.index], autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62', '#8da0cb'])
    st.pyplot(fig)

### Function to perform clustering and visualization for products preferences analysis
def perform_product_clustering(filtered_df, selected_category):
    # Handling key error
    if not ('Age' in df.columns and 'Top Products Ordered' in df.columns):
        st.warning("""Required data not present for segmentation. 
                   Select data source with relevant data""")
        return

    # Handling empty data frame error
    if filtered_df.empty:
        st.warning(f"No data available for the selected age category: {selected_category}.")
        return

    # Prepare data for clustering
    X = filtered_df[['Age', 'Top Products Ordered']].dropna()

    # Convert categorical data to numerical using one-hot encoding
    X_encoded = pd.get_dummies(X, columns=['Top Products Ordered'])

    # Perform K-means clustering
    wcss = []
    for i in range(1, min(11, len(X_encoded) + 1)):  # Ensure the loop runs up to the number of samples if less than 10
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X_encoded)
        wcss.append(kmeans.inertia_)

    # Finding optimal number of clusters
    if len(wcss) > 2:  # Ensure there are enough points to calculate second differences
        differences = np.diff(wcss)
        second_differences = np.diff(differences)
        if len(second_differences) > 0:  # Ensure second differences are not empty
            knee_index = np.argmax(second_differences) + 1
            optimal_clusters = knee_index + 1
        else:
            optimal_clusters = 3  # Default to 3 clusters if no elbow is found
    else:
        optimal_clusters = 3  # Default to 3 clusters if not enough data points

    # Ensure the number of clusters is not greater than the number of samples
    if optimal_clusters > len(X_encoded):
        optimal_clusters = len(X_encoded)

    # Training K-means with optimal number of clusters
    kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
    Y = kmeans.fit_predict(X_encoded)

    # Add cluster labels to the filtered dataframe
    filtered_df['Cluster'] = Y

    # Display insights for top products ordered by age category
    st.markdown(f'<h2 style="text-align:center; color:#ff6347;">Products Preference Distribution for {selected_category}</h2>', unsafe_allow_html=True)
    product_counts = filtered_df['Top Products Ordered'].value_counts()

    # Plotting product preference distribution
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.pie(product_counts, labels=[f'{label} ({product_counts[label]})' for label in product_counts.index], autopct='%1.1f%%', startangle=140, colors=['#66c2a5', '#fc8d62', '#8da0cb', 'red'])
    st.pyplot(fig)


### Streamlit app
# Page configuration
st.set_page_config(page_title="Machine Learning Customer Segmentation System", page_icon=":bar_chart:", layout="wide")

# Sidebar navigation
with st.sidebar:
    st.markdown('<div style="font-size: 17px; font-weight: bold; margin-bottom: -10px; color: green;">Analysis Options</div>', unsafe_allow_html=True)
    page = st.selectbox('', options=['Spending Score Analysis', 'Product Groups Preference Analysis'])

    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown('<div style="font-size: 17px; font-weight: bold; margin-bottom: -10px; color: green;">Configurations</div>', unsafe_allow_html=True)
    
    st.markdown('<br>', unsafe_allow_html=True)

    # Collapsible section for Spending Score Categories
    with st.expander('Spending Score Categories'):
        spending_min = st.number_input('Low Spenders max score', key='spending_min', min_value=0, max_value=100, value=33)
        spending_moderate = st.number_input('Moderate Spenders max score', key='spending_moderate', min_value=0, max_value=100, value=66)
        spending_max = st.number_input('High Spenders max score', key='spending_max', min_value=0, max_value=100, value=99)

    # Collapsible section for Age Categories
    with st.expander('Age Categories'):
        # Children and Teens category
        children_and_teens_max_age = st.number_input('Children and Teens max age', key='children_and_teens_max_age', min_value=0, max_value=100, value=19)

        # Youth category
        youth_max_age = st.number_input('Youth max age', key='youth_max_age', min_value=0, max_value=100, value=29)

        # Adult category
        adult_max_age = st.number_input('Adult max age', key='adult_max_age', min_value=0, max_value=100, value=49)

        # Elderly category
        elderly_max_age = st.number_input('Elderly max age', key='elderly_max_age', min_value=0, max_value=100, value=100)


if page == 'Spending Score Analysis':
    st.markdown('<style>body{background-color: #f0f2f6;}</style>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .header {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            background-color: #ffffff;
            padding: 10px 0;
        }
        .header h1 {
            font-size: 36px;
            margin: 0;
            color: #ff6347;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Full-width header with a title
    st.markdown(
        """
        <div class="header">
            <h1>Machine Learning Customer Segmentation System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)

# Intro
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Upload sales data in CSV or Excel format to perform customer segmentation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<br><br>', unsafe_allow_html=True)

# Data upload
    st.markdown('<label style="font-size: 20px; font-weight: bold;">Upload Data</label>', unsafe_allow_html=True)
    files = st.file_uploader("Upload CSV or Excel files", type=['xlsx', 'csv'], accept_multiple_files=True)
    if files:
        file_names = [file.name for file in files]

        st.markdown('<br>', unsafe_allow_html=True)
        selected_file = st.selectbox('Select Data Source For Customer Segmentation', options=file_names)

# Load the selected file + data overview
        selected_file_index = file_names.index(selected_file)
        if selected_file.endswith('.csv'):
            df = pd.read_csv(files[selected_file_index])
        elif selected_file.endswith('.xlsx'):
            df = pd.read_excel(files[selected_file_index])

        st.markdown('<label style="font-size: 16px; font-weight: bold;">Data Overview</label>', unsafe_allow_html=True)
        st.write(df.head())

        st.markdown('<br><br>', unsafe_allow_html=True)

# Attribute selection
        st.markdown('<label style="font-size: 16px; font-weight: bold; margin-bottom: -10px;">Explore Customer Spending Score by</label>', unsafe_allow_html=True)
        attributes = st.selectbox('', options=['None', 'Age'])

        if attributes == 'None':
            st.warning(f"Select an attribute for customer segmentation")

        if attributes != 'None':
            # st.markdown(f"<h3 style='color:green;'>Selected Attribute: {attributes}</h3>", unsafe_allow_html=True)

            if attributes == 'Age':
                # Categorize age into groups
                age_bins = [0, children_and_teens_max_age, youth_max_age, adult_max_age, elderly_max_age]
                age_labels = ['Children and Teens', 'Youth', 'Adult', 'Elderly']
                df['Age Category'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

                # Allow user to select age category for insights visualization
                selected_category = st.selectbox('Select Age Category', options=age_labels)

                # Filter data by selected age category
                filtered_df = df[df['Age Category'] == selected_category].copy()

                # Perform K-means clustering
                perform_clustering(filtered_df, attributes, selected_category, spending_min, spending_moderate, spending_max)

            else:
                st.warning("Please select an attribute to proceed.")

    ### Summary
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown(
    """
    <div style='text-align: center;'>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True
    )

elif page == 'Product Groups Preference Analysis':
    st.markdown('<style>body{background-color: #f0f2f6;}</style>', unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .header {
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            background-color: #ffffff;
            padding: 10px 0;
        }
        .header h1 {
            font-size: 36px;
            margin: 0;
            color: #ff6347;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# Full-width header with a title
    st.markdown(
        """
        <div class="header">
            <h1>Machine Learning Customer Segmentation System</h1>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<br>', unsafe_allow_html=True)

# Intro
    st.markdown(
        """
        <div style='text-align: center;'>
            <p>Upload sales data in CSV or Excel format to perform customer segmentation.</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<br><br>', unsafe_allow_html=True)

# Data upload
    st.markdown('<label style="font-size: 20px; font-weight: bold;">Upload Data</label>', unsafe_allow_html=True)
    files = st.file_uploader("Upload CSV or Excel files", type=['xlsx', 'csv'], accept_multiple_files=True)
    if files:
        file_names = [file.name for file in files]

        st.markdown('<br>', unsafe_allow_html=True)
        selected_file = st.selectbox('Select Data Source For Customer Segmentation', options=file_names)

# Load the selected file + data overview
        selected_file_index = file_names.index(selected_file)
        if selected_file.endswith('.csv'):
            df = pd.read_csv(files[selected_file_index])
        elif selected_file.endswith('.xlsx'):
            df = pd.read_excel(files[selected_file_index])

        st.markdown('<label style="font-size: 16px; font-weight: bold;">Data Overview</label>', unsafe_allow_html=True)
        st.write(df.head())

        st.markdown('<br><br>', unsafe_allow_html=True)

# Attribute selection
        st.markdown('<label style="font-size: 16px; font-weight: bold; margin-bottom: -10px;">Explore Product Groups Preference by</label>', unsafe_allow_html=True)
        attributes = st.selectbox('', options=['None', 'Age'])

        if attributes == 'None':
            st.warning("Select an attribute for customer segmentation")

        elif attributes != 'None':
            if attributes == 'Age':
                # Categorize age into groups
                age_bins = [0, children_and_teens_max_age, youth_max_age, adult_max_age, elderly_max_age]
                age_labels = ['Children and Teens', 'Youth', 'Adult', 'Elderly']
                df['Age Category'] = pd.cut(df['Age'], bins=age_bins, labels=age_labels, include_lowest=True)

                # Allow user to select age category for insights visualization
                selected_category = st.selectbox('Select Age Category', options=age_labels)

                # Filter data by selected age category
                filtered_df = df[df['Age Category'] == selected_category].copy()

                # Perform K-means clustering
                perform_product_clustering(filtered_df, selected_category)

    ### Summary
    st.markdown('<br><br>', unsafe_allow_html=True)
    st.markdown(
    """
    <div style='text-align: center;'>
        <p></p>
    </div>
    """,
    unsafe_allow_html=True
    )



    