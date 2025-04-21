import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px 
import plotly.graph_objects as go
import numpy as np
import seaborn as sns

### Config
st.set_page_config(
    page_title="Getaround Rental Delay Analysis",
    page_icon="car",
    layout="wide"
)

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

local_css("style.css")

RENTAL_FILE_ID = "1aos7aPFI2nnE4A_S-DACZINCTm6eMzQd"
RENTAL_DATA_URL = f"https://drive.google.com/uc?export=download&id={RENTAL_FILE_ID}"


PRICING_FILE_ID = "1JVd1ZD6PK1nMrcoVTFwMl4swCd337fyK"
PRICING_DATA_URL = f"https://drive.google.com/uc?export=download&id={PRICING_FILE_ID}"


### App
st.title("Getaround Analysis")

# Use `st.cache` when loading data is extremly useful
# because it will cache your data so that your app 
# won't have to reload it each time you refresh your app

### === Raw data ===

@st.cache_data
def load_df_pricing():
    df_pricing = pd.read_csv(PRICING_DATA_URL, sep=",")
    df_pricing.drop(df_pricing.columns[0], axis=1, inplace = True)
    return df_pricing

@st.cache_data
def load_df():
    df = pd.read_excel(RENTAL_DATA_URL)
    return df

st.header("Load and showcase data")


data_load_state = st.text('Loading data...')
df = load_df()
df_pricing = load_df_pricing()
data_load_state.text("") # change text from "Loading data..." to "" once the the load_data function has run

## Run the below code if the check is checked ✅
if st.checkbox('Show raw data'):
    st.subheader('Raw rental data')
    st.write(df) 
    st.write("*"*100)
    st.subheader('Raw pricing data')
    st.write(df_pricing)

st.divider()

### Connected cars vs non-connected 
st.header("Connected cars vs non-connected")

## Create two columns
col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    st.subheader("1️⃣ Ratio of connected cars vs non-connected")
    st.markdown("""
    ##### Connected cars are slightly less represented in the base: 2230 connected vs 2613 non-connected.
        
    """)
    # Get revenue sum per group of cars
    revenue_by_connect = df_pricing.groupby("has_getaround_connect")['rental_price_per_day'].sum().reset_index()
    
    # Create a dictionary for displaying labels
    connect_labels = {
        True: "Connected",
        False: "Non-connected",
        
    }
    revenue_by_connect['connection_type'] = revenue_by_connect['has_getaround_connect'].map(connect_labels)
    
    
    fig = px.pie(
    revenue_by_connect,
    names="connection_type",
    values="rental_price_per_day",
    title="Distribution of Revenue Between Connected and Non-Connected Cars",
    hole=0.3,  # make dohnut
    labels={False: "Non-connected", True: "Connected"},
)

    # Add sums as hoverinfo
    fig.update_traces(textinfo="percent+label", hoverinfo="label+value")

    # Show graph with Streamlit
    st.plotly_chart(fig)

with col2:
    st.subheader("2️⃣ Rental price distribution")
    st.markdown("""
                ##### Average rental prices for the Connected category are a bit greater: 132 EUR/day vs 111 EUR/day for Non-connected.
                ##### So, Connected cars generate almost similar revenue with Non-connected.
                """)

    # Create boxplot
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='has_getaround_connect', y='rental_price_per_day', data=df_pricing, ax=ax)
    ax.set_title('Rental Price Distribution by Connection Type')
    ax.set_xlabel('Has Getaround Connect')
    ax.set_ylabel('Rental Price Per Day')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(['Non-connected', 'Connected'])

    # Show plot in Streamlit
    st.pyplot(fig)

st.markdown(" ")
st.divider()

### === conflict cases analysis ===

st.header("Late checkouts and conflict cases analysis")

# Filtering data
df_valid = df[
    df["delay_at_checkout_in_minutes"].notna() & 
    df["time_delta_with_previous_rental_in_minutes"].notna()
].copy()

# Share of rentals with delay
delayed_rentals = df_valid[df_valid["delay_at_checkout_in_minutes"] > 0]
total_delayed = delayed_rentals.shape[0]
total_rentals = df_valid.shape[0]
delay_share = total_delayed / total_rentals

# Conflict cases (delay greater than buffer)
conflicts = delayed_rentals[
    delayed_rentals["delay_at_checkout_in_minutes"] > 
    delayed_rentals["time_delta_with_previous_rental_in_minutes"]
]
total_conflicts = conflicts.shape[0]
conflict_share = total_conflicts / total_delayed

# Metrics
st.subheader("📌 Key Metrics")
st.markdown(" ")

col1, col2, col3 = st.columns(3)
col1.metric("Rentals with Delay", f"{total_delayed}", f"{delay_share:.1%} of total")
col2.metric("Conflicts with Next Rental", f"{total_conflicts}", f"{conflict_share:.1%} of delayed")
col3.metric("Total Valid Rentals", f"{total_rentals}")

st.markdown(" ")

st.markdown("""
##### Of the 1,515 valid leases, 802 (or ~53%) ended with a delayed vehicle return.

##### Of these delayed rentals, 270 cases (or ~34%) resulted in a conflict with a subsequent rental - meaning the next driver could suffer from the tardiness of the previous driver.

##### 🔍 This means that one in three delays creates a risk of the next rental failing, potentially degrading the customer experience and potentially reducing trust in the platform.
            """)
st.markdown("<div style='margin-top: 120px', 'margin-bottom: 120px'> </div>", unsafe_allow_html=True )


# Create two columns
col1, spacer, col2 = st.columns([1, 0.2, 1])

with col1:
    # Pie chart: rentals with delay ratio
    st.subheader("📊 Share of Rentals with Delay")

    delay_labels = ["With Delay", "No Delay"]
    delay_counts = [total_delayed, total_rentals - total_delayed]

    fig1, ax1 = plt.subplots()
    ax1.pie(delay_counts, labels=delay_labels, autopct='%1.1f%%', startangle=90, colors=["#8661C1", "#97D8B2"])
    ax1.axis('equal')
    st.pyplot(fig1)

with col2:
    # Pie chart: delays with conflicts
    st.subheader("⚠️ Share of Delays Leading to Conflict")

    conflict_labels = ["Conflict", "No Conflict"]
    conflict_counts = [total_conflicts, total_delayed - total_conflicts]

    fig2, ax2 = plt.subplots()
    ax2.pie(conflict_counts, labels=conflict_labels, autopct='%1.1f%%', startangle=90, colors=["#8661C1", "#97D8B2"])
    ax2.axis('equal')
    st.pyplot(fig2)

st.markdown("<div style='margin-top: 120px', 'margin-bottom: 120px'> </div>", unsafe_allow_html=True )

st.divider()

# === Delay statistics with outlier filtering and improved layout ===

# IQR filtering function
def filter_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Section title
st.header("⏱️ Delay Statistics")

# Checkbox for filtering
apply_filter = st.checkbox("Exclude extreme outliers from delay statistics (based on IQR)")

# Filter data
if apply_filter:
    delayed_data = filter_outliers_iqr(delayed_rentals, "delay_at_checkout_in_minutes")
    st.markdown("✅ *Outliers excluded using IQR method*")
else:
    delayed_data = delayed_rentals
    st.markdown("⚠️ *Outliers included (raw data)*")

st.markdown(" ")
st.markdown(" ")
st.markdown(" ")
# Horizontal layout for metrics
col1, col2, col3 = st.columns(3)
col1.metric("Average delay", f"{delayed_data['delay_at_checkout_in_minutes'].mean():.1f} min")
col2.metric("Median delay", f"{delayed_data['delay_at_checkout_in_minutes'].median():.1f} min")
col3.metric("Max delay", f"{delayed_data['delay_at_checkout_in_minutes'].max():.1f} min")

# Text with insights below
st.markdown("""
<div class="insights">
    <h3 class="insights-title">📌 Insights</h3>
    <ul class="insights-list">
        <li class="insight-item">More than half of rentals (<strong>52.9%</strong>) are delayed, indicating that cars are often not returned on time.</li>
        <li class="insight-item">Only ~<strong>33.7%</strong> of delays result in conflicts with subsequent rentals, meaning most delays do not interfere with the next customer.</li>
        <li class="insight-item">The <strong>average delay</strong> is about <strong>160 minutes</strong>, while the <strong>median</strong> is around <strong>50 minutes</strong>, showing that a few long delays (up to <strong>{:.0f} minutes</strong>) skew the average.</li>
    </ul>
</div>
""".format(delayed_data['delay_at_checkout_in_minutes'].max()), unsafe_allow_html=True)

# === Expandable histogram section ===

with st.expander("📈 Show delay distribution histogram "): 
    

    # Radio button inside expander for the histogram
    hist_filter_option = st.radio(
        "Choose data to display:",
        ("Include outliers", "Exclude outliers (IQR filtered)"),
        horizontal=True
    )

    # Apply selected filter
    if hist_filter_option == "Exclude outliers (IQR filtered)":
        data_for_hist = filter_outliers_iqr(delayed_rentals, "delay_at_checkout_in_minutes")
        st.markdown("✅ *Outliers excluded from histogram*")
    else:
        data_for_hist = delayed_rentals
        st.markdown("⚠️ *Histogram includes extreme delays*")

    # Draw histogram
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.histplot(
        data_for_hist["delay_at_checkout_in_minutes"],
        bins=60,
        kde=True,
        color="mediumpurple",
        edgecolor="black",
        ax=ax
    )
    ax.set_title("Distribution of Delays (in minutes)")
    ax.set_xlabel("Delay at Checkout (minutes)")
    ax.set_ylabel("Number of Rentals")
    st.pyplot(fig)
st.markdown("<div style='margin-top: 120px', 'margin-bottom: 120px'> </div>", unsafe_allow_html=True )

st.divider()

### === Efficiency vs revenue losses ===

# List of threshold
thresholds = [30, 60, 90, 120, 180]

# Calculate mean rental prices
pricing_stats = df_pricing.groupby("has_getaround_connect")["rental_price_per_day"].mean()
price_connect = pricing_stats[True]
price_non_connect = pricing_stats[False]

# Filtering data
df_valid = df[
    df["previous_ended_rental_id"].notna() & 
    df["time_delta_with_previous_rental_in_minutes"].notna() &
    df["delay_at_checkout_in_minutes"].notna()
].copy()
st.markdown(" ")

# Section subheader
st.header("Efficiency vs revenue losses per threshold")

# Threshold selection (in min) 
threshold = st.selectbox("Select the threshold (in minutes)", thresholds, index=0)


# Create 2 columns
col1, col2 = st.columns(2)

# Define function for calculation statistics for both scopes
def calculate_scope_data(scope, threshold):
    if scope == "All cars":
        scope_df = df_valid.copy()
    else:
        scope_df = df_valid[df_valid["checkin_type"] == "connect"]

    total_rentals = scope_df.shape[0]
    total_connect = scope_df[scope_df["checkin_type"] == "connect"].shape[0]
    total_non_connect = scope_df[scope_df["checkin_type"] != "connect"].shape[0]
    total_revenue = (total_connect * price_connect) + (total_non_connect * price_non_connect)

    # Affected rentals
    affected = scope_df[scope_df["time_delta_with_previous_rental_in_minutes"] < threshold]
    affected_connect = affected[affected["checkin_type"] == "connect"].shape[0]
    affected_non_connect = affected[affected["checkin_type"] != "connect"].shape[0]
    affected_total = affected.shape[0]
    affected_revenue = (affected_connect * price_connect) + (affected_non_connect * price_non_connect)
    revenue_share = affected_revenue / total_revenue if total_revenue > 0 else 0

    # Saved rentals
    conflict_cases = scope_df[
        scope_df["delay_at_checkout_in_minutes"] > scope_df["time_delta_with_previous_rental_in_minutes"]
    ]
    saved = conflict_cases[conflict_cases["time_delta_with_previous_rental_in_minutes"] < threshold]
    saved_count = saved.shape[0]

    # Efficiency
    efficiency = saved_count / affected_total if affected_total > 0 else 0

    return {
        "Affected rentals": affected_total,
        "Saved rentals": saved_count,
        "Revenue loss (%)": round(revenue_share * 100, 2),
        "Revenue loss (€)": round(affected_revenue, 2),
        "Efficiency": round(efficiency, 3)
        
    }

# Calculate results for both scopes
results_all_cars = calculate_scope_data("All cars", threshold)
results_connect_only = calculate_scope_data("Connect only", threshold)

# Display two dataframes
df_all_cars = pd.DataFrame([results_all_cars])
df_connect_only = pd.DataFrame([results_connect_only])

# Visualization in two columns
with col1:
    st.subheader(f"Results for All cars (threshold = {threshold} min)")
    st.write(df_all_cars)

    # Plot for all cars
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 6))
    ax1.bar(["Revenue loss (%)", "Saved rentals", "Efficiency"], 
            [results_all_cars["Revenue loss (%)"], results_all_cars["Saved rentals"], results_all_cars["Efficiency"]])
    for i, v in enumerate([results_all_cars["Revenue loss (%)"], results_all_cars["Saved rentals"], results_all_cars["Efficiency"]]):
        ax1.text(i, v + 1, f'{v}', ha='center', va='bottom', fontsize=10)  # add notation
    ax1.set_title("Impact Analysis for All Cars")

    # Define common scale for Y axis (max for all the data)
    y_max = max([results_all_cars["Revenue loss (%)"], results_all_cars["Saved rentals"], results_all_cars["Efficiency"], 
                 results_connect_only["Revenue loss (%)"], results_connect_only["Saved rentals"], results_connect_only["Efficiency"]]) + 10
    ax1.set_ylim(0, y_max)
    st.pyplot(fig1)

with col2:
    st.subheader(f"Results for Connect only (threshold = {threshold} min)")
    st.write(df_connect_only)

    # Plot for connect only
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    ax2.bar(["Revenue loss (%)", "Saved rentals", "Efficiency"], 
            [results_connect_only["Revenue loss (%)"], results_connect_only["Saved rentals"], results_connect_only["Efficiency"]])
    for i, v in enumerate([results_connect_only["Revenue loss (%)"], results_connect_only["Saved rentals"], results_connect_only["Efficiency"]]):
        ax2.text(i, v + 1, f'{v}', ha='center', va='bottom', fontsize=10)  # Добавляем подписи
    ax2.set_title("Impact Analysis for Connect Only")

    # Define common scale for Y axis 
    ax2.set_ylim(0, y_max)
    st.pyplot(fig2)

### === Conclusions ===

st.markdown("""
<div class="conclusions">
    <h3 class="conclusions-title">📌 Conclusions</h3>
    <ul class="conclusions-list">
        <li class="conclusions-item">Maximum efficiency is maximized at low thresholds (<strong>30-60 minutes</strong>).</li>
        <li class="conclusions-item">Example: at <strong>30 min</strong>, for all cars we save <strong>~60%</strong> of problem cases while losing <strong>15%</strong> of revenue.</li>
        <br>
        <li class="conclusions-item">As the threshold increases, the efficiency drops dramatically and the losses increase.</li>
        <br>
        <li class="conclusions-item">Filtering for <strong>Connect-only</strong> cars may result in <strong>fewer rentals saved</strong>, but a <strong>larger share of revenue is retained</strong>.</li>
        <li class="conclusions-item">This is because these cars tend to be <strong>more expensive to rent</strong> and saving them is important to retain a significant share of revenue.</li>
        <li class="conclusions-item">Example: at <strong>60 min</strong>: <strong>63</strong> rentals saved, but only <strong>~23%</strong> loss, vs. <strong>~22%</strong> at <strong>176</strong> saved across all cars.</li>        
    </ul>
</div>
""" , unsafe_allow_html=True)
st.divider()

### ===Recommendations ===

st.header("✅ Recommendations")

st.markdown ("#### 🎯 *Base* option:")
st.markdown("""
<div class="recommendations">
    <ul class="recommendations-list">
            <li class="recommendations-item">Threshold: <strong>60 minutes</strong></li>
            <li class="recommendations-item">Scope: <strong>Connect only</strong></li>
            <li class="recommendations-item"><strong>Saves</strong> a decent number of rentals (<strong>63</strong>) with a <strong>reasonable loss</strong> of revenue (<strong>22.7%</strong>)</li>
            <li class="recommendations-item">Efficiency: <strong>42%</strong> - good ratio</li>                      
    </ul>
</div>
""" , unsafe_allow_html=True)

st.markdown ("#### 🛡️ Alternative *conservative* option:")
st.markdown("""
<div class="recommendations">            
    <ul class="recommendations-list">
            <li class="recommendations-item">Threshold: <strong>30 minutes</strong></li>
            <li class="recommendations-item">Scope: <strong>Connect only</strong></li>
            <li class="recommendations-item"> <strong>Minimal loss</strong>: <strong>15.9%</strong> of revenue</li>
            <li class="recommendations-item">Nearly <strong>half of conflicts</strong> are prevented</li>                        
    </ul>
</div>
""" , unsafe_allow_html=True)

st.markdown ("#### 🚀 If the goal is to *maximize customer experience*, not just minimize losses:")
st.markdown("""
<div class="recommendations">
    <ul class="recommendations-list">
            <li class="recommendations-item">Threshold: <strong>60 minutes</strong></li>
            <li class="recommendations-item">Scope: <strong>All cars</strong></li>
            <li class="recommendations-item"> Improves service quality (<strong>176 cases saved!</strong>)</li>
            <li class="recommendations-item">But requires a compromise: <strong>22% of revenue lost</strong></li>                        
    </ul>
</div>
""" , unsafe_allow_html=True)