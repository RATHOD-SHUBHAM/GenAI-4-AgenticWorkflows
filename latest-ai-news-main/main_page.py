import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from fetch_data import main

# Use Streamlit's built-in caching
@st.cache_data(ttl=60)  # Cache for 1 minute
def get_data():
    with st.spinner('Fetching latest AI news...'):
        return main()

def run_dashboard():
    st.title("AI News Dashboard")
    
    # Add a refresh button
    if st.button("Refresh Data"):
        # Clear the cache and get fresh data
        st.cache_data.clear()
        st.rerun()
    
    # Load data with caching
    try:
        df = get_data()
        
        # Check if df is empty
        if df.empty:
            st.error("No news data available. Please try refreshing later.")
            return
            
        # Get min and max dates
        min_date = df['date'].min()
        max_date = df['date'].max()
        
        # Create layout with columns
        col1, col2 = st.columns(2)
        
        with col1:
            selected_dates = st.date_input(
                "Choose Date Range",
                value=(min_date, max_date),
                min_value=min_date,
                max_value=max_date
            )
            
            # Handle single date selection
            if len(selected_dates) == 1:
                start_date = selected_dates[0]
                end_date = selected_dates[0]
            else:
                start_date, end_date = selected_dates
        
        with col2:
            # Get unique sources
            all_sources = sorted(df['Source'].unique().tolist())
            
            # Add "All" option at the beginning of the list
            source_options = ["All"] + all_sources
            
            # Use multiselect
            selected_sources = st.multiselect(
                "Choose one or more sources",
                options=source_options
            )
        
        # Show button
        if st.button("Show News", key="show"):
            if not selected_sources:
                st.error("Please select at least one source to display news.")
            else:
                # Convert dates to datetime
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                # Filter by date range
                df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                
                # Handle "All" selection
                if "All" in selected_sources:
                    # If "All" is selected, don't filter by source
                    pass
                else:
                    # Filter by selected sources
                    df_filtered = df_filtered[df_filtered['Source'].isin(selected_sources)]
                
                # Display results
                if len(df_filtered) > 0:
                    st.success(f"Found {len(df_filtered)} news items")
                    
                    # Show news as cards
                    for index, row in df_filtered.iterrows():
                        st.markdown(f"### [{row['Title']}]({row['Link']})")
                        st.write(f"**Source**: {row['Source']}")
                        st.write(f"**Description**: {row['Description']}")
                        st.write(f"**Date**: {row['date'].strftime('%Y-%m-%d')}")
                        st.markdown("---")  # Add separator between cards
                else:
                    st.warning("No news found with the selected filters. Please adjust your date range or source selection.")
    
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.info("Try refreshing the data using the button above.")

if __name__ == '__main__':
    run_dashboard()