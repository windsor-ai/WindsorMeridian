import requests
from io import StringIO
import pandas as pd


def getDataFromWindsor():
    try:
        print("<--- Meridian MMM using Windsor.ai Data --->")
        # Get the api key from user
        api_key = input("Enter your Windsor.ai data api key: ")

        # Get the Windsor.ai API endpoint from windsor.ai platform with query parameters to get ads data:
        # - api_key: your API key
        # - date_preset: fetches data from last year
        # - fields: specifies which fields to fetch from ads data
        api_url = f"https://connectors.windsor.ai/all?api_key={api_key}&date_preset=last_year&fields=account_name,campaign,clicks,datasource,date,event_count,event_name,impressions,is_conversion_event,medium,sessions,source,spend,transactionrevenue"

        # Make the GET request to the API
        print("Getting Data...")
        response = requests.get(api_url)

        if 'data' not in response.json():
            raise ValueError("Missing 'data' field in response JSON", response.json())

        # Extract the 'data' portion of the JSON response
        data = response.json()['data']

        # Load the data into a pandas DataFrame
        df = pd.DataFrame(data)

        if df.empty:
            raise ValueError("DataFrame is empty after loading data!")

        df.to_csv("raw_data.csv", index=False)
        print("Data Gathered! Raw data csv generated.")

        # Get input from user for KPI Type
        while True:
            try:
                is_revenue = int(input("Select KPI: Revenue or Conversions. Enter 1 for Revenue and 0 for Conversions: "))
                if is_revenue in [0, 1]:
                    break
                else:
                    print("Invalid input! Please enter 1 for Revenue or 0 for Conversions.")
            except ValueError:
                print("Invalid input! Please enter a number (0 or 1).")
        
        kpi = "revenue" if is_revenue == 1 else "conversions"

        print(f"Selected KPI: {kpi.capitalize()}")

        # Get conversions/revenue data from Windsor.ai data. We are using GA4 data to get conversions from Windsor.
        ga4_condition = df['datasource'] == 'googleanalytics4'
        searchconsole_condition = df['datasource'] == 'searchconsole'
        ga4_df = df[ga4_condition]

        # Extract Event Names for conversions
        if kpi == 'conversions':
            events_df = ga4_df[ga4_df['is_conversion_event'] == 'true'].copy()
            events = events_df['event_name'].unique().tolist()
            print("Available GA4 Events for conversions selection:")
            print(", ".join(events))

            while True:
                # Get user input
                user_input = input("Enter the events you want to use as conversions (comma-separated): ")
                # Create list from user input (trim whitespace and lowercase optional)
                selected_events = [event.strip() for event in user_input.split(',')]
                # Validate entered events
                invalid_events = [e for e in selected_events if e not in events]

                if invalid_events:
                    print(f"Invalid event(s) entered: {', '.join(invalid_events)}")
                else:
                    print("Selected events for conversions:", selected_events)
                    break


            # Extract revenue and conversions
            conversions_df = ga4_df[ga4_df['is_conversion_event'] == 'true'].copy()
            conversions_df = conversions_df[conversions_df['event_name'].isin(selected_events)]
            conversions_df = conversions_df[['date', 'event_count']]
            conversions_df.rename(columns={'event_count': 'conversions'}, inplace=True)
            conversions_df = conversions_df.groupby(['date']).sum().reset_index()

        revenue_df = ga4_df[['date', 'transactionrevenue']].copy()
        revenue_df.rename(columns={'transactionrevenue': 'revenue'}, inplace=True)
        revenue_df = revenue_df.groupby(['date']).sum().reset_index()

        if revenue_df['revenue'].sum() == 0 and kpi == 'revenue':
            raise ValueError("Total Revenue (transactionrevenue) is zero. Please check your data.")


        # Extract organic clicks
        organic_df = df[searchconsole_condition]
        if organic_df.empty:
            organic_df = ga4_df[ga4_df['medium'] == 'organic'].copy()
            organic_df = organic_df[['date', 'sessions']]
            organic_df.rename(columns={'sessions': 'organic'}, inplace=True)
        else:
            organic_df = organic_df[['date', 'clicks']]
            organic_df.rename(columns={'clicks': 'organic'}, inplace=True)


        organic_df = organic_df.groupby(['date']).sum().reset_index()

        # Keep only relevant columns depending on your analysis needs
        df = df[~df['datasource'].isin(['googleanalytics4', 'searchconsole'])]
        df = df[['date', 'datasource', 'impressions', 'clicks', 'spend']]

        # Group the data by date and sum numerical metrics across all campaigns/accounts for each day.
        # Make sure you have fields aggregated on which you want to run the model
        # Meridian requires one row per day for time series modeling
        df = df.groupby(['date', 'datasource']).sum().reset_index()

        # Reshape the data to have separate impressions, clicks, conversions, spend for different sources
        df = df.pivot(index='date', columns='datasource',
                                    values=['impressions', 'spend', 'clicks'])

        # Flatten MultiIndex columns
        df.columns = [f"{col[1]}_{col[0]}" for col in df.columns]
        df = df.reset_index()

        # Merge pivoted data with conversions/revenue/organic
        if kpi == 'conversions':
            df = df.merge(conversions_df, on='date', how='outer')
        
        df = df.merge(revenue_df, on='date', how='outer')
        df = df.merge(organic_df, on='date', how='outer')

        # Convert the 'date' column to datetime format
        df['date'] = pd.to_datetime(df['date'])

        # Sort the data by date
        df = df.sort_values(by=['date'])

        # This step is necessary if you dont have evenly spaced time series data required by meridian
        # Resample the data to have one row per day (daily frequency)
        # This fills in missing dates with NaNs (important for time series modeling)
        df = df.set_index('date').resample('D').asfreq()

        # Fill missing values with zeros you can fill with any values suited to your analysis (assumes no activity on missing dates)
        for suffix in ['clicks', 'impressions', 'conversions', 'spend']:
            cols_to_fill = df.columns[df.columns.str.endswith(suffix)]
            df[cols_to_fill] = df[cols_to_fill].fillna(0)

        # Reset the index to move 'date' back into a column
        df = df.reset_index()
        df['geo'] = "Geo0"

        # Set date as index to resample
        df.set_index('date', inplace=True)

        # Resample weekly (Sunday to Saturday) by summing numeric columns and keeping the first geo
        df = df.resample('W-SAT').agg({
            **{col: 'sum' for col in df.columns if col != 'geo'},
            'geo': 'first'
        }).reset_index()

        df.to_csv("processed_data.csv", index=False)
        print("Processed data csv generated!")

        df = pd.read_csv("processed_data.csv")

        # Extract Start and End Date
        start_date = df['date'].iloc[0]
        end_date = df['date'].iloc[-1]

        # Extract channels
        click_columns = [col for col in df.columns if col.endswith('_clicks')]
        if 'organic' in click_columns:
            click_columns.remove('organic')
        spend_columns = [col for col in df.columns if col.endswith('_spend')]
        media_to_channel = {col: col.replace('_clicks', '').capitalize() for col in click_columns}
        media_spend_to_channel = {col: col.replace('_spend', '').capitalize() for col in spend_columns}

        # Convert the final DataFrame to a CSV string in memory using StringIO
        csv_data = StringIO()
        df.to_csv(csv_data, index=False)
        csv_data.seek(0)  # Move to the start of the stream

        # Return the object which contains the required data for mmmm
        return {'data': csv_data, 
                'media': click_columns, 
                'media_spend': spend_columns, 
                'media_to_channel': media_to_channel, 
                'media_spend_to_channel': media_spend_to_channel, 
                'start_date': start_date, 
                'end_date': end_date, 
                'kpi': kpi, 
                'controls': ['organic']}
    
    except requests.exceptions.RequestException as e:
        print(f"Network/API error: {e}")
    except ValueError as ve:
        print(f"Data error: {ve}")
    except KeyError as ke:
        print(f"Missing expected column: {ke}")
    except Exception as ex:
        print(f"Unexpected error: {ex}")
