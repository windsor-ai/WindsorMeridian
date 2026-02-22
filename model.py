import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Import core libraries
import tensorflow as tf
import tensorflow_probability as tfp

# Import Meridian constants and core modules for data loading, modeling, and summarization
from meridian import constants
from meridian.data import load
from meridian.model import model
from meridian.model import spec
from meridian.model import prior_distribution
from meridian.analysis import summarizer

# To check available RAM
from psutil import virtual_memory

# Import your own utility to get data from Windsor.ai
from data import getDataFromWindsor
import warnings
warnings.filterwarnings("ignore")

try:
    # -------------------------------
    # ENVIRONMENT RESOURCE CHECK
    # -------------------------------
    # Check available system memory and devices (useful for tuning model parameters)
    ram_gb = virtual_memory().total / 1e9
    print('Your runtime has {:.1f} GB of available RAM'.format(ram_gb))
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    print("Num CPUs Available: ", len(tf.config.experimental.list_physical_devices('CPU')))


    # -------------------------------
    # STEP 1: Load Data (from Windsor.ai in this case)
    # -------------------------------
    # This function returns processed_csv_data, media, media spend, mapping of media to channels and mapping of media spend to channels
    # start data, end data of the data and kpi 
    # It has daily values for clicks, spend, impressions and conversions.
    model_data = getDataFromWindsor()
    print("Media Channels:", model_data['media'])
    csv_data = model_data['data']

    # -------------------------------
    # STEP 2: Define Mapping from Your CSV Columns to Meridian's Format
    # -------------------------------
    # Coordinate-to-columns mapping: Tells Meridian how to interpret the input columns.
    # Replace the column names with your own if you're using a different dataset.
    coord_to_columns = load.CoordToColumns(
        time="date",            # Timestamp column
        geo="geo",              # Optional: region/location (if available, else keep as None or a dummy column)
        controls=model_data['controls'],            # Any external control variables (economic indicators, organic clicks etc.)
        kpi=model_data['kpi'],      # Your Key Performance Indicator (target variable for prediction)
        revenue_per_kpi=None,   # Optional: Revenue generated per conversion (for revenue models)
        media=model_data['media'],  # Media signals (e.g., impressions, clicks)
        media_spend=model_data['media_spend'],  # Spend per media channel
    )

    # Maps your media column to its corresponding media channel (important for attribution and budgeting)
    correct_media_to_channel = model_data['media_to_channel']

    # Maps spend column to the correct media channel
    correct_media_spend_to_channel = model_data['media_spend_to_channel']

    # -------------------------------
    # STEP 3: Load Data into Meridian
    # -------------------------------
    # This loads and prepares the dataset based on the mappings and parameters defined above.
    loader = load.CsvDataLoader(
        csv_path=csv_data,                              # CSV data from Windsor or your source
        kpi_type='non_revenue',                         # Use 'revenue' if you provide revenue_per_kpi
        coord_to_columns=coord_to_columns,
        media_to_channel=correct_media_to_channel,
        media_spend_to_channel=correct_media_spend_to_channel,
    )
    data = loader.load()


    # -------------------------------
    # STEP 4: Define Priors for the Model
    # -------------------------------
    # ROI prior: Used to inform the model of expected returns on ad spend.
    # LogNormal is used since ROI is strictly positive.
    # Assign roi_mu and roi_sigma values according to your dataset
    roi_mu = 0.2
    roi_sigma = 0.9  # Uncertainty/spread in the ROI prior

    prior = prior_distribution.PriorDistribution(
        roi_m=tfp.distributions.LogNormal(roi_mu, roi_sigma, name=constants.ROI_M)
    )

    # Wrap the prior into the model specification
    model_spec = spec.ModelSpec(prior=prior)

    print("Running Meridian MMM on your data...")
    # -------------------------------
    # STEP 5: Initialize the MMM (Marketing Mix Model)
    # -------------------------------
    # This creates a Meridian model object using the input data and model specifications
    mmm = model.Meridian(input_data=data, model_spec=model_spec)


    # -------------------------------
    # STEP 6: Sample from Prior and Posterior Distributions
    # -------------------------------
    # First, sample from the prior distribution (before seeing data). Good for understanding priors.
    # Adjust these values according to your dataset
    mmm.sample_prior(100)  # 100 samples is enough for exploration

    # Then, sample from the posterior (after observing the data)
    # Smaller values (n_chains=2, n_adapt/burnin=100) make it faster for small data or experimentation.
    mmm.sample_posterior(
        n_chains=2,    # Number of independent chains
        n_adapt=100,   # Adaptation steps for tuning the sampler
        n_burnin=100,  # Number of warm-up iterations (burn-in)
        n_keep=200,    # Number of posterior samples to keep
        seed=1         # Random seed for reproducibility
    )


    # -------------------------------
    # STEP 7: Summarize and Save Results
    # -------------------------------
    # Generate an HTML report with parameter summaries, media channel attribution, ROI, and more.
    # The date arrange should be inside the used dataset
    mmm_summarizer = summarizer.Summarizer(mmm)
    file_path = './'                     # Save location for summary report
    start_date = model_data['start_date']            # Analysis start date (adjust based on your data)
    end_date = model_data['end_date']            # Analysis end date

    # Export the model summary to an HTML file
    mmm_summarizer.output_model_results_summary('summary_output.html', file_path, start_date, end_date)

    print("Process Finished! Summary output file is created.")

except Exception as e:
    print(f"Unexpected error: {e}")
