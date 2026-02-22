<h1>📘 Wiki: Using Windsor.ai Data with Google Meridian</h1>

</br>

🔍 **Overview**

Google Meridian is a Bayesian Marketing Mix Modeling (MMM) library developed by Google. It helps quantify how much different advertising channels contribute to business outcomes like conversions or revenue. This is done through probabilistic modeling that estimates ROI and performance attribution.

Windsor.ai makes it easy to fetch marketing performance data from platforms like Google Ads, Facebook, and others. It saves time by eliminating the need to manually consolidate data, making it ideal for feeding into Meridian for analysis.

In our guide, we use ads data (Google, Facebook, Reddit etc) from Windsor.ai where:
- The KPI is Conversions/Revenue
- We model how Spend influences our KPI over time as per our media channels

🐍 **Installation Requirements**

To use the google-meridian package with your data:

✅ Python Version

Make sure you are using Python 3.11 or higher.

📆 Required Libraries

Required Python libraries:
```python
google-meridian tensorflow tensorflow-probability pandas requests psutil
```

✅ How To Run The Code

Install the required libraries using the given requirements.txt file.
```python
pip install -r requirements.txt
```
Run Model.py.
```python
python model.py
```

🔁 **Data Integration Summary**

To use Windsor.ai data with Google Meridian:

1. Fetch the Data: Use Windsor.ai's API to pull ad performance data with fields like date, spend, impressions, clicks, conversions and revenue etc.

2. Format the Data: Aggregate weekly performance metrics, fill missing values, and structure it using the function getDataFromWindsor() in data.py file that returns processed_csv_data, kpi, media, media spend, mapping of media to channels and mapping of media spend to channels, start data and end data of the data.

3. Configure the Model: Define ROI priors based on your data requirements.

4. Run the Model: Sample from prior and posterior distributions to fit the Bayesian model.

5. Summarize Output: Generate a report that provides ROI estimates and media channel performance.

📌 **Reference**

📚 Google Meridian GitHub: https://github.com/google/meridian

📧 Contact: support@windsor.ai
