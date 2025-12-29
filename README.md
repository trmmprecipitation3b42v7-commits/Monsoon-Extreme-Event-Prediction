g4.areaAvgTimeSeries.TRMM_3B42RT_Daily_7_precipitation.2000030120191231.83E_20N_86E_23N.nc
Source: TRMM 3B42RT Daily precipitation data
•	Time range: March 1, 2000 to December 31, 2019
•	Region: Area average over 83°E-86°E and 20°N-23°N (likely a region in South Asia)
•	Operation: Area-averaged time series
•	D:\Ambika\IIT madras\python\ReFine\ReFine-1\ARIMATemperature_Forecasting\trmm_trial\ncfile_trmm.ipynb: 
Basic time series plot, Monthly climatology, Annual cycle with interannual variability, climatology, seasonal decomposition
•	Time Series Plot: Shows the full daily precipitation record
•	Monthly Climatology: Average precipitation for each month across all years
•	Annual Cycle: Shows each year's pattern plus the overall climatology
•	Basic Statistics: Calculates key precipitation metrics
•	Seasonal Decomposition (optional): Separates trend, seasonality, and residuals
•	SPATIAL CLUSTERING
•	To perform spatial clustering of TRMM precipitation data, you'll need gridded (not area-averaged) data. Your current file (g4.areaAvgTimeSeries...nc) contains only time series data for one region, so you'll need to download the full spatial dataset.
•	Required Variables for Spatial Clustering
o	Precipitation data (3D: time × lat × lon)
o	Latitude coordinate vector
o	Longitude coordinate vector
o	Time coordinate vector
•	Optional but Useful:
o	Quality flags (if available)
o	Elevation data (can help explain spatial patterns)
•	Downloading the Proper TRMM Data
•	Option 1: Download from NASA GES DISC
o	https://disc.gsfc.nasa.gov/datasets/TRMM_3B42_Daily_7/summary
•	Select parameters:
o	Product: TRMM_3B42_Daily (or 3B42RT for real-time)
o	Date range
o	Spatial domain (or global)
o	Download options: NetCDF format
o	Daily or monthly resolution
o	Full spatial grid (not area-averaged)
•	ERROR: SITE UNDER MAINTENANCE
•	A good page on what IMERG algorithm does: https://climatedataguide.ucar.edu/climate-data/gpm-global-precipitation-measurement-mission
•	TO download .nc files from multiple urls in a text file: https://medium.com/@xhl272703370/tutorial-on-how-to-download-multiple-earthdata-urls-78c96df4c1c7
•	 
•	subset_GPM_3IMERGDF_07_20250516_184632_
•	IMERG Final Run Daily V07B (GPM IMERG Final Precipitation L3 1 day 0.1 degree x 0.1 degree V07 (GPM_3IMERGDF)) data file is excellent for spatial clustering analysis. Here's why and how to use it:
•	________________________________________
•	Why This Data is Good for Spatial Clustering
o	High Spatial Resolution: 0.1° × 0.1° grid (~11 km at equator) - Fine enough to capture regional precipitation patterns. provides precipitation estimates on a uniform global grid of 0.1° latitude × 0.1° longitude. Each grid cell represents the estimated precipitation for that specific spatial box (~11 km at the equator). It preserves the spatial variability across the domain.
o	Temporal Resolution: Daily data balances noise reduction while preserving spatial patterns (better than half-hourly for clustering).
o	Data Quality: Final Run (V07B) = Gauge-corrected, research-grade quality.
o	Uses precipitationCal variable (most accurate rainfall estimate).
•	Global Coverage:60°N-60°S (includes most land areas where clustering is useful).
Attribute	Description
Product Name	GPM_3IMERGDF
Spatial Resolution	0.1° × 0.1° (~11 km grid)
Temporal Resolution	Daily (1 day average)
Coverage	Global (60°N–60°S)
Data Type	Gridded
Units	mm/day (millimeters per day)
Processing Level	Level 3 (Merged, gap-filled, validated)

•	Key Variables for Clustering
o	Variable	               Description	              Unit	      Use for Clustering?
o	precipitationCal	Bias-corrected precipitation	mm/h	✅ Primary variable
o	precipitationUncal	Satellite-only estimate	mm/h	❌ Less accurate
o	probabilityLiquidPrecipitation	Rain likelihood	%	❌ Auxiliary
o	qualityIndex	Data quality (0-1)	unitless	✅ Filter low-quality data
•	Why spatial clustering:
o	https://kazumatsuda.medium.com/spatial-clustering-fa2ea5d035a3
•	How single .nc4 file looks ike:
First file: 3B-DAY.MS.MRG.3IMERG.19980101-S000000-E235959.V07B.nc4.nc4
<xarray.Dataset>
Dimensions:                         (time: 1, lon: 3600, lat: 1800, nv: 2)
Coordinates:
  * lat                             (lat) float64 -89.95 -89.85 ... 89.85 89.95
  * lon                             (lon) float32 -179.9 -179.9 ... 179.9 179.9
  * nv                              (nv) float32 0.0 1.0
  * time                            (time) datetime64[ns] 1998-01-01
Data variables:
    MWprecipitation                 (time, lon, lat) float32 ...
    precipitation_cnt_cond          (time, lon, lat) int16 ...
    precipitation                   (time, lon, lat) float32 ...
    MWprecipitation_cnt             (time, lon, lat) int16 ...
    MWprecipitation_cnt_cond        (time, lon, lat) int16 ...
    probabilityLiquidPrecipitation  (time, lon, lat) int16 ...
    randomError                     (time, lon, lat) float32 ...
    randomError_cnt                 (time, lon, lat) int16 ...
    time_bnds                       (time, nv) datetime64[ns] ...
    precipitation_cnt               (time, lon, lat) int16 ...
Attributes:
    BeginDate:       1998-01-01
    BeginTime:       00:00:00.000Z
    EndDate:         1998-01-01
    EndTime:         23:59:59.999Z
    FileHeader:      StartGranuleDateTime=1998-01-01T00:00:00.000Z;\nStopGran...
...
    DOI:             10.5067/GPM/IMERGDF/DAY/07
    ProductionTime:  2024-11-06T16:23:03.571Z
    history:         2025-05-16 18:40:10 GMT hyrax-1.17.1 https://gpm1.gesdis...
    history_json:    [{"$schema":"https:\/\/harmony.earthdata.nasa.gov\/schem...
	
TRMM data is a single .nc file with precipitation values alone. That suggests the TRMM data might be area-averaged over South Asia/India, so it doesn't have spatial dimensions. On the other hand, GPM IMERG data has multiple files with precipitation, lat, and lon variables, meaning it's a grid covering a larger area, and the user needs to subset it to the region of interest (South Asia/India).

17-june-2025

a step-by-step breakdown of the entire process described in your Python code D:\Ambika\IITmadras\python\ReFine\ReFine-1\ARIMA-Temperature_Forecasting\trmm_trial\db4waveletdecomp.ipynb presented as bullet points:
________________________________________
Step-by-Step Procedure of the Extreme Precipitation Prediction Code
•	1. Initial Setup & Data Loading
o	Import necessary libraries (NumPy, Pandas, PyWavelets, Matplotlib, Scikit-learn, TensorFlow/Keras).
o	Set Matplotlib plotting style and figure size for consistent visualization.
o	Define a load_trmm_data function: 
	Reads the raw (3-hourly) TRMM precipitation data from a specified CSV file.
	Assigns column names ('Date', 'Precip').
	Converts 'Date' column to datetime objects and 'Precip' to numeric, handling potential errors.
	Filters out invalid precipitation values (e.g., -999).
	Sets the 'Date' column as the DataFrame index.
	Crucial Resampling Step: Resamples the 3-hourly precipitation data to daily totals (or maximums, if desired) using resample('D').sum().
	Drops any rows with missing values that might arise from resampling.
	Returns the preprocessed daily precipitation DataFrame.
o	Loads the TRMM data using the load_trmm_data function, resulting in a daily trmm_df.
o	Extracts the 'Precip' column into precip_series for wavelet operations.
•	2. Wavelet Decomposition
o	Defines a wavelet_decomposition_extremes function: 
	Performs a multi-level Discrete Wavelet Transform (DWT) on the precip_series using pywt.wavedec (e.g., db4 wavelet at level=6). This breaks down the signal into: 
	An Approximation (A) coefficient: Represents the smoothed, low-frequency trend of the data.
	Several Detail (D) coefficients: Represent high-frequency fluctuations at different scales (e.g., D1 for finest details, D6 for coarsest details).
	Dynamically generates labels for each component (e.g., A6, D6, D5, ..., D1).
	Visualizes each of these coefficient series in separate subplots to show their amplitude and patterns.
o	Calls this function to obtain the wavelet coefficients (coeffs).
•	3. Extreme-Preserving Reconstruction
o	Defines a reconstruct_extremes function: 
	Calculates an adaptive threshold (based on Donoho-Johnstone method) from the finest detail coefficient.
	Applies soft thresholding to all detail coefficients (coeffs[1:]) using pywt.threshold(). This process aims to reduce noise and emphasize significant signal changes (like extreme precipitation). The approximation coefficient is left untouched.
	Performs an inverse wavelet transform using pywt.waverec() to reconstruct the precipitation signal from the (thresholded) coefficients.
	Trims the reconstructed signal to match the original length.
	Visualizes the original precipitation series against this reconstructed (denoised/extreme-preserved) series to show the effect of the filtering.
o	Executes this reconstruction and stores the result in a new column Precip_Reconstructed in trmm_df.
•	4. Feature Engineering
o	Defines a create_extreme_features function: 
	Performs another wavelet decomposition on the original 'Precip' series to get coefficients.
	Wavelet Features: Reconstructs each individual approximation component (A_level) and detail component (D_i) back to the original daily length using pywt.upcoef(). Each of these upsampled components becomes a separate feature column in a new DataFrame (e.g., 'A6', 'D6', 'D5', ..., 'D1'). These capture various frequency characteristics of the precipitation signal.
	Temporal Features: Adds features derived from the date index: dayofyear, sin_doy, and cos_doy to capture annual seasonality.
	Lagged Features: Creates lagged versions of the original 'Precip' column (e.g., 'lag_Precip_1D' for yesterday's precip, 'lag_Precip_7D' for a week ago, etc.). These capture temporal dependencies.
	Drops rows containing NaN values (primarily due to the lags).
	Returns the DataFrame containing all engineered features.
o	Generates the trmm_features DataFrame using this function.
•	5. Data Preparation for LSTM
o	Identifies the target variable (y, which is 'Precip') and the features (X, all other columns in trmm_features).
o	Scales both the features (X) and the target (y) to a range between 0 and 1 using MinMaxScaler. The scalers are saved for later inverse transformation.
o	Defines a create_sequences function: 
	Transforms the scaled data into the (samples, timesteps, features) 3D format required by LSTM models, using a sliding window approach (n_steps_in for input sequence length, n_steps_out for prediction horizon).
o	Creates the LSTM input X_seq and target y_seq sequences.
o	Performs a chronological train-test split (e.g., 80% for training, 20% for testing) to maintain the time series order.
•	6. LSTM Model Definition and Training
o	Defines a Keras Sequential model: 
	Includes multiple LSTM layers (e.g., with 100 units, relu activation), designed to learn long-term dependencies in the sequences.
	Applies Dropout layers to prevent overfitting.
	Adds a Dense output layer to produce the predictions.
o	Compiles the model with the adam optimizer and mse (Mean Squared Error) as the loss function.
o	Sets up an EarlyStopping callback to monitor validation loss and stop training if it doesn't improve for a certain number of epochs, restoring the best model weights.
o	Trains the LSTM model using the model.fit() method on the training data, with a validation split and the early_stopping callback.
o	Plots the training and validation loss history over epochs.
•	7. Prediction and Evaluation
o	Uses the trained model to make predictions (model.predict()) on the unseen test set.
o	Inverse transforms both the scaled actual test values (y_test) and the scaled predicted values (y_pred_scaled) back to their original precipitation units (mm/day) using the saved MinMaxScaler objects.
o	Calculates and prints performance metrics: Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) between the actual and predicted precipitation values on the test set.
o	Generates plots to visualize the Actual vs. Predicted Precipitation for the test set, allowing for a visual assessment of model performance.
o	Includes an additional plot focusing on a smaller segment of the test data for clearer visualization.
