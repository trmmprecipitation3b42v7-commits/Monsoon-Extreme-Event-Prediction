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
Sources 



The LSTM (Long Short-Term Memory) procedure in your code involves several crucial steps, from preparing the data to training the model and evaluating its performance. LSTMs are a type of Recurrent Neural Network (RNN) particularly well-suited for sequence prediction tasks like time series forecasting because they can learn from and remember patterns over long sequences of data.
Here's a step-by-step breakdown of the LSTM procedure in your code:
________________________________________
LSTM Procedure: Step-by-Step
1. Data Preparation for LSTM
•	Identification of Features (X) and Target (y):
o	Your trmm_features DataFrame, which contains the original daily precipitation, engineered wavelet features (A, D components), temporal features (day of year, sin/cos of day), and lagged precipitation values, is prepared.
o	The target_column is set to 'Precip' (your daily precipitation values).
o	X (features) are all columns in trmm_features except the target column.
o	y (target) is the 'Precip' column itself.
•	Data Scaling (Normalization):
o	Why? Neural networks, especially LSTMs, perform better and train faster when input data is scaled to a smaller range (e.g., 0 to 1). This prevents larger input values from dominating the learning process and helps with gradient stability.
o	MinMaxScaler(feature_range=(0, 1)): Both features (X) and the target (y) are scaled using this scaler. This transforms values linearly so that the minimum maps to 0 and the maximum maps to 1.
o	Importance: scaler_X and scaler_y objects are fit_transformed and stored. These specific scaler objects must be used later for inverse_transform to convert predictions back to the original units.
•	Sequence Creation (Sliding Window):
o	Why? LSTMs require input data in a 3D format: (samples, timesteps, features). A single 'sample' for an LSTM is a sequence of observations (timesteps), each having multiple features.
o	create_sequences function: This function implements a sliding window (or look-back) approach. 
	n_steps_in: Defines how many past time steps (days, in your case) the LSTM should "look back" to make a prediction. For example, if n_steps_in = 30, the model uses the previous 30 days of features to make a prediction.
	n_steps_out: Defines how many future time steps (days) the LSTM should predict. For your code, n_steps_out = 1, meaning it predicts one day ahead.
	The function iterates through your scaled data, creating input sequences (X_seq) and corresponding target sequences (y_seq).
o	Output Shape: X_seq will have a shape like (number_of_samples, n_steps_in, number_of_features). y_seq will have (number_of_samples, n_steps_out, number_of_targets) (which is (samples, 1, 1) in your case).
•	Train-Test Split (Chronological):
o	Why? For time series data, it's crucial to maintain the temporal order. You cannot randomly shuffle data for training and testing.
o	Your code performs a chronological split, where the first 80% of the X_seq and y_seq data is used for training (X_train, y_train), and the last 20% is reserved for testing (X_test, y_test). This ensures the model is evaluated on future, unseen data.
2. LSTM Model Architecture
•	Sequential Model:
o	model = Sequential(): This is the simplest type of Keras model, a linear stack of layers.
•	LSTM Layers:
o	model.add(LSTM(units=100, activation='relu', return_sequences=True, input_shape=(n_steps_in, X_seq.shape[2]))): 
	LSTM(units=100): Defines an LSTM layer with 100 memory units (neurons). More units generally mean more learning capacity.
	activation='relu': Applies the Rectified Linear Unit activation function to the output of the LSTM gates.
	return_sequences=True: Important! This tells the LSTM layer to output a sequence (one output for each timestep in the input sequence) rather than just a single output at the end. This is necessary when stacking multiple LSTM layers, as the next LSTM layer expects a sequence input.
	input_shape=(n_steps_in, X_seq.shape[2]): This specifies the shape of the input sequences (timesteps, number of features per timestep) that this first LSTM layer will receive.
o	model.add(Dropout(0.2)): 
	Why? Dropout is a regularization technique. It randomly sets a fraction (here, 20%) of the input units to 0 at each update during training. This helps prevent overfitting by forcing the network to learn more robust features and preventing co-adaptation of neurons.
o	model.add(LSTM(units=100, activation='relu')): 
	Another LSTM layer. Notice return_sequences is not specified, so it defaults to False. This means this layer outputs only the last hidden state of the sequence, which is then fed to the dense layer for prediction. This is typical for a many-to-one or many-to-few sequence prediction.
•	Output Layer (Dense):
o	model.add(Dense(units=n_steps_out)): This is a standard densely connected neural network layer. It takes the output from the final LSTM layer and maps it to the desired output shape. Since n_steps_out is 1 (predicting one day ahead) and you're predicting a single value (precipitation), units=1 is correctly implied.
•	Model Compilation:
o	model.compile(optimizer='adam', loss='mse'): 
	optimizer='adam': An adaptive learning rate optimization algorithm. It's generally a good default choice for many deep learning tasks.
	loss='mse': Mean Squared Error. This is a common loss function for regression problems, aiming to minimize the average squared difference between actual and predicted values.
3. Model Training
•	Early Stopping:
o	early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True): This is a callback that monitors a specific metric (here, val_loss or validation loss) during training. 
	patience=10: If the val_loss doesn't improve for 10 consecutive epochs, training will stop early.
	restore_best_weights=True: Ensures that the model weights from the epoch with the best val_loss are restored at the end of training, rather than the weights from the very last epoch.
o	Why? Prevents overfitting and saves computational time by stopping training once the model's performance on unseen data (validation set) starts to degrade.
•	Fitting the Model:
o	history = model.fit(...): This is where the actual learning happens. 
	X_train, y_train: The scaled training data sequences.
	epochs=50: The maximum number of times the model will iterate over the entire training dataset. Early stopping will likely halt this before 50 if convergence is met.
	batch_size=32: The number of samples processed before the model's weights are updated.
	validation_split=0.2: A portion (20%) of the training data is automatically set aside by Keras to be used as a validation set. The model does not learn from this data; it's used to monitor performance during training to detect overfitting.
	callbacks=[early_stopping]: The early stopping mechanism is activated.
o	The history object stores the loss and validation loss for each epoch, which is then plotted to visualize training progress.
4. Prediction and Evaluation
•	Making Predictions:
o	y_pred_scaled = model.predict(X_test): The trained model generates predictions on the X_test (unseen test features) data. These predictions are still in the scaled (0-1) format.
•	Inverse Transformation:
o	y_test_original = scaler_y.inverse_transform(y_test.reshape(-1, n_steps_out)): Converts the actual y_test values back from their scaled (0-1) range to the original precipitation units (mm/day).
o	y_pred_original = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, n_steps_out)): Converts the model's y_pred_scaled predictions back to the original precipitation units (mm/day).
o	Why? This is crucial for interpreting the predictions in meaningful physical units and for accurate evaluation.
•	Evaluation Metrics:
o	mae = mean_absolute_error(y_test_original, y_pred_original): Calculates the Mean Absolute Error. This tells you, on average, how much your predictions deviate from the actual values, in the original units (mm/day).
o	rmse = np.sqrt(mean_squared_error(y_test_original, y_pred_original)): Calculates the Root Mean Squared Error. This metric also provides error in the original units but penalizes larger errors more heavily than MAE.
o	These metrics quantify the model's performance on the unseen test data.
•	Visualization of Results:
o	The code generates plots comparing the Actual Precipitation and Predicted Precipitation over the test period. This visual comparison is vital for understanding the model's fit, its ability to capture peaks, and its overall performance trends. A smaller segment is also plotted for closer inspection.
This comprehensive procedure ensures that the LSTM model is properly prepared, trained, and evaluated for time series forecasting, leveraging the engineered wavelet and temporal features to capture complex patterns in extreme precipitation data.






18-june-2025
from the previous python codes we created for extreme precipitation prediction, How do we improve forecasts at 3-10 day time horizon for different regions in India? should we Train separate models region wise or any other method?
Response:
Improving extreme precipitation forecasts at the 3-10 day time horizon for different regions in India is a complex but crucial task. Given the diverse topography and climatic influences across India (e.g., Himalayas, coastal regions, peninsular India, Indo-Gangetic plains), a "one-size-fits-all" approach may not be optimal. Here's a breakdown of strategies, including whether to train separate models region-wise, and other methods:
1. Training Separate Models Region-Wise (Hybrid Approach)
Pros of Region-Wise Models:
•	Captures Local Nuances: Different regions in India experience extreme precipitation due to distinct atmospheric phenomena (e.g., Western Disturbances in North India, Monsoonal depressions and cyclones in coastal areas, orographic rainfall in Western Ghats). Region-specific models can better learn these localized patterns and their drivers.
•	Tailored Feature Engineering: You can select and engineer features (e.g., specific atmospheric indices, local topography, land-use) that are most relevant to extreme precipitation in that particular region.
•	Better Performance for Specific Events: Models trained on regional data are likely to outperform a pan-India model in predicting specific, localized extreme events, which are often the most impactful.
•	Manageable Data Volume: For very high-resolution models, breaking down India into regions can make the data volume more manageable for training.
Cons of Region-Wise Models:
•	Increased Development and Maintenance Overhead: You'll need to develop, train, and maintain multiple models, which can be resource-intensive.
•	Data Scarcity in Some Regions: Some regions might have less dense observational networks, leading to data scarcity challenges for training robust models.
•	Difficulty in Capturing Large-Scale Interactions: Extreme events can sometimes be influenced by large-scale atmospheric teleconnections that might be missed by purely regional models.
Recommendation: A hybrid approach is often the most effective.
•	Start with a robust pan-India model: This provides a baseline and captures the overarching large-scale atmospheric patterns influencing precipitation across the country.
•	Develop region-specific "refinement" or "downscaling" models: These models would take the output of the pan-India model and further process it, incorporating regional data and localized characteristics, to improve the forecast at a finer resolution and for specific extreme events. This could involve techniques like statistical or dynamical downscaling.
2. Other Methods to Improve Forecasts (Regardless of Region-Wise or Pan-India)
A. Data and Feature Engineering:
•	High-Resolution Data: Utilize the highest possible resolution for observational data (rainfall, temperature, humidity, wind) and Numerical Weather Prediction (NWP) model outputs. India has recently launched the Bharat Forecasting System (BFS), a high-resolution weather model operating on a 6-kilometer grid, which is a significant step in this direction. Leveraging this kind of output is crucial.
•	More Predictor Variables: Beyond standard meteorological parameters, consider: 
o	Atmospheric Rivers (ARs): Especially relevant for heavy rainfall in certain regions. Looking at integrated vapor transport (IVT) can be more skillful than just precipitation at longer lead times.
o	Madden-Julian Oscillation (MJO) and other Intraseasonal Oscillations (ISOs): These large-scale drivers significantly influence monsoon variability and extreme events.
o	Sea Surface Temperatures (SSTs) and their gradients: Especially in the Arabian Sea and Bay of Bengal, as they can modulate moisture supply.
o	Topography and Orography: Essential for regions like the Western Ghats and Himalayas.
o	Soil Moisture: Can influence local convective activity.
o	Vegetation Indices: Can indicate land surface interactions.
•	Ensemble Forecasting: Instead of relying on a single deterministic forecast, use ensemble predictions from NWP models (like IMD's Global Ensemble Forecasting System - GEFS, or the new Bharat Forecasting System). Machine learning models can then be trained to predict probabilities of extreme events from these ensembles. 
o	Bias Correction of Ensembles: NWP models often have biases. Applying machine learning-based bias correction techniques (e.g., linear regression, MLPs, U-Nets, or ensemble approaches like "BOS" - Bagging of Statistical models) to the raw ensemble outputs can significantly improve accuracy.
•	Lagged Predictors: Incorporate past values of relevant meteorological variables as predictors, as atmospheric processes have memory.
•	Advanced Feature Engineering: Use techniques like principal component analysis (PCA) or autoencoders to extract salient features from high-dimensional meteorological data.
B. Model Architecture and Techniques:
•	Deep Learning Models: 
o	Long Short-Term Memory (LSTM) Networks: Excellent for capturing temporal dependencies in sequential data, which is crucial for forecasting.
o	Convolutional Neural Networks (CNNs): Ideal for spatial feature extraction, especially when dealing with gridded data (like rainfall maps).
o	Transformer Networks: Increasingly used in sequence modeling and could be powerful for weather data.
o	Graph Neural Networks (GNNs): If you can represent your weather stations or grid points as a graph, GNNs can capture spatial relationships effectively.
•	Hybrid Models: Combine physics-based models with machine learning. For example, use NWP model outputs as primary inputs to a machine learning model, which then learns to correct biases or refine predictions based on historical observations.
•	Dynamical Downscaling: This involves running high-resolution regional climate models (RCMs) nested within coarser-resolution global models (GCMs). The Weather Research and Forecasting (WRF) model is a common choice. This is computationally expensive but can provide physically consistent, high-resolution forecasts. 
o	Bias-correction of GCM boundary conditions: Before feeding GCM outputs to RCMs, bias-correcting them can significantly improve the RCM's performance.
•	Statistical Downscaling: Develop statistical relationships between large-scale atmospheric variables (from GCMs/NWP models) and local-scale precipitation. This is computationally cheaper than dynamical downscaling. Machine learning models (like ConvLSTMs) are very effective for this.
•	Ensemble Machine Learning: Instead of just using ensemble forecasts from NWP models, combine predictions from multiple machine learning models (e.g., boosting, bagging, stacking) to improve robustness and reduce uncertainty.
C. Addressing the 3-10 Day Horizon (Sub-seasonal Forecasting):
•	Focus on Sub-seasonal Drivers: The 3-10 day range falls into the sub-seasonal category. This time scale is particularly challenging as it bridges the gap between short-range weather and seasonal climate. It's often influenced by a complex interplay of large-scale atmospheric oscillations (like MJO, as mentioned), teleconnections, and regional processes.
•	Verification Metrics: Beyond standard accuracy metrics, use metrics specifically designed for extreme events (e.g., Probability of Detection, False Alarm Ratio, Equitable Threat Score, Critical Success Index). Focus on reducing "time shifts" in extreme event predictions.
•	Understanding Model Errors: Analyze where and why your previous models failed. For instance, the IIT-Bombay study highlighted that models sometimes predict extreme rains 1-2 days before or after the actual event. Statistical post-processing methods can help correct these temporal biases.
In summary:
For extreme precipitation prediction at the 3-10 day horizon in India, a multi-pronged approach is recommended:
1.	Leverage high-resolution pan-India NWP outputs (like the Bharat Forecasting System) as your foundation.
2.	Employ a hybrid modeling strategy: 
o	Start with pan-India machine learning models that learn the general patterns.
o	Then, apply region-specific refinement techniques. This could involve: 
	Separate machine learning models for specific homogeneous climate zones (as defined by IMD, e.g., North-West India, Peninsular India, East & North East India, Central India). These regional models would ingest the pan-India forecast outputs along with local data and features.
	Dynamical downscaling using RCMs for specific, highly vulnerable regions (like the Himalayas or Western Ghats) where complex terrain plays a dominant role.
	Statistical downscaling with advanced ML techniques (e.g., ConvLSTMs) to generate very high-resolution forecasts.
3.	Prioritize robust ensemble forecasting and bias correction at all stages.
4.	Continuously incorporate new and diverse predictor variables that are known to influence extreme precipitation in the Indian context.
5.	Utilize advanced deep learning architectures capable of handling spatio-temporal data and complex non-linear relationships.
By combining the strengths of large-scale modeling with the specificity of regional approaches and sophisticated machine learning techniques, you can significantly improve extreme precipitation forecasts for India at the challenging 3-10 day horizon.

Comparing different wavelets for extreme precip prediction
	Model	Filter	Level	MAE	RMSE
	Baseline LSTM	-	-	0.3537	0.499
Haar	Wavelet LSTM	haar	6	0.2368	0.3385
Daubechies		Db2	6	0.2535	0.3481
		Db3		0.2361	0.3294
		Db4			
		Db5		0.2232	0.3110
		Db6		0.2117	0.3088
		Db7		0.2221	0.3124
Symlets		Sym4		0.2241	0.3191
					
Coiflets		Coif1		0.2260	0.3255

  
 
 
 
 
 











Wavelets play a significant role in improving extreme precipitation prediction using satellite data like IMERG and TRMM by addressing inherent challenges in these datasets and the nature of precipitation itself. Here's a breakdown of their contributions:
Understanding IMERG and TRMM Data:
•	TRMM (Tropical Rainfall Measuring Mission): Operated from 1997 to 2015, primarily focused on tropical and subtropical regions. It provided valuable insights into rainfall patterns. Your daily data from TRMM are likely derived from its 3-hourly 0.25-degree products (like 3B42).
•	IMERG (Integrated Multi-satellitE Retrievals for Global Precipitation Measurement): The successor to TRMM, part of the GPM mission (Global Precipitation Measurement). It provides higher spatial (0.1 degree) and temporal (30-minute) resolution, extending coverage to higher latitudes. IMERG has different "runs" (Early, Late, Final) based on latency and accuracy.
•	Characteristics of both: 
o	Non-stationarity: Precipitation time series are highly non-stationary, meaning their statistical properties (mean, variance) change over time due to seasonal cycles, climate variability (ENSO, IOD), and chaotic atmospheric processes.
o	Multi-scale variability: Precipitation occurs across a wide range of scales, from localized convective storms (short duration, small spatial extent) to large-scale frontal systems (longer duration, wider extent). Extreme events are often characterized by high-frequency, localized bursts.
o	Intermittency and Extremes: Precipitation, especially extreme precipitation, is intermittent (occurs irregularly) and characterized by sharp changes and sudden spikes. These extreme values are crucial for flood forecasting and disaster management.
o	Noise and Errors: Satellite-derived precipitation data, while invaluable, contain noise and uncertainties due to indirect measurements, sensor limitations, and retrieval algorithms.
Wavelets' Role in Extreme Precipitation Prediction:
Wavelet Transform (WT) is a powerful tool for analyzing non-stationary signals and signals with multi-scale characteristics. Here's how it helps in extreme precipitation prediction with IMERG and TRMM data:
1.	Multi-resolution Decomposition:
o	Wavelets decompose a time series into different frequency components (or "scales") and their corresponding time locations.
o	Approximation Coefficients (A): Represent the low-frequency, smooth, long-term trends (e.g., seasonal cycles, multi-year oscillations). These can reveal the underlying climatic drivers.
o	Detail Coefficients (D): Capture the high-frequency, transient, irregular variations (e.g., daily fluctuations, extreme events, sudden changes).
o	Role in Prediction: By separating these components, you can analyze and model each scale independently. For example: 
	The low-frequency components can capture the general wet/dry periods.
	The high-frequency components are critical for identifying and predicting extreme events and sudden bursts of rainfall. This allows the model to focus on the characteristics of extreme values without being overwhelmed by the smooth daily variations.
2.	Noise Reduction and Feature Extraction:
o	Minor noise or irrelevant fluctuations often reside in the very high-frequency detail coefficients. Wavelets allow for denoising by thresholding these coefficients (as shown in the reconstruct_extremes_visualisation function in the provided code). This can clean up the input data, leading to more robust predictions.
o	By using the specific detail coefficients associated with extreme events as input features, the model can learn patterns related to these events more effectively. For example, a sudden spike in a D1 or D2 coefficient might strongly indicate an impending extreme precipitation event.
3.	Handling Non-stationarity:
o	Unlike Fourier Transform (which assumes stationarity), wavelets are inherently suited for non-stationary signals. They provide both time and frequency localization, meaning they tell you what frequencies are present and when they occur. This is vital for precipitation, where patterns can change dynamically.
4.	Capturing Localized Events (Extremes):
o	Extreme precipitation events are often localized in time. Wavelets, with their ability to capture localized features, are excellent at highlighting these abrupt changes and sudden peaks. By focusing on the characteristics of these "spikes" in the detail coefficients, the LSTM can learn to predict them.
5.	Improved Model Performance (Hybrid Models):
o	Many studies show that combining wavelet decomposition with machine learning models (like LSTMs, ANNs, SVMs) often leads to superior prediction accuracy compared to using the models alone on raw data.
o	The reason is that wavelets act as a powerful feature extractor and data pre-processor. They transform the complex, non-stationary precipitation signal into a set of more stationary, interpretable sub-signals. The LSTM can then learn the relationships within and between these "purified" components more effectively.
o	For extreme precipitation specifically, this decomposition helps the model to discern the distinct drivers and patterns associated with rare, high-impact events from the more common, background precipitation.
In the context of your code:
The create_all_features function using pywt.wavedec and pywt.upcoef precisely implements the idea of using wavelet coefficients as features. By including A_level (approximation, capturing overall trends) and D1 to D_level (details, capturing fluctuations and extremes) as input to your LSTM, you are directly leveraging the multi-resolution capabilities of wavelets to enhance your prediction model's ability to forecast precipitation, including its extreme aspects. The comparison across different Daubechies wavelets allows you to find which specific wavelet's characteristics (e.g., smoothness, number of vanishing moments) best capture the relevant patterns for your precipitation data and prediction task.


Literature review Extreme event prediction problem:
Tabular or graphical
Table: Comparison of Precipitation Forecasting Model Categories
Model	Key chara	Strengths	Limitations	Typical Spatial/Temporal Resolution
Numerical Weather Prediction (NWP)	Physics-based, atmospheric models	Spatially coherent, physically consistent	High computational cost, limited sub-grid resolution, sensitivity to initial conditions, struggles with convection.	Variable (e.g., 1-100 km, hourly)
Statistical Models (e.g., ARIMA, Regression)	Data-driven, linear/simple non-linear relationships	Computationally efficient, interpretable	Fail to capture complex non-linearity, non-stationarity, limited for rare events.	Varies (e.g., daily, monthly)
Traditional Machine Learning (e.g., SVM, RF)	Data-driven, non-linear pattern recognition	Handles non-linearity, good for complex relationships	Can struggle with long sequences, class imbalance for extremes, limited interpretability.	Varies (e.g., daily, monthly)
	ARIMA			
Deep Learning (e.g., LSTMs)	Data-driven, learns complex temporal dependencies	Excels in sequence modeling, handles non-linearity	Vulnerable to class imbalance (under-predicts rare events), sensitive to noisy inputs.	Varies (e.g., daily, sub-daily)
Hybrid Wavelet-LSTM (Present Study)	Integrates multi-scale analysis with deep learning	Addresses noise, non-linearity, and class imbalance (via weighted loss), enhances extreme event focus.	Current focus on area-averaged data (future work will address gridded).	Area-averaged, daily

Table2: Summary of Recent Studies on Extreme Precipitation Forecasting (with ML/DL) in the Indian Context
Study (Year)	method	variables	Data, Region	Metrics
Sojitra et al. (2015)
Sojitra MA, Purohit RC, Pandya PA (2015) Comparative study of
ANFIS method	Wet bulb temperature,
mean temperature, relative
humidity, evaporation,
previous moving
average week	Daily, India	R, Mean square error
(MSE), Normalized
mean square error
(NMSE), percent error
(% error), volumetric
error (EV), coefficient
efficiency (CE)
Venkata Ramana et al.
(2013)
Venkata Ramana R, Krishna B, Kumar SR, Pandey NG (2013) Monthly
WNN	Monthly rainfall, minimum
temperature, maximum
temperature	Monthly India	RMSE, R, CE
WNN (RMSE:63.010,
R:0.974, CE:94.780) more
effective than single ANN
(RMSE:163.790, R:0.807,
CE:64.730) and AR
(RMSE:221.820, R:0.642,
				
				
Zhang, Q., Wang, H., & Singh, V. P. et. al., Journal of hydrology, 2021	Wavelet-LSTM for Precipitation Prediction	Validates wavelet decomposition (DWT) + LSTM for rainfall prediction		wavelet (A2/D2 components) were shown to improve LSTM accuracy
Kratzert, F., Klotz, D., Shalev, G., et al., Water Resources Research (2022]
	Attention Mechanisms			Demonstrates self-attention outperforms LSTM
[Li, X., Chen, Y., & Zhang, G., Environmental modelling and software, 2023]
	 LSTM + attention for extremes			
Prakash, S., Mitra, A. K., & Pai, D. S. Et. al., Remote Sensing of Environment 2020]
	TRMM with LSTM	TRMM-Based Deep Learning		
[Mishra, S. K., Shah, R., & Sahany, S et. al., Climate dynamics, 2023].
	wavelet + LSTM for extreme precipitation prediction			
Prakash et al. (2020)
	CNN-LSTM
	TRMM 3B42 (India)
	RMSE: 0.14 (mm/day)
Correlation (R): 0.90
	CNN-LSTM outperformed traditional ARIMA by 35% in RMSE for monsoon extremes
Wang et al. (2021	CNN-LSTM
	TRMM 3B42
	RMSE: 0.12 (mm/day)
Precision (Extremes): 78%
	Model achieved 78% precision for >99th percentile events.

				



Overall Extreme event prediction problem: 
a graphical representation





























 


















Methodology



























Task	Approach	Reference
Thunderstorm Prediction	EEMD-ANN, EEMD-SVM, ARIMA [18] LRCN-CNN, LSTM[19]	@article{azad2021development,
  title={Development of novel hybrid machine learning models for monthly thunderstorm frequency prediction over Bangladesh},
  author={Azad, Md Abul Kalam and Islam, Abu Reza Md Towfiqul and Rahman, Md Siddiqur and Ayen, Kurratul},
  journal={Natural Hazards},
  volume={108},
  number={1},
  pages={1109--1135},
  year={2021},
  publisher={Springer}
}

@article{guastavino2022prediction,
  title={Prediction of severe thunderstorm events with ensemble deep learning and radar data},
  author={Guastavino, Sabrina and Piana, Michele and Tizzi, Marco and Cassola, Federico and Iengo, Antonio and Sacchetti, Davide and Solazzo, Enrico and Benvenuto, Federico},
  journal={Scientific Reports},
  volume={12},
  number={1},
  pages={20049},
  year={2022},
  publisher={Nature Publishing Group UK London}
}


Thunderstorm Severity Prediction	LSTM-FC, CNN-LSTM, ConvLSTM	@article{essa2022deep,
  title={Deep learning prediction of thunderstorm severity using remote sensing weather data},
  author={Essa, Yaseen and Hunt, Hugh GP and Gijben, Morn{\'e} and Ajoodha, Ritesh},
  journal={IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing},
  volume={15},
  pages={4004--4013},
  year={2022},
  publisher={IEEE}
}

Lightning Prediction	RNN [21]
 ResNet[22]	@article{lin2019attention,
  title={Attention-based dual-source spatiotemporal neural network for lightning forecast},
  author={Lin, Tianyang and Li, Qingyong and Geng, Yangli-Ao and Jiang, Lei and Xu, Liangtao and Zheng, Dong and Yao, Wen and Lyu, Weitao and Zhang, Yijun},
  journal={IEEE Access},
  volume={7},
  pages={158296--158307},
  year={2019},
  publisher={IEEE}
}


@article{lu2022monitoring,
  title={Monitoring lightning location based on deep learning combined with multisource spatial data},
  author={Lu, Mingyue and Zhang, Yadong and Chen, Min and Yu, Manzhu and Wang, Menglong},
  journal={Remote Sensing},
  volume={14},
  number={9},
  pages={2200},
  year={2022},
  publisher={MDPI}
}

Lightning Identification	CNN	@article{qian2022lightning,
  title={Lightning identification method based on deep learning},
  author={Qian, Zheng and Wang, Dongdong and Shi, Xiangbo and Yao, Jinliang and Hu, Lijun and Yang, Hao and Ni, Yongsen},
  journal={Atmosphere},
  volume={13},
  number={12},
  pages={2112},
  year={2022},
  publisher={MDPI}
}

Precipitation Forecast	RNN [24]

CNN [25]

LSTM 









wavelet-LSTM
	@article{sarwade2024predictive,
  title={Predictive Modeling of Extreme Weather Forecasting Events: an LSTM Approach.},
  author={Sarwade, Meena P and Shinde, Santhosh A and Patil, Vaishali S},
  journal={Current World Environment},
  volume={19},
  number={1},
  pages={185},
  year={2024},
  publisher={Enviro Research Publishers}
}

@article{tamilselvi2025novel,
  title={Novel wavelet-LSTM approach for time series prediction},
  author={Tamilselvi, C and Paul, Ranjit Kumar and Yeasin, Md and Paul, AK},
  journal={Neural Computing and Applications},
  volume={37},
  number={17},
  pages={10521--10530},
  year={2025},
  publisher={Springer}
}


Extreme precipitation prediction	 LSTM-Self Attention Model
	@inproceedings{10.1145/3640824.3640838,
author = {Luo, Guilan and Cao, Anqi and Ma, Xin and Hu, Anshun and Wang, Caikui},
title = {Prediction of Extreme Precipitation Events Based on LSTM-Self Attention Model},
year = {2024},
isbn = {9798400707971},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3640824.3640838},
doi = {10.1145/3640824.3640838},
booktitle = {Proceedings of the 2024 8th International Conference on Control Engineering and Artificial Intelligence},
pages = {91–97},
numpages = {7},
keywords = {Event prediction, Extreme precipitation, LSTM, LSTM-SelfAttention model, Self-attention mechanism},
location = {Shanghai, China},
series = {CCEAI '24}
}
Quantitative Precipitation Estimation	CNN-based [27]	
Hailstorm Detection 	CNN, DNN [29]	
Hailstorm Forecast 	Atuencoder, CNN [30]	
Storm prediction	LSTM and
XGBoost













Monte Carlo-based LSTM 	@article{frifra2024harnessing,
  title={Harnessing LSTM and XGBoost algorithms for storm prediction},
  author={Frifra, Ayyoub and Maanan, Mohamed and Maanan, Mehdi and Rhinane, Hassan},
  journal={Scientific Reports},
  volume={14},
  number={1},
  pages={11381},
  year={2024},
  publisher={Nature Publishing Group UK London}
}



@article{SATTARI2025126004,
title = {A probabilistic machine learning framework for daily extreme events forecasting},
journal = {Expert Systems with Applications},
volume = {265},
pages = {126004},
year = {2025},
issn = {0957-4174},
doi = {https://doi.org/10.1016/j.eswa.2024.126004},
url = {https://www.sciencedirect.com/science/article/pii/S0957417424028719},
author = {Ali Sattari and Ehsan Foroumandi and Keyhan Gavahi and Hamid Moradkhani},
}
Standardized Precipitation Index (SPI) 


effective drought index (EDI) 
(Drought prediction)	ANFIS, FFNN, MLR [47] 
ARIMA, ANN, SVR, WA-ANN, WA-SVR [48] 
ANN, SVR, WANN [49] ANN, WANN, ARIMA, SARIMA [50] ARIMA, ANN, WANN [51] EMD-DBN [52] 


WP-ANN, WP-SVR [53]

LSTM

WLSTM	@article{anshuka2022spatio,
  title={Spatio temporal hydrological extreme forecasting framework using LSTM deep learning model},
  author={Anshuka, Anshuka and Chandra, Rohitash and Buzacott, Alexander JV and Sanderson, David and van Ogtrop, Floris F},
  journal={Stochastic environmental research and risk assessment},
  volume={36},
  number={10},
  pages={3467--3485},
  year={2022},
  publisher={Springer}
}



@article{ham2023comparison,
  title={Comparison of LSTM network, neural network and support vector regression coupled with wavelet decomposition for drought forecasting in the western area of the DPRK},
  author={Ham, Yong-Sik and Sonu, Kyong-Bok and Paek, Un-Sim and Om, Kum-Chol and Jong, Sang-Il and Jo, Kum-Ryong},
  journal={Natural Hazards},
  volume={116},
  number={2},
  pages={2619--2643},
  year={2023},
  publisher={Springer}
}

3-day ahead hourly water level prediction, California	probability-enhanced neural network model, called NEC+	@article{Li_Xu_Anastasiu_2023, title={An Extreme-Adaptive Time Series Prediction Model Based on Probability-Enhanced LSTM Neural Networks}, volume={37}, url={https://ojs.aaai.org/index.php/AAAI/article/view/26045}, DOI={10.1609/aaai.v37i7.26045}, number={7}, journal={Proceedings of the AAAI Conference on Artificial Intelligence}, author={Li, Yanhong and Xu, Jack and Anastasiu, David C.}, year={2023}, month={Jun.}, pages={8684-8691} }
Traffic anomaly detection	LSTM	@inproceedings{davis2019lstm,
  title={LSTM-based anomaly detection: detection rules from extreme value theory},
  author={Davis, Neema and Raina, Gaurav and Jagannathan, Krishna},
  booktitle={Progress in Artificial Intelligence: 19th EPIA Conference on Artificial Intelligence, EPIA 2019, Vila Real, Portugal, September 3--6, 2019, Proceedings, Part I 19},
  pages={572--583},
  year={2019},
  organization={Springer}
}

Stock market Prediction	Multimodal Event-Driven LSTM














WLSTM
	@ARTICLE{8966989,
  author={Li, Qing and Tan, Jinghua and Wang, Jun and Chen, Hsinchun},
  journal={IEEE Transactions on Knowledge and Data Engineering}, 
  title={A Multimodal Event-Driven LSTM Model for Stock Prediction Using Online News}, 
  year={2021},
  volume={33},
  number={10},
  pages={3323-3337},
  keywords={Companies;Media;Tensors;Stock markets;Predictive models;Solid modeling;Computational modeling;Stock prediction;tensor;multimodality;deep learning;LSTM},
  doi={10.1109/TKDE.2020.2968894}}


@ARTICLE{11005722,
  author={Singh, Priya and Jha, Manoj and Patel, Harshita},
  journal={IEEE Access}, 
  title={Wavelet-Enhanced Deep Learning Ensemble for Accurate Stock Market Forecasting: A Case Study of Nifty 50 Index}, 
  year={2025},
  volume={13},
  number={},
  pages={87036-87047},
  keywords={Long short term memory;Predictive models;Convolutional neural networks;Stock markets;Accuracy;Indexes;Forecasting;Noise;Deep learning;Noise reduction;Ensemble learning;stock market;random forest;LSTM;TCN;CNN;statistical analysis},
  doi={10.1109/ACCESS.2025.3568634}}
electricity consumption forecasting	Wavelet-Enhanced Hybrid
LSTM-XGBoost
	@ARTICLE{10946099,
  author={Ajder, Ali and Hamza, Hisham A. A. and Ayaz, Ramazan},
  journal={IEEE Access}, 
  title={Wavelet-Enhanced Hybrid LSTM-XGBoost Model for Predicting Time Series Containing Unpredictable Events}, 
  year={2025},
  volume={13},
  number={},
  pages={58671-58679},
  keywords={Predictive models;Long short term memory;Forecasting;Load modeling;Adaptation models;Accuracy;Discrete wavelet transforms;Electricity;Data models;Time series analysis;Discrete wavelet transform;electricity consumption;hybrid LSTM-XGBoost;time series prediction;unpredictable events;wavelet-enhanced forecasting},
  doi={10.1109/ACCESS.2025.3556540}}
	Monte Carlo-based LSTM	
ANFIS Adaptive Neuro-Fuzzy Inference System
ARIMA Autoregressive Integrated Moving Average
CNN Convolutional Neural Networks
RF Random Forest
WANN Wavelet-Artificial Neural Network
XGBoost  Extreme Gradient Boosting


