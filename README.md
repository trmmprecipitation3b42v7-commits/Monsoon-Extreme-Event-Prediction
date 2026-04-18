Multi-Scale Deep Learning for Extreme Monsoon Event Forecasting


Hybrid Wavelet-LSTM Framework for High-Frequency Precipitation Modeling

📌 Project Overview
Predicting extreme precipitation is challenging due to the highly non-linear and "noisy" nature of environmental signals. This project implements a Multi-Scale Feature Engineering pipeline that decomposes 20 years of satellite data into frequency components before training a Deep Learning model.

By combining Discrete Wavelet Transforms (DWT) for denoising and LSTMs for temporal memory, the architecture captures both long-term climate trends and high-frequency extreme "signatures."

📊 Dataset Specifications
Source: TRMM 3B42RT Daily (Gauge-corrected, Research Grade V07B).

Temporal Range: March 2000 – December 2019 (approx. 7,000+ daily observations).

Spatial Resolution: 0.1° × 0.1° (~11 km grid) area-averaged over South Asia (83°E-86°E, 20°N-23°N).

Variables: precipitationCal (mm/day).

🛠️ The Pipeline Architecture
1. Signal Decomposition (DWT)
The raw signal is decomposed using a Daubechies 6 (db6) wavelet filter at a 6-level decomposition.

Approximation (A6): Represents the smoothed, low-frequency seasonal trends.

Details (D1-D6): Captures multi-scale fluctuations, where D1 represents the finest "extreme" signatures.

2. Extreme-Preserving Reconstruction
To separate signal from noise, I applied an adaptive Donoho-Johnstone threshold to the detail coefficients. This ensures that extreme precipitation peaks are preserved while random measurement noise is filtered out via soft thresholding.

3. Feature Engineering & MLOps
Instead of raw sequences, the model is fed a rich feature set:

Wavelet Components: Individual upsampled A and D components.

Temporal Lags: 1-day and 7-day lags to capture autocorrelation.

Cyclical Seasonality: Sine/Cosine transforms of the "Day of Year" to map annual periodicity.

4. Deep Learning (LSTM) Model
Architecture: Stacked LSTM layers (100 units) with ReLU activation.

Regularization: Dropout layers and EarlyStopping (monitoring validation loss) to ensure generalization and prevent overfitting.

Optimization: Adam optimizer with Mean Squared Error (MSE) loss.

🚀 Installation & Usage
Bash
# Clone the repository
git clone https://github.com/trmmprecipitation3b42v7-commits/Monsoon-Extreme-Event-Prediction.git

# Install dependencies
pip install numpy pandas pywavelets scikit-learn tensorflow matplotlib
📈 Key Results
Error Metrics: The model is evaluated using MAE and RMSE on a chronological 20% test split.

Performance: The hybrid reconstruction approach significantly reduces "chatter" in the predictions while accurately capturing the magnitude of high-intensity rainfall events compared to vanilla LSTM models.
