# Misalignment Diagnosis Using Machine Learning ⚙️📊

<h1>Condition-Based Maintenance for Rotating Machinery Using K-Nearest Neighbor (KNN)</h1>

<h2>Project Overview</h2>
This project focuses on diagnosing coupling misalignment in rotating machinery using vibration signal analysis and machine learning (K-Nearest Neighbor - KNN). By implementing Condition-Based Maintenance (CBM), the system enables early fault detection, minimizing downtime and preventing severe mechanical failures. The approach integrates signal processing techniques and statistical feature extraction to improve classification accuracy.
<br />

<h2>Key Features</h2>

- <b>✅ Real-Time Condition Monitoring – Uses vibration signal analysis to detect angular and parallel misalignment.</b>
  
- <b>✅ High-Accuracy Classification – KNN model achieves 99.16% accuracy in misalignment detection.</b>

- <b>✅ Feature Extraction & Selection – Utilizes Ensemble Empirical Mode Decomposition (EEMD) and Distance Evaluation Technique (DET) for optimal feature selection.<b>

- <b>✅ MATLAB-Based Implementation – Data processing, model training, and classification are performed in MATLAB.<b>

- <b>✅ Multi-Speed Testing – Tested at 500 RPM, 1000 RPM, and 1500 RPM under varying misalignment conditions.</b> 

<h2>Development Process</h2>

- <b>1️⃣ Data Collection – Vibration signals recorded using accelerometer sensors at different misalignment levels.<b>
 
- <b>2️⃣ Signal Processing – Applied time-domain & frequency-domain analysis for feature extraction.<b>

- <b>3️⃣ Feature Selection – Implemented EEMD & DET to enhance data efficiency.<b>

- <b>4️⃣ Machine Learning Model – Trained a KNN classifier to differentiate normal vs. misaligned conditions.<b>

- <b>5️⃣ Model Evaluation – Tested classification accuracy using confusion matrix & performance metrics.<b>

<h2>Results & Impact</h2>

- <b>🚀 98.56% classification accuracy on average Highest testing model, demonstrating the feasibility of KNN for misalignment detection.<b>

- <b>🚀 Early failure prediction, reducing unexpected downtime and maintenance costs.<b>

- <b>🚀 Applicable to various industrial settings, including oil & gas, manufacturing, and power generation.<b>

<h2>Future Improvements</h2>

- <b>🔹 Integration with IoT-based monitoring systems for real-time diagnostics.<b>

- <b>🔹 Expansion to other types of faults (e.g., unbalance, bearing defects).<b>

- <b>🔹 Comparison with deep learning models (ANN, CNN) for performance benchmarking.<b>

<h2>Program walk-through:</h2>

<p align="center">
Comparison of Accuracy Results of the DET + KNN Method in Angular Misalignment Radial Condition: <br/>
<img src="https://i.imgur.com/F7CpShH.png" height="80%" width="50%" alt="Disk Sanitization Steps"/>
<br />
<br />
Comparison of Accuracy Results of the DET + KNN Method in Angular Misalignment Axial Condition:  <br/>
<img src="https://i.imgur.com/LIwkjpN.png" height="80%" width="50%" alt="Disk Sanitization Steps"/>
<br />
<br />
Comparison of Accuracy Results of the DET + KNN Method in Parallel Misalignment Radial Condition: <br/>
<img src="https://i.imgur.com/1cVCPzE.png" height="80%" width="50%" alt="Disk Sanitization Steps"/>
<br />
<br />
Comparison of Accuracy Results of the DET + KNN Method in Parallel Misalignment Axial Condition: <br/>
<img src="https://i.imgur.com/wuLJ9uH.png" height="80%" width="50%" alt="Disk Sanitization Steps"/>

<!--
 ```diff
- text in red
+ text in green
! text in orange
# text in gray
@@ text in purple (and bold)@@
```
--!>
