# Curve Fitting Project

## Goal
The task was to find the unknown values **θ (theta)**, **M**, and **X** in the following equations:

\[
x = t \cos(\theta) - e^{M|t|}\sin(0.3t)\sin(\theta) + X
\]
\[
y = 42 + t\sin(\theta) + e^{M|t|}\sin(0.3t)\cos(\theta)
\]

The fit quality was measured using the **L1 distance** between the predicted and actual data points.

---

## Step 1: Simple Version
I first created a simple version using the `least_squares` function from SciPy.  
Here I assumed that all data points were evenly spaced in time (`t`).

This version worked and gave the first set of parameter values, but the **average L1 distance** was high, meaning the curve did not match the data closely.  
Still, it confirmed that my code and equations were correct.

---

## Step 2: Improved Version (Outlier Removal)
To improve accuracy, I modified the code in two ways:

1. **Used a soft L1 loss:**  
   This reduced the influence of large errors caused by outlier points.

2. **Removed outliers:**  
   After the first run, I found and removed data points that were far from the predicted curve, then refitted the model.

These two changes made the curve smoother and closer to the main trend of the data, and reduced the L1 distance.

---

## Step 3: Final Results

| Parameter | Value |
|------------|--------|
| θ (radians) | 0.47579 |
| θ (degrees) | 27.26° |
| M | 0.007214 |
| X | 54.610287 |
| Total L1 Distance | 34187.04 |
| Average L1 per Point | 23.87 |

The final curve follows the overall shape of the dataset well.  
The small error is expected since the data contains some noise and nonlinear variations.

---

## Step 4: Final LaTeX Equation
\left(t\cos(0.475790) - e^{0.007214\left|t\right|}\sin(0.3t)\sin(0.475790) + 54.610287,
42 + t\sin(0.475790) + e^{0.007214\left|t\right|}\sin(0.3t)\cos(0.475790)\right)


You can paste this directly into [Desmos](https://www.desmos.com/calculator) to visualize the fitted curve.

---

## Step 5: Files Used

| File | Description |
|------|--------------|
| `src/main.py` | Python code for curve fitting |
| `data/xy_data.csv` | Input dataset |
| `results/fitted_plot.png` | Output plot of the fitted curve |

---

## Step 6: Summary
- Started with a **simple version** using uniform `t` values.  
- Added **soft L1 loss** and **outlier removal** for better results.  
- Reduced L1 distance and obtained a smoother, more accurate fit.  
- The final equation captures the main shape and trend of the data effectively.



