import numpy as np
import pandas as pd
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
import os, random

os.makedirs("../results", exist_ok=True)


df = pd.read_csv("../data/xy_data.csv")
xs, ys = df["x"].values, df["y"].values
n = len(xs)
t = np.linspace(6, 60, n)


def curve(p):
    th, M, X = p
    k = np.exp(M * np.abs(t)) * np.sin(0.3 * t)
    x = t * np.cos(th) - k * np.sin(th) + X
    y = 42 + t * np.sin(th) + k * np.cos(th)
    return x, y

def residuals(p):
    x_pred, y_pred = curve(p)
    return np.hstack([x_pred - xs, y_pred - ys])

def L1_error(p):
    x_pred, y_pred = curve(p)
    return np.sum(np.abs(x_pred - xs) + np.abs(y_pred - ys))


p0 = [np.deg2rad(20), 0.0, 50.0]
low = [0, -0.15, 0]
high = [np.deg2rad(50), 0.05, 100]

res1 = least_squares(
    residuals, p0, bounds=(low, high),
    max_nfev=15000, loss="soft_l1", f_scale=5.0
)


pred_x, pred_y = curve(res1.x)
residual_mag = np.sqrt((pred_x - xs)**2 + (pred_y - ys)**2)
median_res = np.median(residual_mag)
mad = np.median(np.abs(residual_mag - median_res))
threshold = median_res + 3 * mad
mask = residual_mag < threshold

xs_f, ys_f = xs[mask], ys[mask]
t_f = np.linspace(6, 60, len(xs_f))

print(f"Removed {np.sum(~mask)} outliers of {n} total points.")


def curve_f(p):
    th, M, X = p
    k = np.exp(M * np.abs(t_f)) * np.sin(0.3 * t_f)
    x = t_f * np.cos(th) - k * np.sin(th) + X
    y = 42 + t_f * np.sin(th) + k * np.cos(th)
    return x, y

def residuals_f(p):
    x_pred, y_pred = curve_f(p)
    return np.hstack([x_pred - xs_f, y_pred - ys_f])

def L1_error_f(p):
    x_pred, y_pred = curve_f(p)
    return np.sum(np.abs(x_pred - xs_f) + np.abs(y_pred - ys_f))

best_p, best_score = None, np.inf
for _ in range(5):
    p0 = [np.deg2rad(random.uniform(0, 50)),
          random.uniform(-0.1, 0.05),
          random.uniform(0, 100)]
    res = least_squares(
        residuals_f, p0,
        bounds=(low, high),
        max_nfev=20000,
        ftol=1e-10, xtol=1e-10,
        loss="soft_l1", f_scale=5.0
    )
    score = L1_error_f(res.x)
    if score < best_score:
        best_score, best_p = score, res.x

th, M, X = best_p
print(f"\nTheta = {th:.6f} rad ({np.rad2deg(th):.3f}°)")
print(f"M = {M:.6f}")
print(f"X = {X:.6f}")
print(f"Total L1 = {best_score:.3f}")
print(f"Average L1 per point = {best_score/len(xs_f):.3f}")
latex_eq = (
    "\\left("
    f"t\\cos({th:.6f}) - e^{{{M:.6f}\\left|t\\right|}}\\sin(0.3t)\\sin({th:.6f}) + {X:.6f},"
    f"42 + t\\sin({th:.6f}) + e^{{{M:.6f}\\left|t\\right|}}\\sin(0.3t)\\cos({th:.6f})"
    "\\right)"
)

print("\nLaTeX Equation for Submission:\n")
print(latex_eq)


x_fit, y_fit = curve_f(best_p)
plt.figure(figsize=(8,5))
plt.scatter(xs_f, ys_f, s=25, color="blue", label="Filtered Data")
plt.plot(x_fit, y_fit, "r--", label="Robust Refined Fit")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Robust Fitted Curve with Outlier Removal")
plt.legend(); plt.grid(True)
plt.savefig("../results/fitted_plot.png", dpi=200)
plt.show()
