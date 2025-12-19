import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from geomdl import NURBS, knotvector
import matplotlib.image as mpimg
import pandas as pd
import datetime
import os
import gspread

st.set_page_config(page_title="NURBS Car Editor", layout="wide")
st.title("NURBS Car Silhouette Editor")

st.markdown(st.markdown("""
本アンケートは、**早稲田大学の研究プロジェクト**の一環として実施しているものです。  
""")
)

CAR_MODELS = {
    "SUV": {
        "degree": 3,
        "ctrlpts":[
            [4.0, 0.0], [-3.0, 1.5], [-2.0, 2.0], [-1.0, 2.2], [0.0, 2.2], [1.0, 2.0], [2.0, 1.5], [3.0, 0.5], [4.0, 0.0]
        ],
        "weights":[1,1,1,1,1,1,1,1,1]
    },
    "Compact": {
        "degree": 3,
        "ctrlpts": [
        [-3.5, 0.0],
        [-2.5, 1.2],
        [-1.5, 1.6],
        [ 0.0, 1.7],
        [ 1.5, 1.5],
        [ 2.5, 0.8],
        [ 3.5, 0.0],],
        "weights": [1, 1, 1, 1, 1, 1, 1]
    }
}

st.sidebar.header("car Model Settings")

selected_model = st.sidebar.selectbox("車種を選択してください", list(CAR_MODELS.keys()))

model_data = CAR_MODELS[selected_model]
degree = model_data["degree"]
ctrlpts = model_data["ctrlpts"]
weights = model_data["weights"]

st.sidebar.subheader("Control Points")

new_ctrlpts = []

for i, pt in enumerate(ctrlpts):
    x = st.sidebar.slider(
        f"Point {i} - X",
        float(pt[0] - 1),
        float(pt[0] + 1),
        float(pt[0]),
        0.01
    )

    y = st.sidebar.slider(
        f"Point {i} - Y",
        float(pt[1] - 1),
        float(pt[1] + 1),
        float(pt[0]),
        0.01
    )

    new_ctrlpts.append([x, y])

curve = NURBS.Curve()
curve.degree = degree
curve.ctrlpts = new_ctrlpts
curve.knotvector = knotvector.generate(curve.degree, len(new_ctrlpts))
curve.evaluate()

fig, ax = plt.subplots(figsize=(10, 5))

curve_points = np.array(curve.evalpts)
ax.plot(
    curve_points[:, 0],
    curve_points[:, 1],
    color="blue",
    linewidth=2

)

ctrlpts_np = np.array(new_ctrlpts)
ax.plot(
    ctrlpts_np[:, 0],
    ctrlpts_np[:, 1],
    "o--",
    color="red"
)

ax.set_aspect("equal")
ax.grid(True)
ax.set_xlim(-2, 12)
ax.set_ylim(-2, 6)

st.pyplot(fig)

alpha = st.sidebar.slider(
    "車体の濃さ",
    0.0,
    1.0,
    0.3,
    0.05
)

Polygon__points = list(curve_points) + [new_ctrlpts[-1], new_ctrlpts[0]]

ax.add_patch(
    Polygon(
        Polygon__points,
        closed=True,
        color="black",
        alpha=alpha
    )
)

tire_positions = [(1.5, -0.5), (8.5, -0.5)]

for x, y in tire_positions:
    ax.add_patch(
    Circle((x, y), 0.8, color="black")
)