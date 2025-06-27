
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random
import math

# Fungsi jarak
def euclidean_distance(x1, y1, point2):
    x2, y2 = point2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def compute_distance_to_packing(df, packing_point):
    return np.sqrt((df["x"] - packing_point[0])**2 + (df["y"] - packing_point[1])**2)

def total_weighted_distance(df):
    return np.sum(df["order_frequency"] * df["distance_to_packing"])

def simulated_annealing(df, packing_point, max_iter=1000, initial_temp=100, cooling_rate=0.99):
    current_df = df.copy()
    current_df["distance_to_packing"] = compute_distance_to_packing(current_df, packing_point)
    current_cost = total_weighted_distance(current_df)

    best_df = current_df.copy()
    best_cost = current_cost

    temp = initial_temp

    for _ in range(max_iter):
        new_df = current_df.copy()
        idx1, idx2 = np.random.choice(len(df), 2, replace=False)
        new_df.at[idx1, "x"], new_df.at[idx2, "x"] = new_df.at[idx2, "x"], new_df.at[idx1, "x"]
        new_df.at[idx1, "y"], new_df.at[idx2, "y"] = new_df.at[idx2, "y"], new_df.at[idx1, "y"]

        new_df["distance_to_packing"] = compute_distance_to_packing(new_df, packing_point)
        new_cost = total_weighted_distance(new_df)

        delta = new_cost - current_cost
        if delta < 0 or np.random.rand() < math.exp(-delta / temp):
            current_df = new_df
            current_cost = new_cost
            if new_cost < best_cost:
                best_df = new_df.copy()
                best_cost = new_cost

        temp *= cooling_rate

    return best_df, best_cost

def simulate_total_distance(orders, layout_df, packing_point):
    slot_dict = layout_df.set_index("slot_id")[["x", "y"]].to_dict("index")
    distances = []
    for order in orders:
        coords = [packing_point] + [tuple(slot_dict[slot].values()) for slot in order] + [packing_point]
        dist = sum(euclidean_distance(*coords[i], coords[i+1]) for i in range(len(coords)-1))
        distances.append(dist)
    return np.array(distances)

# Sidebar
st.sidebar.header("⚙️ Konfigurasi Gudang")
num_slots = st.sidebar.slider("Jumlah Slot", 50, 200, 100)
grid_x = st.sidebar.slider("Grid X", 5, 20, 10)
grid_y = st.sidebar.slider("Grid Y", 5, 20, 12)
num_orders = st.sidebar.slider("Jumlah Order", 10, 100, 50)
items_per_order = st.sidebar.slider("Barang per Order", 1, 5, 3)
picker_speed = st.sidebar.number_input("Kecepatan Picker (m/s)", value=1.5)
packing_x = st.sidebar.number_input("Packing Point X", value=9)
packing_y = st.sidebar.number_input("Packing Point Y", value=5)

packing_point = (packing_x, packing_y)

# Generate dummy layout
np.random.seed(42)
x_coords = np.random.randint(0, grid_x, num_slots)
y_coords = np.random.randint(0, grid_y, num_slots)
order_freq = np.random.randint(1, 101, num_slots)
distance_to_packing = np.sqrt((x_coords - packing_point[0])**2 + (y_coords - packing_point[1])**2)

dummy_layout_df = pd.DataFrame({
    "slot_id": [f"S{i:03d}" for i in range(num_slots)],
    "x": x_coords,
    "y": y_coords,
    "order_frequency": order_freq,
    "distance_to_packing": distance_to_packing.round(2)
})

# Optimasi layout
optimized_df, _ = simulated_annealing(dummy_layout_df, packing_point)

# Buat order
valid_slot_ids = optimized_df["slot_id"].tolist()
orders_fixed = [random.sample(valid_slot_ids, items_per_order) for _ in range(num_orders)]

# Simulasi jarak dan waktu
before_distances = simulate_total_distance(orders_fixed, dummy_layout_df, packing_point)
after_distances = simulate_total_distance(orders_fixed, optimized_df, packing_point)

before_times = before_distances / picker_speed
after_times = after_distances / picker_speed

comparison_df = pd.DataFrame({
    "order_id": range(1, len(orders_fixed)+1),
    "distance_before": before_distances.round(2),
    "distance_after": after_distances.round(2),
    "time_before_sec": before_times.round(2),
    "time_after_sec": after_times.round(2),
    "time_saved_sec": (before_times - after_times).round(2)
})

# Visualisasi
st.title("📦 Simulasi Optimasi Layout Gudang 🚀")

st.subheader("🗺️ Visualisasi Layout Gudang")
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
scatter1 = axs[0].scatter(dummy_layout_df["x"], dummy_layout_df["y"],
                          c=dummy_layout_df["order_frequency"], cmap="plasma", s=80)
axs[0].set_title("Before Optimization")
axs[0].invert_yaxis()

scatter2 = axs[1].scatter(optimized_df["x"], optimized_df["y"],
                          c=optimized_df["order_frequency"], cmap="plasma", s=80)
axs[1].set_title("After Optimization")
axs[1].invert_yaxis()
st.pyplot(fig)

# Tabel perbandingan
st.subheader("⏱️ Perbandingan Waktu Picking")
st.dataframe(comparison_df)

# Histogram
st.subheader("📊 Distribusi Waktu Picking")
fig_hist = plt.figure(figsize=(10, 6))
plt.hist(comparison_df["time_before_sec"], bins=15, alpha=0.6, label="Before")
plt.hist(comparison_df["time_after_sec"], bins=15, alpha=0.6, label="After")
plt.xlabel("Waktu (detik)")
plt.ylabel("Jumlah Order")
plt.title("Distribusi Waktu Picking")
plt.legend()
st.pyplot(fig_hist)

# Ringkasan
avg_before = comparison_df["time_before_sec"].mean()
avg_after = comparison_df["time_after_sec"].mean()
avg_saved = comparison_df["time_saved_sec"].mean()
summary = {
    "Rata-rata waktu sebelum optimasi (detik)": round(avg_before, 2),
    "Rata-rata waktu setelah optimasi (detik)": round(avg_after, 2),
    "Rata-rata waktu dihemat (detik)": round(avg_saved, 2),
    "Efisiensi relatif (%)": round((avg_saved / avg_before) * 100, 2)
}
st.subheader("📈 Ringkasan Efisiensi Waktu")
st.json(summary)
