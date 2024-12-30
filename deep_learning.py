import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import shap
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt

# 한글 폰트 설정
font_path = "C:/Windows/Fonts/malgun.ttf" 
font_name = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# Step 1: Load the data
file_path = './data/game_data.xlsx'
data = pd.read_excel(file_path)

# Step 2: Process traits and units
def process_traits(traits_data):
    total_level = 0
    if not isinstance(traits_data, str):
        return total_level
    for trait_info in traits_data.split(","):
        trait_info = trait_info.strip()
        if " (Level " in trait_info:
            try:
                _, level_info = trait_info.split(" (Level ")
                total_level += int(level_info.strip(")"))
            except ValueError:
                continue
    return total_level

def process_units(units_data):
    total_tier = 0
    total_items = 0
    if not isinstance(units_data, str):
        return total_tier, total_items
    for unit_info in units_data.split(";"):
        unit_info = unit_info.strip()
        if unit_info:
            tier_info = unit_info.split("Tier ")[-1].split(",")[0].strip()
            tier = int(tier_info) if tier_info.isdigit() else 0
            items_section = unit_info.split("Items: ")[-1].strip(" )")
            num_items = len(items_section.split(", ")) if items_section else 0
            total_tier += tier
            total_items += num_items
    return total_tier, total_items

# Apply processing to the data
data['Total Traits Level'] = data['Traits'].apply(process_traits)
data[['Total Unit Tier', 'Total Unit Items']] = data['Units'].apply(
    lambda x: pd.Series(process_units(x))
)

# Add additional features
additional_features = ['Total Traits Level', 'Total Unit Tier', 'Total Unit Items', '\ub808\ubca8']
X = data[additional_features]
y = data['\uc21c\uc704']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 5: Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),  # 첫 번째 히든 레이어
    Dense(64, activation='relu'),  # 두 번째 히든 레이어
    Dense(32, activation='relu'),  # 세 번째 히든 레이어
    Dense(1, activation='linear')  # 출력 레이어
])

model.compile(optimizer=Adam(learning_rate=0.05), loss='mse', metrics=['mae'])

# EarlyStopping 콜백 설정
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Step 6: Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=8000, batch_size=32, 
                    verbose=1, callbacks=[early_stopping])

# Step 7: Evaluate the model
y_pred = model.predict(X_test).flatten()
y_pred_rounded = np.round(y_pred)

mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# Calculate accuracy and F1 score for rounded predictions
accuracy = accuracy_score(y_test, y_pred_rounded)
f1 = f1_score(y_test, y_pred_rounded, average='weighted')
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")


# Check for missing values and fill them
X_train_df = pd.DataFrame(X_train, columns=additional_features)
X_test_df = pd.DataFrame(X_test, columns=additional_features)

if X_train_df.isnull().sum().sum() > 0 or X_test_df.isnull().sum().sum() > 0:
    print("Warning: Missing values found. Filling with 0.")
    X_train_df = X_train_df.fillna(0)
    X_test_df = X_test_df.fillna(0)

# Step 8: SHAP analysis with DeepExplainer

background = X_train[:100]  # SHAP background samples (reduces computation time)
explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(X_test[:100])  # Use a subset of test data for efficiency

# Step 9: Summary plot
plt.figure()
shap.summary_plot(shap_values, X_test_df[:100], plot_type="bar")
plt.title("SHAP Feature Importance (Bar Plot)")

plt.figure()
shap.summary_plot(shap_values, X_test_df[:100])
plt.title("SHAP Feature Importance (Summary Plot)")


# Step 10: Save predictions
test_results = pd.concat([
    pd.DataFrame(X_test, columns=additional_features),
    y_test.reset_index(drop=True),
    pd.Series(y_pred, name="Predicted Rank")
], axis=1)

output_path = 'predicted_test_data_with_shap_nn.xlsx'
test_results.to_excel(output_path, index=False)
print(f"Predictions and SHAP analysis saved to {output_path}")
