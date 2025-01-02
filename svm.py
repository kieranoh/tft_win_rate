import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score, f1_score
import shap
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager, rc

# korean font
font_path = "C:/Windows/Fonts/malgun.ttf"  # Windows: Malgun Gothic
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

def process_win(units_data):
    if units_data >=4:
        return  1
    else:
        return 0

# Apply processing to the data
data['Total Traits Level'] = data['Traits'].apply(process_traits)
data['result'] = data['placement'].apply(process_win)
data[['Total Unit Tier', 'Total Unit Items']] = data['Units'].apply(
    lambda x: pd.Series(process_units(x))
)

# Add additional features
additional_features = ['Total Traits Level', 'Total Unit Tier', 'Total Unit Items',  'level']
X = data[additional_features]
y = data['result']

# Step 3: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Step 4: Train the model
# param_grid = {
#     'C': [0.1, 1, 10, 100],
#     'gamma': [0.001, 0.01, 0.1, 1],
#     'kernel': ['rbf', 'poly', 'sigmoid']
# }

# grid_search = GridSearchCV(SVR(), param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)
# grid_search.fit(X_train, y_train)

# # Best hyperparameters
# best_params = grid_search.best_params_
# print(f"Best Parameters: {best_params}")

# # Train the model with best parameters
# model = grid_search.best_estimator_

model = SVR(C=1,gamma=0.01, kernel = 'rbf')
model.fit(X_train, y_train)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)
y_pred_rounded = np.round(y_pred)

# Calculate metrics
mse = mean_squared_error(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred_rounded)
f1 = f1_score(y_test, y_pred_rounded, average='weighted')

print(f"Mean Squared Error: {mse}")
print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")

# Step 6: SHAP analysis
explainer = shap.KernelExplainer(model.predict, X_train)
shap_values = explainer.shap_values(X_test, nsamples=100)

# Step 7: Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")
shap.summary_plot(shap_values, X_test)

# Step 8: Save predictions
test_results = pd.concat([
    X_test.reset_index(drop=True),
    y_test.reset_index(drop=True),
    pd.Series(y_pred, name="Predicted Rank")
], axis=1)

output_path = 'predicted_test_data_with_shap_svm.xlsx'
test_results.to_excel(output_path, index=False)
print(f"Predictions and SHAP analysis saved to {output_path}")
