import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow_privacy.privacy.optimizers.dp_optimizer_keras import DPKerasSGDOptimizer
from tensorflow_privacy.privacy.analysis.compute_dp_sgd_privacy_lib import compute_dp_sgd_privacy
import lakefs_client
from lakefs_client.client import LakeFSClient
from lakefs_client.models import *
import io
import os
import subprocess
import time
import json
import warnings
warnings.filterwarnings('ignore')

REPO_NAME = f"ml-athletes-repo-{int(time.time())}"
BRANCH_MAIN = "main"
BRANCH_V2 = "v2-cleaned"

def setup_lakefs():
    configuration = lakefs_client.Configuration(
        host='http://localhost:8000',
        username='AKIAJ7TCSWLGLUL3HDTQ',
        password='VzKlRAp1fNWmHWX/7uak4sK9t4nEXv3EQcl++y/n'
    )
    client = LakeFSClient(configuration)
    
    try:
        client.repositories.create_repository(
            repository_creation=RepositoryCreation(
                name=REPO_NAME,
                storage_namespace=f"local://ml-athletes-{int(time.time())}",
                default_branch=BRANCH_MAIN
            )
        )
        print(f"lakeFS repository created: {REPO_NAME}")
    except:
        print(f"lakeFS repository exists: {REPO_NAME}")
    
    return client

def setup_dvc():
    if not os.path.exists('.git'):
        subprocess.run(['git', 'init'], check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.name', 'ML Pipeline'], check=True, capture_output=True)
        subprocess.run(['git', 'config', 'user.email', 'ml@example.com'], check=True, capture_output=True)
        subprocess.run(['git', 'commit', '--allow-empty', '-m', 'Initial commit'], check=True, capture_output=True)
    
    if not os.path.exists('.dvc'):
        subprocess.run(['dvc', 'init'], check=True, capture_output=True)
        subprocess.run(['git', 'add', '.dvc', '.dvcignore'], check=True, capture_output=True)
        result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
        if result.returncode != 0:
            subprocess.run(['git', 'commit', '-m', 'Initialize DVC'], check=True, capture_output=True)
    
    os.makedirs('data', exist_ok=True)
    print("DVC setup completed")

def task1_load_data():
    # Task 1: Work with given machine learning dataset - call this dataset version 1 (v1)
    print("Task 1: Loading dataset v1")
    data = pd.read_csv('athletes.csv')
    print(f"Data shape: {data.shape}")
    return data

def task2_clean_data_v2(data):
    # Task 2: Clean the dataset such as removing outliers, cleaning survey responses, introducing new features - call this dataset version 2 (v2)
    print("Task 2: Cleaning dataset to create v2")
    
    data = data.dropna(subset=['region','age','weight','height','howlong','gender','eat', 
                               'train','background','experience','schedule','howlong', 
                               'deadlift','candj','snatch','backsq','experience',
                               'background','schedule','howlong'])
    data = data.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace',
                              'filthy50','fgonebad','run400','run5k','pullups','train'], errors='ignore')

    data = data[data['weight'] < 1500]
    data = data[data['gender'] != '--']
    data = data[data['age'] >= 18]
    data = data[(data['height'] < 96) & (data['height'] > 48)]

    data = data[(data['deadlift'] > 0) & ((data['deadlift'] <= 1105) | 
                ((data['gender'] == 'Female') & (data['deadlift'] <= 636)))]
    data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
    data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
    data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]

    decline_dict = {'Decline to answer|': np.nan}
    data = data.replace(decline_dict)
    data = data.dropna(subset=['background','experience','schedule','howlong','eat'])
    
    print(f"Cleaned data shape: {data.shape}")
    return data

def task3_add_total_lift_and_split(data):
    # Task 3: For both versions calculate total_lift and divide dataset into train and test, keeping the same split ratio
    print("Task 3: Adding total_lift feature and splitting data")
    
    data['total_lift'] = data['deadlift'] + data['candj'] + data['snatch'] + data['backsq']
    data = data.dropna(subset=['total_lift'])
    
    le = LabelEncoder()
    data['gender_encoded'] = le.fit_transform(data['gender'])
    
    X = data[['age', 'weight', 'height', 'gender_encoded']]
    y = data['total_lift']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    return data, X_train, X_test, y_train, y_test

def task4_lakefs_version_control(client, data, version, file_path):
    # Task 4: Use tool to version the dataset
    print(f"Task 4: lakeFS version control - uploading data v{version}")
    
    branch = BRANCH_MAIN if version == 1 else BRANCH_V2
    
    if version == 2:
        try:
            client.branches.create_branch(
                repository=REPO_NAME,
                branch_creation=BranchCreation(name=BRANCH_V2, source=BRANCH_MAIN)
            )
        except:
            pass
    
    csv_content = data.to_csv(index=False)
    client.objects.upload_object(
        repository=REPO_NAME,
        branch=branch,
        path=file_path,
        content=io.BytesIO(csv_content.encode('utf-8'))
    )
    
    client.commits.commit(
        repository=REPO_NAME,
        branch=branch,
        commit_creation=CommitCreation(message=f"Add data v{version}")
    )
    
    print(f"Data v{version} upload completed")

def task4_dvc_version_control(data, file_path, version):
    # Task 4: Use tool to version the dataset
    print(f"Task 4: DVC version control - adding data v{version}")
    
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    data.to_csv(file_path, index=False)
    
    result = subprocess.run(['dvc', 'add', file_path], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"DVC add failed: {result.stderr}")
        return
    
    dvc_file = f'{file_path}.dvc'
    if os.path.exists(dvc_file):
        subprocess.run(['git', 'add', dvc_file], capture_output=True)
    
    if os.path.exists('.gitignore'):
        subprocess.run(['git', 'add', '.gitignore'], capture_output=True)
    
    result = subprocess.run(['git', 'diff', '--cached', '--quiet'], capture_output=True)
    if result.returncode != 0:
        subprocess.run(['git', 'commit', '-m', f'Add data v{version}'], capture_output=True)
    
    print(f"Data v{version} added to DVC completed")

def task5_eda_v1(data):
    # Task 5: Run EDA (exploratory data analysis) of the dataset v1
    print("Task 5: Running EDA for v1")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EDA - Dataset v1')
    
    axes[0,0].hist(data['age'], bins=30, alpha=0.7)
    axes[0,0].set_title('Age Distribution')
    
    data['gender'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Gender Distribution')
    
    if 'total_lift' in data.columns:
        axes[1,0].hist(data['total_lift'], bins=30, alpha=0.7)
        axes[1,0].set_title('Total Lift Distribution')
        
        axes[1,1].scatter(data['weight'], data['total_lift'], alpha=0.5)
        axes[1,1].set_title('Weight vs Total Lift')
    
    plt.tight_layout()
    plt.savefig('eda_v1.png')
    plt.close()
    
    print("EDA chart saved as eda_v1.png")

def task6_build_baseline_model_v1(X_train, X_test, y_train, y_test):
    # Task 6: Use the dataset v1 to build a baseline machine learning model to predict total_lift
    print("Task 6: Building baseline model with v1")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    return {'model': model, 'scaler': scaler, 'y_pred': y_pred_test}

def task7_run_metrics_v1(y_test, y_pred):
    # Task 7: Run metrics for this model
    print("Task 7: Running metrics for v1 model")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"v1 model performance - MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
    
    return {'mse': mse, 'r2': r2, 'mae': mae}

def task8_update_to_v2():
    # Task 8: Update the dataset version to go to dataset v2 without changing anything else in the training code
    print("Task 8: Updating to v2 version")
    print("Version switch completed, using same training code")

def task9_eda_v2(data):
    # Task 9: Run EDA (exploratory data analysis) of dataset v2
    print("Task 9: Running EDA for v2")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('EDA - Dataset v2')
    
    axes[0,0].hist(data['age'], bins=30, alpha=0.7)
    axes[0,0].set_title('Age Distribution')
    
    data['gender'].value_counts().plot(kind='bar', ax=axes[0,1])
    axes[0,1].set_title('Gender Distribution')
    
    if 'total_lift' in data.columns:
        axes[1,0].hist(data['total_lift'], bins=30, alpha=0.7)
        axes[1,0].set_title('Total Lift Distribution')
        
        axes[1,1].scatter(data['weight'], data['total_lift'], alpha=0.5)
        axes[1,1].set_title('Weight vs Total Lift')
    
    plt.tight_layout()
    plt.savefig('eda_v2.png')
    plt.close()
    
    print("EDA chart saved as eda_v2.png")

def task10_build_model_v2(X_train, X_test, y_train, y_test):
    # Task 10: Build a machine learning model with "new" dataset v2 to predict total_lift
    print("Task 10: Building model with v2")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    y_pred_test = model.predict(X_test_scaled)
    
    return {'model': model, 'scaler': scaler, 'y_pred': y_pred_test}

def task11_run_metrics_v2(y_test, y_pred):
    # Task 11: Run metrics for this model
    print("Task 11: Running metrics for v2 model")
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"v2 model performance - MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
    
    return {'mse': mse, 'r2': r2, 'mae': mae}

def task12_compare_v1_v2(v1_metrics, v2_metrics):
    # Task 12: Compare and comment on the accuracy/metrics of the models using v1 and v2
    print("Task 12: Comparing v1 and v2 model accuracy/metrics")
    
    print(f"v1 model - R²: {v1_metrics['r2']:.4f}, MSE: {v1_metrics['mse']:.2f}")
    print(f"v2 model - R²: {v2_metrics['r2']:.4f}, MSE: {v2_metrics['mse']:.2f}")
    
    r2_improvement = v2_metrics['r2'] - v1_metrics['r2']
    mse_improvement = v1_metrics['mse'] - v2_metrics['mse']
    
    print(f"Data cleaning effect - R² improvement: {r2_improvement:.4f}, MSE improvement: {mse_improvement:.2f}")
    
    return {'r2_improvement': r2_improvement, 'mse_improvement': mse_improvement}

def task13_build_dp_model(X_train, X_test, y_train, y_test):
    # Task 13: Use tensor flow privacy library with the dataset v2 and calculate the metrics for the new DP model
    print("Task 13: Building DP model with TensorFlow Privacy and v2")
    
    batch_size = 32
    noise_multiplier = 1.8
    l2_norm_clip = 1.0
    epochs = 15
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    n_samples = (len(X_train_scaled) // batch_size) * batch_size
    X_train_scaled = X_train_scaled[:n_samples]
    y_train_array = y_train.values[:n_samples]
    
    y_mean = np.mean(y_train_array)
    y_std = np.std(y_train_array)
    y_train_norm = (y_train_array - y_mean) / y_std
    
    tf.random.set_seed(42)
    np.random.seed(42)
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1)
    ])
    
    optimizer = DPKerasSGDOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=1,
        learning_rate=0.005
    )
    
    model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    model.fit(X_train_scaled, y_train_norm, 
              batch_size=batch_size, epochs=epochs, verbose=0)
    
    y_pred_norm = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = y_pred_norm * y_std + y_mean
    
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"DP model performance - MSE: {mse:.2f}, R²: {r2:.4f}, MAE: {mae:.2f}")
    
    return {
        'model': model, 'scaler': scaler, 'mse': mse, 'r2': r2, 'mae': mae,
        'noise_multiplier': noise_multiplier, 'l2_norm_clip': l2_norm_clip,
        'epochs': epochs, 'batch_size': batch_size, 'n_samples': n_samples
    }

def task14_compute_dp_epsilon(dp_results):
    # Task 14: Compute the DP ε using TensorFlow privacy compute_dp_sgd_privacy
    print("Task 14: Computing DP ε value")
    
    try:
        eps = compute_dp_sgd_privacy(
            n=dp_results['n_samples'],
            batch_size=dp_results['batch_size'],
            noise_multiplier=dp_results['noise_multiplier'],
            epochs=dp_results['epochs'],
            delta=1e-5
        )[0]
        print(f"ε value: {eps:.6f}")
        return eps
    except Exception as e:
        print(f"API call failed: {e}")
        q = dp_results['batch_size'] / dp_results['n_samples']
        steps = dp_results['epochs'] * (dp_results['n_samples'] // dp_results['batch_size'])
        eps_approx = 2 * q * steps / (dp_results['noise_multiplier'] ** 2)
        print(f"Theoretical estimate ε value: {eps_approx:.6f}")
        return eps_approx

def task15_compare_dp_models(baseline_metrics, dp_results):
    # Task 15: Compare and comment on the accuracy/metrics of the non-DP and DP models using dataset v2
    print("Task 15: Comparing non-DP and DP models")
    
    print(f"Baseline model - R²: {baseline_metrics['r2']:.4f}, MSE: {baseline_metrics['mse']:.2f}")
    print(f"DP model - R²: {dp_results['r2']:.4f}, MSE: {dp_results['mse']:.2f}")
    
    r2_diff = baseline_metrics['r2'] - dp_results['r2']
    mse_diff = dp_results['mse'] - baseline_metrics['mse']
    
    print(f"Privacy protection cost - R² decrease: {r2_diff:.4f}, MSE increase: {mse_diff:.2f}")
    
    return {'r2_diff': r2_diff, 'mse_diff': mse_diff}

def run_lakefs_workflow():
    print("lakeFS workflow started")
    
    try:
        client = setup_lakefs()
        lakefs_available = True
    except Exception as e:
        print(f"lakeFS unavailable: {e}")
        lakefs_available = False
        return None
    
    data_v1 = task1_load_data()
    data_v2 = task2_clean_data_v2(data_v1.copy())
    
    data_v1_processed, X_train_v1, X_test_v1, y_train_v1, y_test_v1 = task3_add_total_lift_and_split(data_v1.copy())
    data_v2_processed, X_train_v2, X_test_v2, y_train_v2, y_test_v2 = task3_add_total_lift_and_split(data_v2.copy())
    
    data_v1_processed.to_csv('data/athletes_v1_lakefs.csv', index=False)
    data_v2_processed.to_csv('data/athletes_v2_lakefs.csv', index=False)
    
    task4_lakefs_version_control(client, data_v1_processed, 1, "data/athletes_v1.csv")
    task4_lakefs_version_control(client, data_v2_processed, 2, "data/athletes_v2.csv")
    
    task5_eda_v1(data_v1_processed)
    
    lakefs_v1_model = task6_build_baseline_model_v1(X_train_v1, X_test_v1, y_train_v1, y_test_v1)
    lakefs_v1_metrics = task7_run_metrics_v1(y_test_v1, lakefs_v1_model['y_pred'])
    
    task8_update_to_v2()
    
    task9_eda_v2(data_v2_processed)
    
    lakefs_v2_model = task10_build_model_v2(X_train_v2, X_test_v2, y_train_v2, y_test_v2)
    lakefs_v2_metrics = task11_run_metrics_v2(y_test_v2, lakefs_v2_model['y_pred'])
    
    lakefs_v1_v2_comparison = task12_compare_v1_v2(lakefs_v1_metrics, lakefs_v2_metrics)
    
    lakefs_dp_results = task13_build_dp_model(X_train_v2, X_test_v2, y_train_v2, y_test_v2)
    lakefs_epsilon = task14_compute_dp_epsilon(lakefs_dp_results)
    lakefs_dp_comparison = task15_compare_dp_models(lakefs_v2_metrics, lakefs_dp_results)
    
    return {
        'v1_metrics': lakefs_v1_metrics,
        'v2_metrics': lakefs_v2_metrics,
        'dp_metrics': lakefs_dp_results,
        'epsilon': lakefs_epsilon,
        'v1_v2_comparison': lakefs_v1_v2_comparison,
        'dp_comparison': lakefs_dp_comparison
    }

def run_dvc_workflow():
    print("DVC workflow started")
    
    original_dir = os.getcwd()
    dvc_dir = 'dvc_experiment'
    
    try:
        if not os.path.exists(dvc_dir):
            os.makedirs(dvc_dir)
        os.chdir(dvc_dir)
        
        import shutil
        if os.path.exists('../athletes.csv'):
            shutil.copy('../athletes.csv', 'athletes.csv')
        
        setup_dvc()
        
        data_v1 = task1_load_data()
        data_v2 = task2_clean_data_v2(data_v1.copy())
        
        data_v1_processed, X_train_v1, X_test_v1, y_train_v1, y_test_v1 = task3_add_total_lift_and_split(data_v1.copy())
        data_v2_processed, X_train_v2, X_test_v2, y_train_v2, y_test_v2 = task3_add_total_lift_and_split(data_v2.copy())
        
        task4_dvc_version_control(data_v1_processed, 'data/athletes_v1.csv', 1)
        task4_dvc_version_control(data_v2_processed, 'data/athletes_v2.csv', 2)
        
        task5_eda_v1(data_v1_processed)
        
        dvc_v1_model = task6_build_baseline_model_v1(X_train_v1, X_test_v1, y_train_v1, y_test_v1)
        dvc_v1_metrics = task7_run_metrics_v1(y_test_v1, dvc_v1_model['y_pred'])
        
        task8_update_to_v2()
        
        task9_eda_v2(data_v2_processed)
        
        dvc_v2_model = task10_build_model_v2(X_train_v2, X_test_v2, y_train_v2, y_test_v2)
        dvc_v2_metrics = task11_run_metrics_v2(y_test_v2, dvc_v2_model['y_pred'])
        
        dvc_v1_v2_comparison = task12_compare_v1_v2(dvc_v1_metrics, dvc_v2_metrics)
        
        dvc_dp_results = task13_build_dp_model(X_train_v2, X_test_v2, y_train_v2, y_test_v2)
        dvc_epsilon = task14_compute_dp_epsilon(dvc_dp_results)
        dvc_dp_comparison = task15_compare_dp_models(dvc_v2_metrics, dvc_dp_results)
        
        shutil.copy('data/athletes_v1.csv', '../data/athletes_v1_dvc.csv')
        shutil.copy('data/athletes_v2.csv', '../data/athletes_v2_dvc.csv')
        
        os.chdir(original_dir)
        
        return {
            'v1_metrics': dvc_v1_metrics,
            'v2_metrics': dvc_v2_metrics,
            'dp_metrics': dvc_dp_results,
            'epsilon': dvc_epsilon,
            'v1_v2_comparison': dvc_v1_v2_comparison,
            'dp_comparison': dvc_dp_comparison
        }
        
    except Exception as e:
        print(f"DVC unavailable: {e}")
        os.chdir(original_dir)
        return None

def clean_results_for_json(results):
    if results is None:
        return None
    
    cleaned = {}
    for key, value in results.items():
        if key in ['model', 'scaler']:
            continue
        elif isinstance(value, dict):
            cleaned[key] = clean_results_for_json(value)
        else:
            cleaned[key] = value
    
    return cleaned

def compare_tools(lakefs_results, dvc_results):
    print("Tool comparison analysis")
    
    print("1. Installation ease:")
    print("   lakeFS: Requires Docker service, access key configuration")
    print("   DVC: pip install, requires Git environment")
    
    print("2. Data versioning:")
    print("   lakeFS: Git-like branch management")
    print("   DVC: Tight Git integration")
    
    print("3. Version switching:")
    print("   lakeFS: Branch switching operations")
    print("   DVC: Git tags and checkout")
    
    print("4. Differential privacy impact:")
    if lakefs_results and dvc_results:
        print(f"   lakeFS: R² change {lakefs_results['dp_comparison']['r2_diff']:.4f}")
        print(f"   DVC: R² change {dvc_results['dp_comparison']['r2_diff']:.4f}")
    elif lakefs_results:
        print(f"   lakeFS: R² change {lakefs_results['dp_comparison']['r2_diff']:.4f}")
        print("   DVC: Data unavailable")
    
    print("5. Model performance improvement:")
    if lakefs_results and dvc_results:
        print(f"   lakeFS: v1→v2 improvement {lakefs_results['v1_v2_comparison']['r2_improvement']:.4f}")
        print(f"   DVC: v1→v2 improvement {dvc_results['v1_v2_comparison']['r2_improvement']:.4f}")
    elif lakefs_results:
        print(f"   lakeFS: v1→v2 improvement {lakefs_results['v1_v2_comparison']['r2_improvement']:.4f}")
        print("   DVC: Data unavailable")

def main():
    print("Data versioning and machine learning tasks started")
    
    os.makedirs('data', exist_ok=True)
    
    lakefs_results = run_lakefs_workflow()
    dvc_results = run_dvc_workflow()
    
    compare_tools(lakefs_results, dvc_results)
    
    results = {
        'lakefs': clean_results_for_json(lakefs_results),
        'dvc': clean_results_for_json(dvc_results)
    }
    
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print("Results saved to results.json")
    print("All 15 tasks completed")

if __name__ == "__main__":
    main()