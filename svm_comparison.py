# svm_comparison.py

import time
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.datasets import load_digits # データセットのロード
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA # 次元削減
from sklearn.svm import SVC # 古典的SVM
from sklearn.metrics import accuracy_score

from qiskit import QuantumCircuit, transpile
from qiskit_machine_learning.kernels.fidelity_quantum_kernel import FidelityQuantumKernel as QuantumKernel # QSVMのカーネル
from qiskit.primitives import Sampler, Estimator, StatevectorSampler
from qiskit.circuit.library import ZFeatureMap # <-- ZFeatureMapをインポート (Qiskitのコアライブラリから)

# figディレクトリが存在しない場合は作成
output_dir = 'fig_svm'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def create_feature_map(num_qubits):
    """QSVM用の特徴マップ（量子回路）を作成"""
    # ZFeatureMapを使用することで、データ次元と量子ビット数の整合性を自動で処理
    # rotation_blocks=['ry'] なども指定できますが、ここではシンプルな状態を維持
    feature_map = ZFeatureMap(feature_dimension=num_qubits, reps=2) # entanglement引数を削除
    return feature_map

def plot_decision_boundary(X, y, model, title, filename):
    """決定境界とデータポイントを可視化 """
    # 2次元グリッドを作成し、各点でモデルの予測を行う
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # モデルの種類に応じて予測方法を調整
    if hasattr(model, "predict"): # scikit-learnのモデル
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    else:
        # 予測方法が不明な場合はエラーまたはスキップ
        print(f"Warning: Cannot plot decision boundary for model type {type(model)}")
        return

    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Class')
    plt.savefig(os.path.join(output_dir, filename))
    plt.close() # プロットを閉じてメモリを解放

def run_classification_experiment(digits, target_digits, dataset_name):
    """指定された数字のデータセットで分類実験を実行する関数"""
    print(f"\n--- Running experiment for digits {target_digits[0]} and {target_digits[1]} ({dataset_name}) ---")

    # 1. データセットの選択と準備
    idx = np.logical_or(digits.target == target_digits[0], digits.target == target_digits[1])
    X = digits.data[idx]
    y = digits.target[idx]
    y[y == target_digits[0]] = 0 # クラスラベルを0と1に変換
    y[y == target_digits[1]] = 1

    # データの正規化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 次元削減 (PCA: 64次元から2次元へ) [cite: 66]
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 訓練セットとテストセットに分割
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    # --- CSVMの実行と評価 ---
    print("\n--- Classical SVM (CSVM) ---")

    # 異なるカーネル関数を試す
    kernels = ['linear', 'rbf', 'poly']
    for kernel_type in kernels:
        print(f"  Kernel: {kernel_type}")
        start_time = time.time()
        csvm = SVC(kernel=kernel_type, random_state=42)
        csvm.fit(X_train, y_train)
        csvm_train_time = time.time() - start_time
        y_pred_csvm = csvm.predict(X_test)
        csvm_accuracy = accuracy_score(y_test, y_pred_csvm)

        print(f"    Execution Time: {csvm_train_time:.4f} seconds ")
        print(f"    Accuracy: {csvm_accuracy:.4f} ")

        # 決定境界とデータポイントの可視化
        plot_decision_boundary(X_pca, y, csvm,
                               f'CSVM Decision Boundary (Kernel: {kernel_type}) for {dataset_name}',
                               f'{dataset_name}_csvm_{kernel_type}_boundary.png')

    # --- QSVMの実行と評価 ---
    print("\n--- Quantum Support Vector Machine (QSVM) ---")

    # QSVMのための特徴マップを作成 (量子カーネルの基盤)
    # 2次元に削減されたデータのため、num_qubitsは2
    feature_map = create_feature_map(num_qubits=2)

    # 量子インスタンスの設定 (Qiskit 1.0以降の推奨プラクティス)
    # Samplerを使用して、カーネル計算をバックエンドで実行
    sampler = StatevectorSampler() # DeprecationWarning解消のためStatevectorSamplerを使用

    # QuantumKernelのインスタンス化
    quantum_kernel = QuantumKernel(feature_map=feature_map) # sampler引数を削除

    # SVCモデルに量子カーネルを「precomputed」として渡す

    # トレーニングデータでカーネル行列を計算
    start_time_kernel_calc = time.time()
    # evaluateメソッドにsamplerを渡す (x_train, y_trainの引数名を削除し、位置引数として渡す)
    kernel_matrix_train = quantum_kernel.evaluate(X_train, X_train)
    kernel_calc_time = time.time() - start_time_kernel_calc
    print(f"    Quantum Kernel Calculation Time (Train): {kernel_calc_time:.4f} seconds")

    # scikit-learnのSVC with precomputed kernel
    start_time_qsvc_train = time.time()
    qsvm_model = SVC(kernel='precomputed') # ここでprecomputedカーネルを使用
    qsvm_model.fit(kernel_matrix_train, y_train)
    qsvm_train_time = time.time() - start_time_qsvc_train
    print(f"    QSVM Training Time: {qsvm_train_time:.4f} seconds ")

    # テストデータとトレーニングデータ間のカーネル行列を計算
    # evaluateメソッドにsamplerを渡す (x_test, x_trainの引数名を削除し、位置引数として渡す)
    kernel_matrix_test = quantum_kernel.evaluate(X_test, X_train)

    y_pred_qsvm = qsvm_model.predict(kernel_matrix_test)
    qsvm_accuracy = accuracy_score(y_test, y_pred_qsvm)

    print(f"    Accuracy: {qsvm_accuracy:.4f} ")

    # QSVMの決定境界の可視化
    # QuantumKernelを直接プロットすることは難しいため、
    # 学習済みのSVCモデル (`qsvm_model`) を使用してプロットします。
    # ここでは、SVCがprecomputedカーネルで学習されているため、
    # グリッドポイントに対してもカーネル行列を計算する必要があります。

    # 2次元グリッドを作成し、各点でモデルの予測を行う
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.5))

    # グリッドポイントの特徴量配列
    grid_points = np.c_[xx.ravel(), yy.ravel()]

    # グリッドポイントと訓練データ間のカーネル行列を計算
    # これは計算コストが高い場合がある
    # evaluateメソッドにsamplerを渡す (x_test, x_trainの引数名を削除し、位置引数として渡す)
    kernel_matrix_grid = quantum_kernel.evaluate(grid_points, X_train)

    Z_qsvm = qsvm_model.predict(kernel_matrix_grid)
    Z_qsvm = Z_qsvm.reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z_qsvm, alpha=0.8, cmap=plt.cm.coolwarm)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')
    plt.title(f'QSVM Decision Boundary for {dataset_name}')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(label='Class')
    plt.savefig(os.path.join(output_dir, f'{dataset_name}_qsvm_boundary.png'))
    plt.close()


if __name__ == "__main__":
    digits = load_digits() # 手書き数字データセットをロード

    # 実験1: 数字 3 と 4 
    run_classification_experiment(digits, [3, 4], "Digits_3_and_4")

    # 実験2: 数字 1 と 2
    run_classification_experiment(digits, [1, 2], "Digits_1_and_2")

    print("\n--- All experiments completed. Check the 'fig_svm' directory for plots. ---")