# deutsch_algorithm.py

from qiskit import QuantumCircuit, transpile
from qiskit_aer import Aer
from qiskit.visualization import plot_histogram, plot_bloch_multivector
import numpy as np

# Quantum simulator backend
aer_sim = Aer.get_backend('aer_simulator')

def run_circuit_and_get_results(qc):
    """量子回路を実行し、測定結果のヒストグラムとBloch球の状態を返すヘルパー関数"""
    print(f"\n--- Running Circuit ---")
    print(qc.draw(output='text')) # 回路図をテキストで表示

    # 最後のハダマールゲート適用直前の状態 (q0) を取得
    # 注意: QiskitのStatevectorシミュレータは、回路全体の状態を取得するため、
    # 特定のゲートの直前の状態をピンポイントで取得するには、その時点までの回路を別に構築するか、
    # シミュレーションの途中で状態ベクトルをダンプする機能を使用する必要があります。
    # ここでは、簡略化のため、Hゲートの直前までを一度実行し、状態を取得します。
    # ただし、実際のBloch球の可視化は、測定前の状態全体に対して行います。

    # 回路をトランスパイル
    t_qc = transpile(qc, aer_sim)

    # ショット数を指定して回路を実行し、測定結果を取得
    results = aer_sim.run(t_qc, shots=1024).result()
    counts = results.get_counts()
    print(f"Measured counts: {counts}")

    # Bloch球の可視化のために、測定ゲートを除く回路をStatevectorシミュレータで実行
    # 測定ゲートは状態を崩壊させるため、その前の状態を見るために別途回路を準備
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements() # 最後の測定ゲートを削除

    # Statevectorシミュレータで状態ベクトルを取得
    state_vector_results = Aer.get_backend('statevector_simulator').run(qc_no_measure).result()
    state_vector = state_vector_results.get_statevector(qc_no_measure)
    print(f"Final state vector before measurement: {state_vector}")

    # Bloch球のプロット
    bloch_plot = plot_bloch_multivector(state_vector)

    return counts, bloch_plot

# --- Case 1: Constant function f(x) = 0 (no CNOT) ---
# Uf が何もしない場合 (常に0を出力)
# q0 |0> -- H -- (Uf) -- H -- M --
# q1 |1> -- X -- H -- (Uf) -- H --
def create_case1_circuit():
    qc = QuantumCircuit(2, 1) # 2量子ビット, 1古典ビット
    qc.x(1) # q1 を |1> に初期化
    qc.h(0) # q0 にHゲート
    qc.h(1) # q1 にHゲート
    # Uf の実装: 何もしない (f(x)=0)
    qc.h(0) # 最後のHゲート
    qc.measure(0, 0) # q0 を測定し、c0 に格納
    return qc

# --- Case 2: Constant function f(x) = 1 ---
# Uf が常に1を出力する場合 (Xゲートのみ)
# q0 |0> -- H -- (Uf: Xゲート) -- H -- M --
# q1 |1> -- X -- H -- (Uf) -- H --
def create_case2_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1) # q1 を |1> に初期化
    qc.h(0)
    qc.h(1)
    # Uf の実装: q0 にXゲート (f(x)=1)
    qc.x(0) # constant f(x)=1
    qc.h(0) # 最後のHゲート
    qc.measure(0, 0)
    return qc

# --- Case 3: Balanced function f(x) = x ---
# Uf が f(x)=x の場合 (CNOTゲート)
# q0 |0> -- H -- (Uf: CNOT (q0->q1)) -- H -- M --
# q1 |1> -- X -- H -- (Uf) -- H --
def create_case3_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1) # q1 を |1> に初期化
    qc.h(0)
    qc.h(1)
    # Uf の実装: q0 を制御、q1 をターゲットとするCNOT (f(x)=x)
    qc.cx(0, 1)
    qc.h(0) # 最後のHゲート
    qc.measure(0, 0)
    return qc

# --- Case 4: Balanced function f(x) = NOT x ---
# Uf が f(x)=NOT x の場合 (CNOTゲート + Xゲート)
# q0 |0> -- H -- (Uf: CNOT (q0->q1), X (q1)) -- H -- M --
# q1 |1> -- X -- H -- (Uf) -- H --
def create_case4_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1) # q1 を |1> に初期化
    qc.h(0)
    qc.h(1)
    # Uf の実装: q0 を制御、q1 をターゲットとするCNOTの後にq1にXゲート (f(x)=NOT x)
    qc.cx(0, 1)
    qc.x(1) # q1 にXゲートを追加
    qc.h(0) # 最後のHゲート
    qc.measure(0, 0)
    return qc

# --- 各ケースの実行と結果の表示 ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Case 1
    print("----- Case 1: Constant (f(x) = 0) -----")
    qc1 = create_case1_circuit()
    counts1, bloch1 = run_circuit_and_get_results(qc1)
    plt.figure(figsize=(8, 4))
    qc1.draw(output='mpl', filename='case1_circuit.png')
    bloch1.savefig('case1_bloch.png')
    plot_histogram(counts1, title="Case 1 Measurement Results").savefig('case1_histogram.png')
    print("--- 最後のHゲート適用前のq0とq1の状態 ---")
    # Statevectorシミュレータを使って、最後のHゲート適用直前の状態を計算
    qc1_pre_H = QuantumCircuit(2)
    qc1_pre_H.x(1)
    qc1_pre_H.h(0)
    qc1_pre_H.h(1)
    state_pre_H1 = Aer.get_backend('statevector_simulator').run(qc1_pre_H).result().get_statevector(qc1_pre_H)
    print(f"Pre-H state (Case 1): {state_pre_H1}")
    plot_bloch_multivector(state_pre_H1).savefig('case1_bloch_pre_H.png')

    # Case 2
    print("\n----- Case 2: Constant (f(x) = 1) -----")
    qc2 = create_case2_circuit()
    counts2, bloch2 = run_circuit_and_get_results(qc2)
    plt.figure(figsize=(8, 4))
    qc2.draw(output='mpl', filename='case2_circuit.png')
    bloch2.savefig('case2_bloch.png')
    plot_histogram(counts2, title="Case 2 Measurement Results").savefig('case2_histogram.png')
    print("--- 最後のHゲート適用前のq0とq1の状態 ---")
    qc2_pre_H = QuantumCircuit(2)
    qc2_pre_H.x(1)
    qc2_pre_H.h(0)
    qc2_pre_H.h(1)
    qc2_pre_H.x(0)
    state_pre_H2 = Aer.get_backend('statevector_simulator').run(qc2_pre_H).result().get_statevector(qc2_pre_H)
    print(f"Pre-H state (Case 2): {state_pre_H2}")
    plot_bloch_multivector(state_pre_H2).savefig('case2_bloch_pre_H.png')

    # Case 3
    print("\n----- Case 3: Balanced (f(x) = x) -----")
    qc3 = create_case3_circuit()
    counts3, bloch3 = run_circuit_and_get_results(qc3)
    plt.figure(figsize=(8, 4))
    qc3.draw(output='mpl', filename='case3_circuit.png')
    bloch3.savefig('case3_bloch.png')
    plot_histogram(counts3, title="Case 3 Measurement Results").savefig('case3_histogram.png')
    print("--- 最後のHゲート適用前のq0とq1の状態 ---")
    qc3_pre_H = QuantumCircuit(2)
    qc3_pre_H.x(1)
    qc3_pre_H.h(0)
    qc3_pre_H.h(1)
    qc3_pre_H.cx(0, 1)
    state_pre_H3 = Aer.get_backend('statevector_simulator').run(qc3_pre_H).result().get_statevector(qc3_pre_H)
    print(f"Pre-H state (Case 3): {state_pre_H3}")
    plot_bloch_multivector(state_pre_H3).savefig('case3_bloch_pre_H.png')

    # Case 4
    print("\n----- Case 4: Balanced (f(x) = NOT x) -----")
    qc4 = create_case4_circuit()
    counts4, bloch4 = run_circuit_and_get_results(qc4)
    plt.figure(figsize=(8, 4))
    qc4.draw(output='mpl', filename='case4_circuit.png')
    bloch4.savefig('case4_bloch.png')
    plot_histogram(counts4, title="Case 4 Measurement Results").savefig('case4_histogram.png')
    print("--- 最後のHゲート適用前のq0とq1の状態 ---")
    qc4_pre_H = QuantumCircuit(2)
    qc4_pre_H.x(1)
    qc4_pre_H.h(0)
    qc4_pre_H.h(1)
    qc4_pre_H.cx(0, 1)
    qc4_pre_H.x(1)
    state_pre_H4 = Aer.get_backend('statevector_simulator').run(qc4_pre_H).result().get_statevector(qc4_pre_H)
    print(f"Pre-H state (Case 4): {state_pre_H4}")
    plot_bloch_multivector(state_pre_H4).savefig('case4_bloch_pre_H.png')

    plt.show() # 全てのプロットを表示