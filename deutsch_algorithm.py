# deutsch_algorithm.py

from qiskit import QuantumCircuit
from qiskit.primitives import Sampler
from qiskit.visualization import plot_histogram, plot_bloch_multivector
from qiskit_aer import Aer
import os
import matplotlib.pyplot as plt

# Samplerプリミティブを使用
sampler = Sampler()

# figディレクトリが存在しない場合は作成
output_dir = 'fig_deutsch'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def run_circuit_and_get_results(qc):
    """量子回路を実行し、測定結果のヒストグラムとBloch球の状態を返すヘルパー関数"""
    print(f"\n--- Running Circuit ---")
    print(qc.draw(output='text'))

    # Samplerを使用して回路を実行し、結果（確率分布）を取得
    # Samplerは測定ゲートを自動的に扱います
    job = sampler.run(qc)
    result = job.result()
    quasi_dist = result.quasi_dists[0]
    print(f"Measured quasi-distribution: {quasi_dist}")
    
    # Bloch球の可視化のために、測定ゲートを除く回路をStatevectorシミュレータで実行
    qc_no_measure = qc.copy()
    qc_no_measure.remove_final_measurements()
    state_vector_results = Aer.get_backend('statevector_simulator').run(qc_no_measure).result()
    state_vector = state_vector_results.get_statevector(qc_no_measure)
    print(f"Final state vector before measurement: {state_vector}")

    # Bloch球のプロット
    bloch_plot = plot_bloch_multivector(state_vector)

    return quasi_dist, bloch_plot

# --- Case 1: Constant function f(x) = 0 (assignment3.pdf 図1(a)) ---
def create_case1_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.h([0, 1])
    qc.barrier()  # オラクル前のバリア
    # Uf の実装: 何もしない (f(x)=0)
    qc.barrier()  # オラクル後のバリア
    qc.h(0)
    qc.measure(0, 0)
    return qc

# --- Case 2: Balanced function f(x) = x (assignment3.pdf 図1(b)) ---
def create_case2_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.h([0, 1])
    qc.barrier()  # オラクル前のバリア
    # Uf の実装: q0 を制御、q1 をターゲットとするCNOT (f(x)=x)
    qc.cx(0, 1)
    qc.barrier()  # オラクル後のバリア
    qc.h(0)
    qc.measure(0, 0)
    return qc

# --- Case 3: Balanced function f(x) = NOT x (assignment3.pdf 図1(c)) ---
def create_case3_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.h([0, 1])
    qc.barrier()  # オラクル前のバリア
    # Uf の実装: q0 から q1 へのCNOTの後にq1にX (f(x)=NOT x)
    qc.cx(0, 1)
    qc.x(1)
    qc.barrier()  # オラクル後のバリア
    qc.h(0)
    qc.measure(0, 0)
    return qc

# --- Case 4: Constant function f(x) = 1 (assignment3.pdf 図1(d)) ---
def create_case4_circuit():
    qc = QuantumCircuit(2, 1)
    qc.x(1)
    qc.h([0, 1])
    qc.barrier()  # オラクル前のバリア
    # Uf の実装: 図1(d)に忠実な実装
    qc.cx(0, 1)
    qc.x(1)
    qc.cx(0, 1)
    qc.barrier()  # オラクル後のバリア
    qc.h(0)
    qc.measure(0, 0)
    return qc

# --- 各ケースの実行と結果の表示 ---
if __name__ == "__main__":
    
    plot_output_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(plot_output_dir):
        os.makedirs(plot_output_dir)

    def plot_and_save_histogram(quasi_dist, title, filename):
        fig = plot_histogram(quasi_dist, title=title)
        ax = fig.gca()
        ax.set_xlim([-0.5, 1.5])
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['0', '1'])
        fig.savefig(os.path.join(plot_output_dir, filename))
        plt.close(fig)

    # ケース1
    print("----- Case 1: Constant (f(x) = 0) -----")
    qc1 = create_case1_circuit()
    quasi_dist1, bloch1 = run_circuit_and_get_results(qc1)
    qc1.draw(output='mpl', filename=os.path.join(plot_output_dir, 'case1_circuit.jpg'))
    bloch1.savefig(os.path.join(plot_output_dir, 'case1_bloch.jpg'))
    plot_and_save_histogram(quasi_dist1, "Case 1 Measurement Results", 'case1_histogram.jpg')
    
    qc1_pre_H = QuantumCircuit(2)
    qc1_pre_H.x(1)
    qc1_pre_H.h([0, 1])
    state_pre_H1 = Aer.get_backend('statevector_simulator').run(qc1_pre_H).result().get_statevector(qc1_pre_H)
    plot_bloch_multivector(state_pre_H1).savefig(os.path.join(plot_output_dir, 'case1_bloch_pre_H.jpg'))

    # ケース2
    print("\n----- Case 2: Balanced (f(x) = x) -----")
    qc2 = create_case2_circuit()
    quasi_dist2, bloch2 = run_circuit_and_get_results(qc2)
    qc2.draw(output='mpl', filename=os.path.join(plot_output_dir, 'case2_circuit.jpg'))
    bloch2.savefig(os.path.join(plot_output_dir, 'case2_bloch.jpg'))
    plot_and_save_histogram(quasi_dist2, "Case 2 Measurement Results", 'case2_histogram.jpg')
    
    qc2_pre_H = QuantumCircuit(2)
    qc2_pre_H.x(1)
    qc2_pre_H.h([0, 1])
    qc2_pre_H.cx(0, 1)
    state_pre_H2 = Aer.get_backend('statevector_simulator').run(qc2_pre_H).result().get_statevector(qc2_pre_H)
    plot_bloch_multivector(state_pre_H2).savefig(os.path.join(plot_output_dir, 'case2_bloch_pre_H.jpg'))

    # ケース3
    print("\n----- Case 3: Balanced (f(x) = NOT x) -----")
    qc3 = create_case3_circuit()
    quasi_dist3, bloch3 = run_circuit_and_get_results(qc3)
    qc3.draw(output='mpl', filename=os.path.join(plot_output_dir, 'case3_circuit.jpg'))
    bloch3.savefig(os.path.join(plot_output_dir, 'case3_bloch.jpg'))
    plot_and_save_histogram(quasi_dist3, "Case 3 Measurement Results", 'case3_histogram.jpg')
    
    qc3_pre_H = QuantumCircuit(2)
    qc3_pre_H.x(1)
    qc3_pre_H.h([0, 1])
    qc3_pre_H.cx(0, 1)
    qc3_pre_H.x(1)
    state_pre_H3 = Aer.get_backend('statevector_simulator').run(qc3_pre_H).result().get_statevector(qc3_pre_H)
    plot_bloch_multivector(state_pre_H3).savefig(os.path.join(plot_output_dir, 'case3_bloch_pre_H.jpg'))

    # ケース4
    print("\n----- Case 4: Constant (f(x) = 1) -----")
    qc4 = create_case4_circuit()
    quasi_dist4, bloch4 = run_circuit_and_get_results(qc4)
    qc4.draw(output='mpl', filename=os.path.join(plot_output_dir, 'case4_circuit.jpg'))
    bloch4.savefig(os.path.join(plot_output_dir, 'case4_bloch.jpg'))
    plot_and_save_histogram(quasi_dist4, "Case 4 Measurement Results", 'case4_histogram.jpg')
    
    qc4_pre_H = QuantumCircuit(2)
    qc4_pre_H.x(1)
    qc4_pre_H.h([0, 1])
    qc4_pre_H.cx(0, 1)
    qc4_pre_H.x(1)
    qc4_pre_H.cx(0, 1)
    state_pre_H4 = Aer.get_backend('statevector_simulator').run(qc4_pre_H).result().get_statevector(qc4_pre_H)
    plot_bloch_multivector(state_pre_H4).savefig(os.path.join(plot_output_dir, 'case4_bloch_pre_H.jpg'))

    print("\nAll plots have been saved to the 'fig_deutsch/plots' directory.")