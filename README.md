# assignment3_quantum_computing

## 🧪 開発環境セットアップ手順（Windows向け）

このプロジェクトでは、仮想環境を使用して Python パッケージの依存関係を分離しています。以下の手順に従って環境構築を行ってください。pythonはインストール済みである前提です。.gitignoreにvenv書いたけど除外できなかったのでvenvはリポジトリの外で作成しています。

### 1. Python環境の準備

- セットアップ手順:

  - プロジェクト用ディレクトリを作成して移動する:

    ```bash
    mkdir quantum_assignment
    cd quantum_assignment
    ```

  - 仮想環境を作成する:

    ```bash
    python -m venv venv
    ```

    - `venv` という名前の仮想環境が作成されます

  - 仮想環境を有効化する（Windowsの場合）:

    ```bash
    .\venv\Scripts\activate
    ```

    - コマンドプロンプトの先頭に `(venv)` と表示されれば成功です

---

### 2. clone

- 仮想環境を有効にした状態で以下を実行：

  ```bash
  git clone git@github.com:tanakaharuki0/assignment3_quantum_computing.git
  ```
  
### 2. packageのインストール
- 必要なパッケージを一括でインストール（WSL上では無理だったから環境依存かも？）：

  ```bash
  pip install -r .\assignment3_quantum_computing\requirements.txt
  ```

- 上の一括でのパッケージインストールが無理そうならこれでいけるはず（足りないのあるなら各自インストール）：

  ```bash
  pip install qiskit qiskit-aer matplotlib pylatexenc scikit-learn qiskit-machine-learning
  ```
  
