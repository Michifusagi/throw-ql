# Throw Q-learning

2リンクの平面アームで、球を投げる・壁当てをする動作を学習するための簡易Streamlitアプリです。

深層強化学習は使わず、特徴量ベースのApproximate Q-learningと、壁当てモードではSARSA風の更新を使っています。

## セットアップ

```bash
cd /Users/moriyasu/dev/throw-ql-workspace/throw-ql
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 起動方法

```bash
streamlit run app.py
```

サイドバーの `mode` で、次の2つのモードを切り替えられます。

- `Throw distance`: 球を目標距離へ投げるモード
- `Wall rally`: 壁で反射して戻ってくる球をアーム先端でキャッチし続けるモード

## Throw Distance

距離投げモードでは、2自由度アームが球を投げ、指定した距離付近に着地させる動作を学習します。

操作できる主な項目:

- epsilon
- learning rate
- discount factor
- episode count
- target distance
- random seed
- demo form hint
- motion playback speed

現在の簡易物理モデルと速度制約では、安定して到達できる範囲を `1.0〜5.0m` に制限しています。

学習は、低レベルの関節操作を毎step自由に選ぶのではなく、`pull / coast / brake / release` からなる短いmotion primitive planを候補として用意し、その中からよいplanをApproximate Q-learningで選ぶ形にしています。

主なplan特徴量:

- `plan:target_distance`
- `plan:predicted_landing_error`
- `plan:predicted_landing_delta`
- `plan:release_progress`
- `plan:impulse`
- `plan:success_window`

`Draw greedy motion` と `Draw demo motion` は、重みを更新せずに動作シーケンスを描画します。

## Wall Rally

壁当てモードでは、球を壁に向かって投げ、壁で反射し、地面で1バウンドして戻ってくる球をアーム先端でキャッチします。キャッチに成功すると再び投げ、設定した目標回数まで繰り返します。

操作できる主な項目:

- epsilon
- learning rate
- discount factor
- episode count
- target catches
- random seed
- motion delay

`target catches` は `1〜10回` の範囲で設定できます。

壁は固定ではなく、一定範囲内をゆっくり動きます。そのため、固定壁よりも反射後の軌道が変わりやすく、連続キャッチは難しくなっています。

壁当てモードの主な状態特徴量:

- アームの角度・角速度
- 球の位置・速度
- 壁の位置・速度
- 球とアーム先端の相対位置
- 球と壁の相対距離
- バウンド済みか
- キャッチ可能区間か
- 現在のキャッチ回数
- episode内の経過step

壁当てモードのactionは、低レベルの関節入力ではなくcatch primitiveです。

- `track`: 球または待機点を追う
- `track-high`: 球より少し上を追う
- `track-low`: 球より少し下を追う

報酬には、アーム先端と球の距離が縮むことへの小報酬、キャッチ成功報酬、目標回数達成報酬、ミス時のペナルティを含めています。

## テスト

```bash
pytest
```

## ファイル構成

- `app.py`: Streamlit UIと描画
- `environment.py`: 距離投げモードのアーム物理、投射運動、報酬、特徴量
- `training.py`: 距離投げモードの学習、motion primitive plan、評価、描画用記録
- `wall_task.py`: 壁当てモードの物理、特徴量、SARSA風更新、評価、描画用記録
- `agent.py`: 特徴量ベースのApproximate Q-learning agentとweight保存/読み込み
- `tests/`: 主要処理のテスト

調整履歴は `DEVELOPMENT_NOTES.md` に簡潔に残しています。
