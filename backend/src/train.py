import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.utils.data as data

from utils import voc_single_track
from model import Seq2SeqTranscriber
from dataset import CustomDataset
from utils import token_seg_list_to_midi, unpack_sequence    
from eval import evaluate_midi

class LitTranscriber(pl.LightningModule):
    def __init__(
        self,
        transcriber_args: dict, # Seq2SeqTranscriber モジュールの初期化に必要な引数を含む辞書. 例えばn_melsやn_fft, hop_lengthなど
        lr: float, # 学習率（Learning Rate）。
        lr_decay: float = 1.0, # 学習率の減衰率。デフォルト値は 1.0（減衰なし）。
        lr_decay_interval: int = 1, # 学習率の減衰間隔（エポック単位）。
    ):
        super().__init__()
        self.save_hyperparameters() # 渡された引数をハイパーパラメータとして保存（PyTorch Lightning機能）。
        self.voc_dict = voc_single_track # voc_dict をインスタンス変数に保存。
        self.n_voc = self.voc_dict["n_voc"] # トークンの総数（n_voc）も保存。

        # 渡された引数 transcriber_args を展開して Seq2SeqTranscriber モジュールを初期化。トークン辞書（voc_dict）を追加で渡す。
        self.transcriber = Seq2SeqTranscriber(
            **transcriber_args, voc_dict=self.voc_dict
        )
        # 引数から渡された学習率や減衰関連の設定を保存。
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_interval = lr_decay_interval

    # 推論時に Seq2SeqTranscriber の推論機能（infer メソッド）を呼び出す関数
    def forward(self, y: torch.Tensor):
        transcriber_infer = self.transcriber.infer(y) # 入力波形（y）を Seq2SeqTranscriber の infer メソッドに渡して推論を実行。
        return transcriber_infer #推論結果として生成されたトークン列を返します。

    # モデルのトレーニング中に1バッチの処理を実行する関数
    def training_step(self, batch, batch_idx):
        y, t = batch # 入力波形（y）と正解トークン列（t）を分割。
        tf_out = self.transcriber(y, t) # 入力データを Seq2SeqTranscriber に渡してtranscriberのforwardで損失値（loss）を取得。出力は、損失値やロジット（各トークンのスコア）を含むオブジェクト。
        loss = tf_out.loss # T5モデルの出力に含まれる損失値（loss）。学習時の損失計算は、CrossEntropyLoss に基づいています。
        t = t.detach() # 正解トークン（t）を計算グラフから切り離して、後続の計算が逆伝播に影響しないようにする。
        mask = t != self.voc_dict["pad"] # 正解トークン列のうち、パディングトークン（pad）を無視するためのマスクを作成。

        # モデルの出力（logits）から最も高いスコアのトークン（argmax(-1)）を取得し、マスクされた正解トークン（t[mask]）と比較。
        # 正しく予測されたトークンの数を、マスクされたトークン全体の数で割って精度を計算。
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        #損失値（loss）と精度（accr）を PyTorch Lightning のログ機能を使って記録。
        self.log("train_loss", loss)
        self.log("train_accr", accr)
        return loss # 計算された損失を返します。この値はPyTorch Lightningがバックプロパゲーションを行うために使用します。

    # モデルの検証時に1バッチの処理を実行する関数
    def validation_step(self, batch, batch_idx):
        assert not self.transcriber.training # モデルがトレーニングモードでないことを確認
        y, t = batch
        tf_out = self.transcriber(y, t)
        loss = tf_out.loss
        t = t.detach()
        mask = t != self.voc_dict["pad"]
        accr = (tf_out.logits.argmax(-1)[mask] == t[mask]).sum() / mask.sum()
        self.log("vali_loss", loss)
        self.log("vali_accr", accr)
        return loss

    def test_step(self, batch, batch_idx):
        y, ref_midi = batch # 音声データ（y）と参照MIDI（ref_midi）を分割
        # テストデータセットでは、通常1つのサンプルを1バッチとして処理するため、インデックス0の要素を取り出します。
        y = y[0]
        ref_midi = ref_midi[0]
        with torch.no_grad(): # 推論中は勾配計算が不要なため、この文脈を使ってメモリ消費を削減。
            est_tokens = self.forward(y) # 入力音声 y を用いてモデルに推論を実行し、MIDIトークン列（est_tokens）を生成。モデルの forward メソッドを呼び出して推論を実行。この時forward内では分割されて実行される, 形状： (セグメント数, 最大トークン長)
            unpadded_tokens = unpack_sequence(est_tokens.cpu().numpy()) # 推論結果（est_tokens）からパディングを取り除き、有効なトークン列だけを取得。GPU上のテンソルをCPUに移動してNumPy配列に変換
            unpadded_tokens = [t[1:] for t in unpadded_tokens] # 各トークン列から開始トークン（<sos>など）を除外。
            est_midi = token_seg_list_to_midi(unpadded_tokens) # パディングを除去したトークン列（unpadded_tokens）をMIDI形式に変換。
        dict_eval = evaluate_midi(est_midi, ref_midi) # 参照MIDI（ref_midi）と推論MIDI（est_midi）を比較し、性能を評価。
        # 評価結果をPyTorch Lightningのログ機能で記録。
        dict_log = {}
        for key in dict_eval:
            dict_log["test/" + key] = dict_eval[key]
        self.log_dict(dict_log, batch_size=1)

    def train_dataloader(self):
        # 事前に定義されたデータセットクラス。音声（波形データ）とMIDIファイルをペアとしてロードします。
        dset = CustomDataset(
            split="train",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=False
        )
        # データローダーの作成
        return data.DataLoader(
            dataset=dset, # 使用するデータセットとして、初期化済みの Maestro を指定。
            collate_fn=dset.collate_batch, # バッチ処理時のデータ整形関数。
            batch_size=64, # 1バッチに含めるサンプル数。
            shuffle=True, # データセットをランダムにシャッフル。
            pin_memory=True, # データをピン留め（ページロック）してメモリに保存。
            num_workers=32, # データローディングに使用するプロセス数。
        )

    def val_dataloader(self):
        # 事前に定義されたデータセットクラス。音声（波形データ）とMIDIファイルをペアとしてロードします。
        dset = CustomDataset(
            split="valid",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=False
        )
        # データローダーの作成
        return data.DataLoader(
            dataset=dset, # 使用するデータセットとして、初期化済みの Maestro を指定。
            collate_fn=dset.collate_batch, # バッチ処理時のデータ整形関数。
            batch_size=32, # 1バッチに含めるサンプル数。
            shuffle=True, # データセットをランダムにシャッフル。
            pin_memory=True, # データをピン留め（ページロック）してメモリに保存。
            num_workers=16, # データローディングに使用するプロセス数。
        )
    def test_dataloader(self):
        dset = CustomDataset(
            split="test",                      # データセットの分割
            sample_rate=16000,
            apply_pedal=True,
            whole_song=True
        )
        return data.DataLoader(
            dataset=dset,
            collate_fn=dset.collate_wholesong,
            batch_size=1,
            shuffle=False, # テストデータの順序を固定。理由: 再現性を確保するため。
            pin_memory=True,
        )

    # モデルの学習時に、損失関数の値を最小化するようにパラメータを更新するオプティマイザ（最適化手法）を指定する関数
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    # Adamオプティマイザ の改良版であり、L2正則化の代わりに ウェイトデカイ（Weight Decay） を導入したオプティマイザ。
    # self.parameters(): モデルの学習可能なすべてのパラメータ（重みとバイアス）を取得。

from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

# チェックポイント保存の設定
if __name__ == "__main__":
    checkpoint_callback = ModelCheckpoint(
        dirpath="/content/drive/MyDrive/B4/Humtrans/m2m100/checkpoints",  # 保存先
        filename="epoch{epoch:02d}-vali_loss{vali_loss:.2f}",  # ファイル名フォーマット
        save_top_k=3,  # 上位3つのチェックポイントを保存
        monitor="vali_loss",  # 訓練データの損失を監視
        mode="min",  # 損失が小さいほど良いモデル
        every_n_epochs=1,  # 1エポックごとに保存
    )

    # ロガー設定
    logger = TensorBoardLogger(
        save_dir="/content/drive/MyDrive/B4/Humtrans/m2m100/logs",
        name="transcription"
    )

    # トレーナーの設定
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],  # チェックポイントコールバックを追加
        enable_checkpointing=True,
        accelerator="gpu",
        devices=1,
        max_epochs=400,
    )

    # モデルの引数
    args = {
        "n_mels": 128,
        "sample_rate": 16000,
        "n_fft": 1024,
        "hop_length": 128,
    }

    lightning_module = LitTranscriber(
        transcriber_args=args,
        lr=1e-4, # 0.0001
        lr_decay=0.99, # 各エポックの度に0.99倍して減衰させる
    )

    # 学習の開始
    trainer.fit(lightning_module)


