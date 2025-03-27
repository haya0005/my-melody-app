import React, { useState, useRef } from "react";

function Recorder() {
  const [isRecording, setIsRecording] = useState(false);
  const [audioURL, setAudioURL] = useState(null);
  const [statusMessage, setStatusMessage] = useState("");  // ステータスメッセージの状態を追加
  const mediaRecorderRef = useRef(null);
  const chunksRef = useRef([]);
  const blobRef = useRef(null); // ← 送信用に保存しておく

  const startRecording = async () => {
    setStatusMessage("");  // 録音開始時にメッセージをリセット
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    mediaRecorderRef.current = new MediaRecorder(stream);

    mediaRecorderRef.current.ondataavailable = (e) => {
      chunksRef.current.push(e.data);
    };

    mediaRecorderRef.current.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: "audio/wav" });
      const url = URL.createObjectURL(blob);
      setAudioURL(url);
      blobRef.current = blob;
      chunksRef.current = [];
      setStatusMessage("録音完了！");
    };

    mediaRecorderRef.current.start();
    setIsRecording(true);
    setStatusMessage("録音中...");

    setTimeout(() => {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }, 5000);
  };

  const sendToServer = async () => {
    if (!blobRef.current) return;

    setStatusMessage("変換中..."); // 変換中のメッセージを表示

    const formData = new FormData();
    formData.append("file", blobRef.current, "recorded.wav");

    const res = await fetch("http://localhost:5001/predict", {
      method: "POST",
      body: formData,
    });

    if (res.ok) {
      const midiBlob = await res.blob();
      const url = window.URL.createObjectURL(midiBlob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "output.mid";
      a.click();
      setStatusMessage("✅ MIDI変換完了！ダウンロード中...");
    } else {
      setStatusMessage("❌ 変換エラーが発生しました。もう一度試してください。");
    }
  };

  return (
    <div>
      <button onClick={startRecording} disabled={isRecording}>
        {isRecording ? "録音中..." : "録音開始（5秒）"}
      </button>

      {audioURL && (
        <div>
          <p>録音結果:</p>
          <audio controls src={audioURL}></audio>
          <br />
          <button onClick={sendToServer}>MIDIに変換してダウンロード</button>
        </div>
      )}

      {/* ステータスメッセージを表示 */}
      {statusMessage && <p>{statusMessage}</p>}
    </div>
  );
}

export default Recorder;
