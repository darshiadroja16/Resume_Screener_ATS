import { useState } from "react";

export default function UploadForm({ onResults }) {
  const [jdText, setJdText] = useState("");
  const [files, setFiles] = useState([]);
  const [error, setError] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (!jdText.trim()) {
      setError("Please paste a job description.");
      return;
    }
    if (files.length === 0) {
      setError("Please upload at least one resume.");
      return;
    }

    try {
      const formData = new FormData();
      formData.append("jd_text", jdText);
      for (const file of files) formData.append("files", file);

      const res = await fetch("http://localhost:8000/rank", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok || !Array.isArray(data.results)) {
        setError(data.error || "Something went wrong while ranking.");
        onResults([]);
        return;
      }

      onResults(data.results);
    } catch {
      setError("Could not reach the server. Is the backend running?");
      onResults([]);
    }
  };

  return (
    <form onSubmit={handleSubmit}>
      {error && <p className="error">{error}</p>}
      <textarea
        placeholder="Paste job description"
        value={jdText}
        onChange={(e) => setJdText(e.target.value)}
      />
      <input
        type="file"
        multiple
        accept=".pdf,.docx"
        onChange={(e) => setFiles(e.target.files)}
      />
      <button type="submit">Rank Resumes</button>
    </form>
  );
}