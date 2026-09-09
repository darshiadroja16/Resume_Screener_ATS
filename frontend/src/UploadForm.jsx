import { useState } from "react";

export default function UploadForm({ onResults }) {
  const [jdText, setJdText] = useState("");
  const [files, setFiles] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("jd_text", jdText);
    for (const file of files) formData.append("files", file);

    const res = await fetch("http://localhost:8000/rank", {
      method: "POST",
      body: formData,
    });
    const data = await res.json();
    onResults(data.results);
  };

  return (
    <form onSubmit={handleSubmit}>
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