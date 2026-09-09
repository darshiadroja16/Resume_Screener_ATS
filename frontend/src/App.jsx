import { useState } from "react";
import UploadForm from "./UploadForm";
import ResultsTable from "./ResultsTable";

export default function App() {
  const [results, setResults] = useState([]);

  return (
    <div>
      <h1>Resume Screener</h1>
      <UploadForm onResults={setResults} />
      <ResultsTable results={results} />
    </div>
  );
}