export default function ResultsTable({ results }) {
  if (!results.length) return null;

  return (
    <table>
      <thead>
        <tr>
          <th>ID</th>
          <th>Final Score</th>
          <th>Semantic</th>
          <th>Keyword</th>
          <th>Missing Keywords</th>
        </tr>
      </thead>
      <tbody>
        {results.map((r) => (
          <tr key={r.id}>
            <td>{r.id}</td>
            <td>{r.final_score}</td>
            <td>{r.semantic_score}</td>
            <td>{r.keyword_score}</td>
            <td>{r.missing_keywords.join(", ")}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}