import { Navigate, Route, Routes } from "react-router-dom";
import { ControlPage } from "./pages/ControlPage";
import { OutputPage } from "./pages/OutputPage";
import { HomePage } from "./pages/HomePage";

export function App() {
  return (
    <Routes>
      <Route path="/" element={<HomePage />} />
      <Route path="/control" element={<ControlPage />} />
      <Route path="/output" element={<OutputPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
