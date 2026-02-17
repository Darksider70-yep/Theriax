import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Select from "react-select";
import api from "../utils/api";

const MotionForm = motion.form;

const customSelectStyles = {
  control: (provided) => ({
    ...provided,
    backgroundColor: "#f9fafb",
    borderColor: "#d1d5db",
    borderRadius: "0.5rem",
    boxShadow: "none",
    "&:hover": {
      borderColor: "#2563eb",
    },
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isFocused ? "#e0f2fe" : "white",
    color: "#111827",
    fontSize: "0.875rem",
  }),
  multiValue: (provided) => ({
    ...provided,
    backgroundColor: "#dbeafe",
    borderRadius: "9999px",
    padding: "2px 6px",
  }),
  multiValueLabel: (provided) => ({
    ...provided,
    color: "#1e3a8a",
    fontWeight: "500",
  }),
  menu: (provided) => ({
    ...provided,
    zIndex: 9999,
  }),
};

export default function AISearch() {
  const [symptoms, setSymptoms] = useState([]);
  const [age, setAge] = useState("");
  const [weight, setWeight] = useState("");
  const [severity, setSeverity] = useState("medium");
  const [condition, setCondition] = useState("");
  const [conditions, setConditions] = useState([]);
  const [symptomOptions, setSymptomOptions] = useState([]);
  const [errors, setErrors] = useState({});
  const [aiResult, setAiResult] = useState(null);
  const [suggestedMeds, setSuggestedMeds] = useState([]);
  const [loading, setLoading] = useState(false);
  const [requestError, setRequestError] = useState("");

  useEffect(() => {
    const fetchDropdowns = async () => {
      try {
        const [conditionsRes, symptomsRes] = await Promise.all([
          api.get("/conditions"),
          api.get("/symptoms"),
        ]);
        setConditions((conditionsRes.data || []).sort((a, b) => a.name.localeCompare(b.name)));
        setSymptomOptions((symptomsRes.data || []).sort());
      } catch {
        setRequestError("Failed to load dropdown data.");
      }
    };

    fetchDropdowns();
  }, []);

  const validateForm = () => {
    const newErrors = {};
    if (!symptoms.length) newErrors.symptoms = "Please select at least one symptom.";
    if (!age || Number.isNaN(+age) || +age < 1) newErrors.age = "Enter a valid age.";
    if (!weight || Number.isNaN(+weight) || +weight < 1) newErrors.weight = "Enter a valid weight.";
    if (!condition) newErrors.condition = "Please select a condition.";
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setRequestError("");
    setAiResult(null);
    setSuggestedMeds([]);

    if (!validateForm()) return;

    setLoading(true);
    try {
      const resAI = await api.post("/ai-recommend", {
        symptoms: symptoms.map((s) => s.value).join(", "),
        age: +age,
        weight: +weight,
        condition,
        severity,
      });
      setAiResult(resAI.data);

      const resSuggested = await api.get("/medicines-by-condition", {
        params: { condition, severity },
      });
      setSuggestedMeds(resSuggested.data || []);
    } catch (err) {
      setRequestError(err.response?.data?.detail || "Failed to fetch suggestions.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto p-6 w-full">
      <MotionForm
        onSubmit={handleSubmit}
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-white p-6 rounded-xl shadow space-y-4"
      >
        <h2 className="text-xl font-bold">AI Medicine Recommendation</h2>

        <div className="mb-3">
          <label className="block mb-1 font-medium">Symptoms</label>
          <Select
            styles={customSelectStyles}
            options={symptomOptions.map((s) => ({ label: s, value: s }))}
            value={symptoms}
            onChange={(selected) => setSymptoms(selected || [])}
            isMulti
            placeholder="Select or search symptoms"
            menuPlacement="auto"
            menuShouldScrollIntoView
            menuPortalTarget={document.body}
          />
          {errors.symptoms && <p className="text-red-500 text-sm">{errors.symptoms}</p>}
        </div>

        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="block mb-1 font-medium">Age</label>
            <input
              type="number"
              value={age}
              onChange={(e) => setAge(e.target.value)}
              className="w-full border rounded p-2"
            />
            {errors.age && <p className="text-red-500 text-sm">{errors.age}</p>}
          </div>
          <div>
            <label className="block mb-1 font-medium">Weight (kg)</label>
            <input
              type="number"
              value={weight}
              onChange={(e) => setWeight(e.target.value)}
              className="w-full border rounded p-2"
            />
            {errors.weight && <p className="text-red-500 text-sm">{errors.weight}</p>}
          </div>
        </div>

        <div className="mt-4">
          <label className="block mb-1 font-medium">Condition</label>
          <Select
            styles={customSelectStyles}
            options={conditions.map((c) => ({ label: c.name, value: c.name }))}
            value={condition ? { label: condition, value: condition } : null}
            onChange={(selected) => setCondition(selected?.value || "")}
            placeholder="Select or search condition"
            menuPlacement="auto"
            menuShouldScrollIntoView
            menuPortalTarget={document.body}
          />
          {errors.condition && <p className="text-red-500 text-sm">{errors.condition}</p>}
        </div>

        <div>
          <label className="block mb-1 font-medium">Severity</label>
          <select
            value={severity}
            onChange={(e) => setSeverity(e.target.value)}
            className="w-full border rounded p-2"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
          </select>
        </div>

        <button
          type="submit"
          disabled={loading}
          className="mt-4 w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
        >
          {loading ? "Loading..." : "Submit"}
        </button>
      </MotionForm>

      {requestError && <p className="mt-4 text-sm text-red-600">{requestError}</p>}

      {suggestedMeds.length > 0 && (
        <div className="mt-6 bg-blue-50 border border-blue-300 p-4 rounded">
          <h3 className="text-lg font-bold mb-2">Condition-Based Suggestions</h3>
          <ul className="list-disc ml-6 text-sm text-gray-700">
            {suggestedMeds.map((med) => (
              <li key={`${med.name}-${med.dosage}`}>
                <strong>{med.name}</strong> - {med.dosage}, {med.cost} {med.is_generic ? "(Generic)" : ""}
              </li>
            ))}
          </ul>
        </div>
      )}

      {aiResult && (
        <div className="mt-6 bg-green-50 border border-green-300 p-4 rounded">
          <h3 className="text-lg font-bold mb-2">AI Recommended Medicine</h3>
          {aiResult?.top_predictions?.length > 0 && (
            <div className="mt-6 bg-yellow-50 border border-yellow-300 p-4 rounded">
              <h3 className="text-lg font-bold mb-2">Top 3 AI Predictions</h3>
              <ul className="list-decimal ml-6 text-sm text-gray-800">
                {aiResult.top_predictions.map((prediction) => (
                  <li key={prediction.name}>
                    {prediction.name} - <span className="text-blue-600 font-medium">{(prediction.confidence * 100).toFixed(2)}%</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          <p className="text-lg font-semibold">{aiResult.ai_model}</p>
          <p className="text-sm text-gray-600 mt-1">{aiResult.info}</p>
          {aiResult.unknown_symptoms?.length > 0 && (
            <p className="text-xs text-amber-700 mt-2">Unknown symptoms ignored: {aiResult.unknown_symptoms.join(", ")}</p>
          )}
        </div>
      )}
    </div>
  );
}
