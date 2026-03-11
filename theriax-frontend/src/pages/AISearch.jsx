import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import Select from "react-select";
import { Card } from "../components/Card";
import WorkspaceShell from "../components/WorkspaceShell";
import api from "../utils/api";

const severityOptions = [
  { value: "low", label: "Low" },
  { value: "medium", label: "Medium" },
  { value: "high", label: "High" },
];

const menuPortalTarget = typeof document !== "undefined" ? document.body : null;

const customSelectStyles = {
  control: (provided, state) => ({
    ...provided,
    minHeight: "46px",
    borderRadius: "12px",
    backgroundColor: "rgba(255,255,255,0.9)",
    borderColor: state.isFocused ? "rgba(15,118,110,0.55)" : "rgba(19,36,54,0.17)",
    boxShadow: state.isFocused ? "0 0 0 4px rgba(15,118,110,0.24)" : "none",
    "&:hover": {
      borderColor: "rgba(15,118,110,0.7)",
    },
  }),
  placeholder: (provided) => ({
    ...provided,
    color: "#6a8193",
    fontSize: "0.92rem",
  }),
  menu: (provided) => ({
    ...provided,
    borderRadius: "12px",
    overflow: "hidden",
    border: "1px solid rgba(19,36,54,0.16)",
    boxShadow: "0 12px 28px rgba(10,34,51,0.16)",
  }),
  menuPortal: (provided) => ({
    ...provided,
    zIndex: 9999,
  }),
  option: (provided, state) => ({
    ...provided,
    backgroundColor: state.isSelected
      ? "rgba(15,118,110,0.2)"
      : state.isFocused
        ? "rgba(15,118,110,0.11)"
        : "white",
    color: "#143247",
    fontSize: "0.88rem",
  }),
  multiValue: (provided) => ({
    ...provided,
    backgroundColor: "rgba(15,118,110,0.18)",
    borderRadius: "999px",
    paddingLeft: "4px",
  }),
  multiValueLabel: (provided) => ({
    ...provided,
    color: "#0f5b55",
    fontWeight: 700,
  }),
};

function formatCurrency(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return value || "N/A";
  return `$${numericValue.toFixed(2)}`;
}

function formatConfidence(value) {
  const numericValue = Number(value);
  if (!Number.isFinite(numericValue)) return "N/A";
  return `${(numericValue * 100).toFixed(2)}%`;
}

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
        const [conditionsRes, symptomsRes] = await Promise.all([api.get("/conditions"), api.get("/symptoms")]);
        setConditions((conditionsRes.data || []).sort((a, b) => a.name.localeCompare(b.name)));
        setSymptomOptions((symptomsRes.data || []).sort());
      } catch {
        setRequestError("Failed to load dropdown data.");
      }
    };

    fetchDropdowns();
  }, []);

  const validateForm = () => {
    const nextErrors = {};
    if (!symptoms.length) nextErrors.symptoms = "Select at least one symptom.";
    if (!age || Number.isNaN(+age) || +age < 1) nextErrors.age = "Enter a valid age.";
    if (!weight || Number.isNaN(+weight) || +weight < 1) nextErrors.weight = "Enter a valid weight.";
    if (!condition) nextErrors.condition = "Select a condition.";
    setErrors(nextErrors);
    return Object.keys(nextErrors).length === 0;
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    setRequestError("");
    setAiResult(null);
    setSuggestedMeds([]);

    if (!validateForm()) return;

    setLoading(true);
    try {
      const aiResponse = await api.post("/ai-recommend", {
        symptoms: symptoms.map((selectedSymptom) => selectedSymptom.value).join(", "),
        age: +age,
        weight: +weight,
        condition,
        severity,
      });
      setAiResult(aiResponse.data);

      const suggestedResponse = await api.get("/medicines-by-condition", {
        params: { condition, severity },
      });
      setSuggestedMeds(suggestedResponse.data || []);
    } catch (error) {
      setRequestError(error.response?.data?.detail || "Failed to fetch suggestions.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <WorkspaceShell
      title="AI Recommendation Search"
      subtitle="Enter patient details and symptoms to generate model-backed medicine recommendations."
    >
      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <motion.section
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35 }}
        >
          <Card title="Case Input" subtitle="Define a clinical scenario and run AI recommendation search">
            <form onSubmit={handleSubmit} className="space-y-5">
              <div>
                <label className="theriax-label">Symptoms</label>
                <Select
                  styles={customSelectStyles}
                  options={symptomOptions.map((item) => ({ label: item, value: item }))}
                  value={symptoms}
                  onChange={(selected) => setSymptoms(selected || [])}
                  isMulti
                  placeholder="Select or search symptoms"
                  menuPortalTarget={menuPortalTarget}
                  menuPlacement="auto"
                />
                {errors.symptoms && <p className="theriax-error">{errors.symptoms}</p>}
              </div>

              <div className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="age" className="theriax-label">
                    Age
                  </label>
                  <input
                    id="age"
                    type="number"
                    value={age}
                    onChange={(event) => setAge(event.target.value)}
                    className={`theriax-input ${errors.age ? "theriax-input-error" : ""}`}
                    placeholder="Years"
                  />
                  {errors.age && <p className="theriax-error">{errors.age}</p>}
                </div>

                <div>
                  <label htmlFor="weight" className="theriax-label">
                    Weight (kg)
                  </label>
                  <input
                    id="weight"
                    type="number"
                    value={weight}
                    onChange={(event) => setWeight(event.target.value)}
                    className={`theriax-input ${errors.weight ? "theriax-input-error" : ""}`}
                    placeholder="Kilograms"
                  />
                  {errors.weight && <p className="theriax-error">{errors.weight}</p>}
                </div>
              </div>

              <div>
                <label className="theriax-label">Condition</label>
                <Select
                  styles={customSelectStyles}
                  options={conditions.map((item) => ({ label: item.name, value: item.name }))}
                  value={condition ? { label: condition, value: condition } : null}
                  onChange={(selected) => setCondition(selected?.value || "")}
                  placeholder="Select or search condition"
                  menuPortalTarget={menuPortalTarget}
                  menuPlacement="auto"
                />
                {errors.condition && <p className="theriax-error">{errors.condition}</p>}
              </div>

              <div>
                <label className="theriax-label">Severity</label>
                <div className="grid grid-cols-3 gap-2">
                  {severityOptions.map((option) => {
                    const isActive = severity === option.value;
                    return (
                      <button
                        key={option.value}
                        type="button"
                        onClick={() => setSeverity(option.value)}
                        className={`rounded-xl border px-3 py-2 text-sm font-semibold transition ${
                          isActive
                            ? "border-teal-600 bg-teal-600 text-white shadow-md shadow-teal-600/25"
                            : "border-slate-300 bg-white/80 text-slate-600 hover:border-teal-500 hover:text-teal-700"
                        }`}
                      >
                        {option.label}
                      </button>
                    );
                  })}
                </div>
              </div>

              <button type="submit" disabled={loading} className="theriax-btn theriax-btn-primary w-full">
                {loading ? "Generating recommendation..." : "Run AI Recommendation"}
              </button>

              {requestError && <div className="theriax-alert theriax-alert-error">{requestError}</div>}
            </form>
          </Card>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35, delay: 0.06 }}
          className="space-y-6"
        >
          <Card title="AI Recommendation" subtitle="Primary model output for the submitted case">
            {aiResult ? (
              <div className="space-y-3">
                <p className="theriax-display text-xl font-bold text-slate-900">{aiResult.ai_model || "N/A"}</p>
                <p className="theriax-muted text-sm">{aiResult.info || "No additional details provided."}</p>
                {aiResult.unknown_symptoms?.length > 0 && (
                  <div className="theriax-alert theriax-alert-error">
                    Unknown symptoms ignored: {aiResult.unknown_symptoms.join(", ")}
                  </div>
                )}
              </div>
            ) : (
              <p className="theriax-muted text-sm">No AI output yet. Submit the form to generate a recommendation.</p>
            )}
          </Card>

          {aiResult?.top_predictions?.length > 0 && (
            <Card title="Top Predictions" subtitle="Highest confidence model candidates">
              <ol className="space-y-2 text-sm">
                {aiResult.top_predictions.map((prediction, index) => (
                  <li
                    key={`${prediction.name}-${index}`}
                    className="flex items-center justify-between rounded-xl border border-slate-200 bg-white/75 px-3 py-2"
                  >
                    <span className="font-semibold text-slate-800">{prediction.name || "N/A"}</span>
                    <span className="rounded-full bg-teal-100 px-2 py-1 text-xs font-bold text-teal-800">
                      {formatConfidence(prediction.confidence)}
                    </span>
                  </li>
                ))}
              </ol>
            </Card>
          )}

          <Card title="Condition-Based Suggestions" subtitle="Rule-based medicines filtered by severity">
            {suggestedMeds.length > 0 ? (
              <ul className="space-y-2 text-sm">
                {suggestedMeds.map((medicine, index) => (
                  <li
                    key={`${medicine.name || "medicine"}-${medicine.dosage || "dose"}-${index}`}
                    className="rounded-xl border border-slate-200 bg-white/75 px-3 py-3"
                  >
                    <p className="font-semibold text-slate-800">{medicine.name || "Unknown medicine"}</p>
                    <p className="theriax-muted mt-1 text-xs">
                      Dosage: {medicine.dosage || "N/A"} | Cost: {formatCurrency(medicine.cost)}{" "}
                      {medicine.is_generic ? "| Generic" : ""}
                    </p>
                  </li>
                ))}
              </ul>
            ) : (
              <p className="theriax-muted text-sm">
                No suggestions yet. Run a search to view matching medicines for the selected condition.
              </p>
            )}
          </Card>
        </motion.section>
      </div>
    </WorkspaceShell>
  );
}
