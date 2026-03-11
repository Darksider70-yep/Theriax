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
      title="🚀 AI Recommendation Engine"
      subtitle="Enter patient details and symptoms to generate AI-powered medicine recommendations with clinical confidence scores."
    >
      {/* Premium Background Gradient Layer */}
      <div className="fixed -inset-0 -z-10 overflow-hidden pointer-events-none">
        <div className="absolute top-1/4 -right-48 w-96 h-96 bg-gradient-to-bl from-secondary-300/10 to-transparent rounded-full blur-3xl animate-float"></div>
        <div className="absolute bottom-0 -left-32 w-80 h-80 bg-gradient-to-tr from-tertiary-300/10 to-transparent rounded-full blur-3xl animate-float" style={{ animationDelay: "2s" }}></div>
        <div className="absolute top-1/2 left-1/2 w-72 h-72 bg-gradient-to-br from-primary-300/8 to-transparent rounded-full blur-3xl animate-float" style={{ animationDelay: "4s" }}></div>
      </div>
      <div className="grid gap-6 xl:grid-cols-[1.15fr_0.85fr]">
        <motion.section
          initial={{ opacity: 0, x: -10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35 }}
        >
          <Card title="📋 Patient Case Input" subtitle="Define clinical scenario parameters" variant="glow">
            <form onSubmit={handleSubmit} className="space-y-6">
              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.1 }}>
                <label className="theriax-label">🔍 Symptoms</label>
                <Select
                  styles={customSelectStyles}
                  options={symptomOptions.map((item) => ({ label: item, value: item }))}
                  value={symptoms}
                  onChange={(selected) => setSymptoms(selected || [])}
                  isMulti
                  placeholder="Search and select symptoms..."
                  menuPortalTarget={menuPortalTarget}
                  menuPlacement="auto"
                />
                {errors.symptoms && <p className="theriax-error">{errors.symptoms}</p>}
              </motion.div>

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.15 }} className="grid gap-4 sm:grid-cols-2">
                <div>
                  <label htmlFor="age" className="theriax-label">
                    👤 Age (years)
                  </label>
                  <input
                    id="age"
                    type="number"
                    value={age}
                    onChange={(event) => setAge(event.target.value)}
                    className={`theriax-input ${errors.age ? "theriax-input-error" : ""}`}
                    placeholder="Enter age"
                  />
                  {errors.age && <p className="theriax-error">{errors.age}</p>}
                </div>

                <div>
                  <label htmlFor="weight" className="theriax-label">
                    ⚖️ Weight (kg)
                  </label>
                  <input
                    id="weight"
                    type="number"
                    value={weight}
                    onChange={(event) => setWeight(event.target.value)}
                    className={`theriax-input ${errors.weight ? "theriax-input-error" : ""}`}
                    placeholder="Enter weight"
                  />
                  {errors.weight && <p className="theriax-error">{errors.weight}</p>}
                </div>
              </motion.div>

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.2 }}>
                <label className="theriax-label">🏥 Medical Condition</label>
                <Select
                  styles={customSelectStyles}
                  options={conditions.map((item) => ({ label: item.name, value: item.name }))}
                  value={condition ? { label: condition, value: condition } : null}
                  onChange={(selected) => setCondition(selected?.value || "")}
                  placeholder="Select condition..."
                  menuPortalTarget={menuPortalTarget}
                  menuPlacement="auto"
                />
                {errors.condition && <p className="theriax-error">{errors.condition}</p>}
              </motion.div>

              <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.25 }}>
                <label className="theriax-label">⚠️ Severity Level</label>
                <div className="grid grid-cols-3 gap-3">
                  {severityOptions.map((option) => {
                    const isActive = severity === option.value;
                    const colors = {
                      low: "border-green-400 bg-gradient-to-br from-green-50 to-green-100/50 text-green-700 shadow-lg shadow-green-200/40",
                      medium: "border-yellow-400 bg-gradient-to-br from-yellow-50 to-yellow-100/50 text-yellow-700 shadow-lg shadow-yellow-200/40",
                      high: "border-red-400 bg-gradient-to-br from-red-50 to-red-100/50 text-red-700 shadow-lg shadow-red-200/40",
                    };
                    return (
                      <motion.button
                        key={option.value}
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        type="button"
                        onClick={() => setSeverity(option.value)}
                        className={`rounded-xl border-2 px-4 py-3 font-semibold transition-all ${
                          isActive
                            ? colors[option.value]
                            : "border-slate-200 bg-white/60 text-slate-600 hover:border-primary-400 hover:bg-primary-50/40"
                        }`}
                      >
                        {option.label}
                      </motion.button>
                    );
                  })}
                </div>
              </motion.div>

              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                type="submit"
                disabled={loading}
                className="theriax-btn theriax-btn-primary w-full py-3 text-base font-semibold group"
              >
                <span>⚡</span>
                <span>{loading ? "Analyzing..." : "Generate Recommendation"}</span>
              </motion.button>

              {requestError && (
                <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} className="theriax-alert theriax-alert-error">
                  {requestError}
                </motion.div>
              )}
            </form>
          </Card>
        </motion.section>

        <motion.section
          initial={{ opacity: 0, x: 10 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.35, delay: 0.06 }}
          className="space-y-6"
        >
          <Card title="🧠 AI Recommendation" subtitle="Model-generated output" variant="gradient">
            {aiResult ? (
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} className="space-y-4">
                <div className="rounded-xl bg-gradient-to-br from-primary-50/80 to-secondary-50/60 border border-primary-200/40 p-4">
                  <p className="theriax-display text-2xl font-bold bg-gradient-to-r from-primary-700 to-secondary-700 bg-clip-text text-transparent">
                    {aiResult.ai_model || "N/A"}
                  </p>
                </div>
                <p className="theriax-muted text-sm leading-relaxed">{aiResult.info || "No additional details provided."}</p>
                {aiResult.unknown_symptoms?.length > 0 && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="theriax-alert theriax-alert-error">
                    ⚠️ Unknown symptoms ignored: {aiResult.unknown_symptoms.join(", ")}
                  </motion.div>
                )}
              </motion.div>
            ) : (
              <div className="text-center py-8">
                <p className="text-3xl mb-2">🔍</p>
                <p className="theriax-muted text-sm">Submit the form to generate AI recommendation</p>
              </div>
            )}
          </Card>

          {aiResult?.top_predictions?.length > 0 && (
            <Card title="🎯 Top Predictions" subtitle="Highest confidence candidates" variant="glow">
              <motion.ol className="space-y-2" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {aiResult.top_predictions.map((prediction, index) => (
                  <motion.li
                    key={`${prediction.name}-${index}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.1 }}
                    className="flex items-center justify-between rounded-xl bg-gradient-to-r from-primary-50/80 to-secondary-50/40 border border-primary-200/40 px-4 py-3 hover:border-primary-300/60 hover:shadow-lg transition-all"
                  >
                    <span className="font-semibold text-slate-800 flex items-center gap-2">
                      <span className="text-sm font-bold text-primary-600">#{index + 1}</span>
                      {prediction.name || "N/A"}
                    </span>
                    <span className="rounded-full bg-gradient-to-r from-primary-500 to-secondary-500 px-3 py-1 text-xs font-bold text-white shadow-lg shadow-primary-200/40">
                      {formatConfidence(prediction.confidence)}
                    </span>
                  </motion.li>
                ))}
              </motion.ol>
            </Card>
          )}

          <Card title="💊 Medicine Suggestions" subtitle="Condition and severity-matched options">
            {suggestedMeds.length > 0 ? (
              <motion.ul className="space-y-3" initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                {suggestedMeds.map((medicine, index) => (
                  <motion.li
                    key={`${medicine.name || "medicine"}-${medicine.dosage || "dose"}-${index}`}
                    initial={{ opacity: 0, x: -10 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: index * 0.05 }}
                    className="rounded-xl bg-gradient-to-r from-secondary-50/60 to-tertiary-50/40 border border-secondary-200/30 px-4 py-4 hover:shadow-md hover:border-secondary-300/50 transition-all"
                  >
                    <p className="font-bold text-slate-900 flex items-center gap-2">
                      💊 {medicine.name || "Unknown medicine"}
                      {medicine.is_generic && <span className="text-xs font-semibold bg-tertiary-200/40 text-tertiary-700 px-2 py-0.5 rounded-full">Generic</span>}
                    </p>
                    <p className="theriax-muted mt-2 text-xs flex items-center gap-4">
                      <span>📋 Dosage: <span className="text-slate-700 font-semibold">{medicine.dosage || "N/A"}</span></span>
                      <span>💰 Cost: <span className="text-slate-700 font-semibold">{formatCurrency(medicine.cost)}</span></span>
                    </p>
                  </motion.li>
                ))}
              </motion.ul>
            ) : (
              <div className="text-center py-8">
                <p className="text-2xl mb-2">💊</p>
                <p className="theriax-muted text-sm">Run a search to view medicine suggestions</p>
              </div>
            )}
          </Card>
        </motion.section>
      </div>
    </WorkspaceShell>
  );
}
