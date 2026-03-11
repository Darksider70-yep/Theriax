export function FormInput({
  id,
  label,
  type = "text",
  placeholder,
  value,
  onChange,
  required = true,
  autoComplete,
  error,
}) {
  return (
    <div>
      {label && (
        <label htmlFor={id} className="theriax-label">
          {label}
        </label>
      )}
      <input
        id={id}
        className={`theriax-input ${error ? "theriax-input-error" : ""}`}
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        required={required}
        autoComplete={autoComplete}
      />
      {error && <p className="theriax-error">{error}</p>}
    </div>
  );
}
