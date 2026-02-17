export function FormInput({ type, placeholder, value, onChange }) {
    return (
      <input
        className="w-full p-3 mb-4 border border-gray-300 rounded-lg focus:ring focus:ring-blue-200 dark:bg-gray-700 dark:border-gray-600 dark:text-white"
        type={type}
        placeholder={placeholder}
        value={value}
        onChange={onChange}
        required
      />
    );
}