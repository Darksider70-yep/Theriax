export function Card({ title, children, className = "" }) {
    return (
      <div className={`bg-white dark:bg-gray-800 rounded-2xl p-6 shadow-md ${className}`}>
        {title && <h2 className="text-xl font-semibold mb-4">{title}</h2>}
        {children}
      </div>
    );
}
  