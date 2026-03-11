export function Card({ title, subtitle, children, className = "" }) {
  return (
    <section className={`theriax-surface p-5 sm:p-6 ${className}`}>
      {title && (
        <div className="mb-4">
          <h3 className="theriax-display text-xl font-bold text-slate-900">{title}</h3>
          {subtitle && <p className="theriax-muted mt-1 text-sm">{subtitle}</p>}
        </div>
      )}
      {children}
    </section>
  );
}
  
