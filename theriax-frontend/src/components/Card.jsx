import { motion } from "framer-motion";

export function Card({ title, subtitle, children, className = "", variant = "default" }) {
  const baseClass = `theriax-card p-6 sm:p-7 ${className}`;

  const variants = {
    default: baseClass,
    glow: `${baseClass} theriax-surface-glow animate-pulse-glow`,
    gradient: `${baseClass} bg-gradient-to-br from-primary-50 via-white to-secondary-50 border-primary-200/40`,
  };

  const containerVariants = {
    hidden: { opacity: 0, y: 20 },
    show: {
      opacity: 1,
      y: 0,
      transition: { duration: 0.5, ease: "easeOut" }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0 },
    show: { opacity: 1, transition: { duration: 0.3 } }
  };

  return (
    <motion.section
      variants={containerVariants}
      initial="hidden"
      animate="show"
      className={variants[variant] || variants.default}
    >
      {title && (
        <motion.div variants={itemVariants} className="mb-5">
          <h3 className="theriax-display text-2xl font-bold bg-gradient-to-r from-primary-700 to-secondary-700 bg-clip-text text-transparent">
            {title}
          </h3>
          {subtitle && (
            <p className="theriax-muted mt-2 text-sm leading-relaxed">{subtitle}</p>
          )}
        </motion.div>
      )}
      <motion.div variants={itemVariants}>
        {children}
      </motion.div>
    </motion.section>
  );
}
  
