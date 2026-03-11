import React from 'react';
import './ConfidenceBar.css';

/**
 * Renders a single denomination row with an animated confidence bar.
 *
 * Props:
 *   denomination  – display string, e.g. "20 GHS"
 *   percentage    – number 0-100
 *   isTop         – bool, highlights the winner row
 *   rank          – number (1-based), shown as a small badge
 */
function ConfidenceBar({ denomination, percentage, isTop, rank }) {
  const pct = Math.min(100, Math.max(0, percentage));

  // Gradient colour: gold-green for winner, scaled by confidence for others
  const barColor = isTop
    ? 'linear-gradient(90deg, #006B3F, #00a05a)'
    : pct > 50
    ? 'linear-gradient(90deg, #15803d, #22c55e)'
    : pct > 20
    ? 'linear-gradient(90deg, #ca8a04, #eab308)'
    : 'linear-gradient(90deg, #94a3b8, #cbd5e1)';

  return (
    <div className={`conf-row ${isTop ? 'conf-row--top' : ''}`}>
      {/* Rank badge */}
      <span className={`conf-rank ${isTop ? 'conf-rank--top' : ''}`}>
        {rank}
      </span>

      {/* Label */}
      <span className="conf-label">{denomination}</span>

      {/* Bar track */}
      <div className="conf-track">
        <div
          className="conf-fill"
          style={{ width: `${pct}%`, background: barColor }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>

      {/* Percentage */}
      <span className={`conf-pct ${isTop ? 'conf-pct--top' : ''}`}>
        {pct.toFixed(1)}%
      </span>
    </div>
  );
}

export default ConfidenceBar;
