import React from 'react';
import ConfidenceBar from './ConfidenceBar';
import './ResultDisplay.css';

const DENOM_NAMES = {
  '1':   'One Ghana Cedi',
  '2':   'Two Ghana Cedis',
  '5':   'Five Ghana Cedis',
  '10':  'Ten Ghana Cedis',
  '20':  'Twenty Ghana Cedis',
  '50':  'Fifty Ghana Cedis',
  '100': 'One Hundred Ghana Cedis',
  '200': 'Two Hundred Ghana Cedis',
};

function formatAmount(raw) {
  if (!raw) return '';
  return raw.replace('_GHS', '').replace(' GHS', '');
}

function getDenomName(raw) {
  const num = formatAmount(raw);
  return DENOM_NAMES[num] || `${num} Ghana Cedis`;
}

function confidenceLevel(pct) {
  if (pct >= 90) return { label: 'Very High', color: '#006B3F', meter: 'linear-gradient(90deg,#006B3F,#00a05a)' };
  if (pct >= 75) return { label: 'High',      color: '#15803d', meter: 'linear-gradient(90deg,#15803d,#22c55e)' };
  if (pct >= 55) return { label: 'Moderate',  color: '#ca8a04', meter: 'linear-gradient(90deg,#ca8a04,#FCD116)' };
  return               { label: 'Low',        color: '#CE1126', meter: 'linear-gradient(90deg,#CE1126,#f87171)' };
}

function ResultDisplay({ result, loading }) {

  /* ── Skeleton ─────────────────────────────────────────────────────────── */
  if (loading) {
    return (
      <div className="result-card">
        <div className="result-card__accent" />
        <div className="result-card__header">
          <h3>Analysing Image&hellip;</h3>
        </div>
        <div className="skeleton skeleton-hero" />
        <div className="skeleton skeleton-conf" />
        <div className="result-card__body">
          {[70, 85, 60, 75, 50, 65, 45, 55].map((w, i) => (
            <div key={i} className="skeleton skeleton-line" style={{ width: `${w}%`, marginBottom: '0.6rem' }} />
          ))}
        </div>
      </div>
    );
  }

  if (!result) return null;

  const { raw_key, prediction, confidence_percentage, all_predictions } = result;
  const amount = formatAmount(raw_key || prediction);
  const level  = confidenceLevel(confidence_percentage);

  return (
    <div className="result-card">
      {/* Rainbow accent bar */}
      <div className="result-card__accent" />

      {/* Header */}
      <div className="result-card__header">
        <h3>Recognition Result</h3>
        <span className="result-tag" style={{ color: level.color, borderColor: level.color }}>
          {level.label} Confidence
        </span>
      </div>

      {/* Big denomination display */}
      <div className="result-denom-hero">
        <p className="result-denom-label">Identified Denomination</p>
        <p className="result-denom-value">GH&#8373; {amount}</p>
        <p className="result-denom-name">{getDenomName(raw_key || prediction)}</p>
      </div>

      {/* Confidence meter */}
      <div className="result-confidence">
        <div className="result-confidence-top">
          <span>Confidence Score</span>
          <span className="result-confidence-pct" style={{ color: level.color }}>
            {confidence_percentage.toFixed(1)}%
          </span>
        </div>
        <div className="result-meter-track">
          <div
            className="result-meter-fill"
            style={{
              width: `${confidence_percentage}%`,
              background: level.meter,
            }}
          />
        </div>
      </div>

      {/* All class probabilities */}
      <div className="result-card__body">
        <div className="result-all-header">
          <h4>All Denomination Probabilities</h4>
          <span className="result-count">{all_predictions.length} classes</span>
        </div>

        <div className="result-bars">
          {all_predictions.map((item, idx) => (
            <ConfidenceBar
              key={item.raw_key || item.denomination}
              denomination={`GH₵ ${formatAmount(item.raw_key) || item.denomination}`}
              percentage={item.percentage}
              isTop={idx === 0}
              rank={idx + 1}
            />
          ))}
        </div>

        {confidence_percentage < 70 && (
          <div className="result-tip">
            <strong>Tip:</strong> For better accuracy, make sure the banknote fills the frame,
            the image is well-lit, and not blurry or cropped.
          </div>
        )}
      </div>
    </div>
  );
}

export default ResultDisplay;
