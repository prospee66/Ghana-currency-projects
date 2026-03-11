import React, { useState } from 'react';
import Header from './components/Header';
import ImageUploader from './components/ImageUploader';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [image,    setImage]    = useState(null);
  const [imageUrl, setImageUrl] = useState(null);
  const [result,   setResult]   = useState(null);
  const [loading,  setLoading]  = useState(false);
  const [error,    setError]    = useState(null);

  const handleImageSelect = (file, url) => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImage(file);
    setImageUrl(url);
    setResult(null);
    setError(null);
  };

  const handlePredict = async () => {
    if (!image) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const body = new FormData();
      body.append('file', image);
      const res  = await fetch(`${API_URL}/predict`, { method: 'POST', body });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `Server error ${res.status}`);
      setResult(data);
    } catch (err) {
      setError(
        err.message.includes('fetch')
          ? 'Could not reach the server. Make sure Flask is running on port 5000.'
          : err.message
      );
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    if (imageUrl) URL.revokeObjectURL(imageUrl);
    setImage(null);
    setImageUrl(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <Header />

      {/* Hero banner */}
      <section className="hero">
        <div className="hero-inner">
          <div className="hero-label">AI-Powered Recognition</div>
          <h2 className="hero-title">Identify <span>Ghanaian Banknotes</span> Instantly</h2>
          <p className="hero-sub">
            Upload any photo of a Ghana Cedi note — our deep learning model
            identifies the denomination in under a second.
          </p>
        </div>
      </section>

      <main className="main-content">
        <div className="container">

          {/* Main workspace — two-column on desktop */}
          <div className="workspace">
            <div className="workspace-left">
              <ImageUploader
                onImageSelect={handleImageSelect}
                imageUrl={imageUrl}
                loading={loading}
                onPredict={handlePredict}
                onReset={handleReset}
                hasImage={!!image}
              />
              {error && (
                <div className="error-banner" role="alert">
                  <span className="error-icon">!</span>
                  <span>{error}</span>
                </div>
              )}
            </div>

            <div className="workspace-right">
              {(loading || result) ? (
                <ResultDisplay result={result} loading={loading} />
              ) : (
                <div className="placeholder-panel">
                  <div className="placeholder-icon"></div>
                  <h3>No Image Yet</h3>
                  <p>Upload a banknote photo on the left to see the AI prediction here.</p>
                </div>
              )}
            </div>
          </div>

          {/* Stats row */}
          <div className="stats-row">
            <div className="stat-item">
              <span className="stat-number">8</span>
              <span className="stat-label">Denominations</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">85–95%</span>
              <span className="stat-label">Accuracy</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">&lt;1s</span>
              <span className="stat-label">Response Time</span>
            </div>
            <div className="stat-divider" />
            <div className="stat-item">
              <span className="stat-number">CNN</span>
              <span className="stat-label">MobileNetV2</span>
            </div>
          </div>

          {/* Feature cards */}
          <div className="features-section">
            <h3 className="features-title">Why Use This System?</h3>
            <div className="feature-cards">
              <div className="feature-card">
                <div className="feature-icon-wrap green">ML</div>
                <h4>Deep Learning</h4>
                <p>MobileNetV2 CNN trained with transfer learning on real Ghana Cedi images</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon-wrap gold">95%</div>
                <h4>High Accuracy</h4>
                <p>85–95% accuracy under different lighting, angles and distances</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon-wrap red">A11y</div>
                <h4>Accessibility</h4>
                <p>Empowers visually impaired users to identify banknotes independently</p>
              </div>
              <div className="feature-card">
                <div className="feature-icon-wrap green">&lt;1s</div>
                <h4>Real-Time</h4>
                <p>Flask REST API delivers predictions in under one second</p>
              </div>
            </div>
          </div>

        </div>
      </main>

      <footer className="footer">
        <div className="footer-content">
          <div className="footer-brand">
            <span>Ghana Currency Recognition System</span>
          </div>
          <div className="footer-flag">
            <span style={{ background: '#CE1126' }} />
            <span style={{ background: '#FCD116' }} />
            <span style={{ background: '#006B3F' }} />
          </div>
          <p className="footer-copy">
            &copy; {new Date().getFullYear()} &nbsp;|&nbsp; Built with React &amp; PyTorch
          </p>
        </div>
      </footer>
    </div>
  );
}

export default App;
