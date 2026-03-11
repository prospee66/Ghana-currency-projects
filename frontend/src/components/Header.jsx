import React from 'react';
import './Header.css';

function Header() {
  return (
    <header className="header">
      {/* Ghana-flag colour bar at very top */}
      <div className="flag-stripe">
        <div className="stripe red"   />
        <div className="stripe gold"  />
        <div className="stripe green" />
      </div>

      <div className="header-inner">
        {/* Brand */}
        <div className="brand">
          <div className="logo-wrap">
            <span className="logo-star">GH</span>
          </div>
          <div className="brand-text">
            <h1 className="brand-title">Ghana Currency Recognizer</h1>
            <p className="brand-sub">AI-Powered Banknote Identification</p>
          </div>
        </div>

        {/* Pill badges */}
        <div className="badges">
          <span className="badge badge-green">PyTorch</span>
          <span className="badge badge-red">Flask API</span>
          <span className="badge badge-gold">React</span>
        </div>
      </div>
    </header>
  );
}

export default Header;
