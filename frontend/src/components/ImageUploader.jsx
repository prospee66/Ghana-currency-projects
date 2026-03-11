import React, { useRef, useState, useCallback } from 'react';
import './ImageUploader.css';

const ACCEPTED_TYPES = ['image/jpeg', 'image/png', 'image/bmp', 'image/gif', 'image/webp'];
const MAX_SIZE_MB    = 16;

function ImageUploader({ onImageSelect, imageUrl, loading, onPredict, onReset, hasImage }) {
  const fileInputRef  = useRef(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState(null);

  // ── File validation ────────────────────────────────────────────────────────
  const validateAndSelect = useCallback((file) => {
    setFileError(null);

    if (!file) return;

    if (!ACCEPTED_TYPES.includes(file.type)) {
      setFileError('Unsupported file type. Please upload a JPG, PNG, BMP, GIF, or WebP image.');
      return;
    }

    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      setFileError(`File too large. Maximum size is ${MAX_SIZE_MB} MB.`);
      return;
    }

    const url = URL.createObjectURL(file);
    onImageSelect(file, url);
  }, [onImageSelect]);

  // ── Drag-and-drop handlers ─────────────────────────────────────────────────
  const onDragOver  = (e) => { e.preventDefault(); setDragging(true);  };
  const onDragLeave = (e) => { e.preventDefault(); setDragging(false); };
  const onDrop      = (e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files?.[0];
    validateAndSelect(file);
  };

  // ── Click-to-browse ────────────────────────────────────────────────────────
  const onFileChange = (e) => validateAndSelect(e.target.files?.[0]);
  const openBrowser  = () => { if (!loading) fileInputRef.current?.click(); };

  // ── Keyboard a11y ──────────────────────────────────────────────────────────
  const onKeyDown = (e) => { if (e.key === 'Enter' || e.key === ' ') openBrowser(); };

  return (
    <div className="uploader-card">
      <div className="uploader-card__header">
        <h3>Upload Currency Note</h3>
        <p>Drag &amp; drop an image or click to browse</p>
      </div>

      <div className="uploader-card__body">

        {/* ── Drop zone ────────────────────────────────────────────────────── */}
        {!hasImage && (
          <div
            className={`drop-zone ${dragging ? 'drop-zone--active' : ''}`}
            onDragOver={onDragOver}
            onDragLeave={onDragLeave}
            onDrop={onDrop}
            onClick={openBrowser}
            onKeyDown={onKeyDown}
            role="button"
            tabIndex={0}
            aria-label="Upload currency image"
          >
            <div className="drop-zone__icon">
              <svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="#94a3b8" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                <polyline points="17 8 12 3 7 8"/>
                <line x1="12" y1="3" x2="12" y2="15"/>
              </svg>
            </div>
            <p className="drop-zone__primary">Drop your image here</p>
            <p className="drop-zone__secondary">or click to browse</p>
            <p className="drop-zone__hint">JPG, PNG, BMP, WebP &mdash; max {MAX_SIZE_MB} MB</p>

            {dragging && <div className="drop-zone__overlay">Release to upload</div>}
          </div>
        )}

        {/* ── Preview ──────────────────────────────────────────────────────── */}
        {hasImage && imageUrl && (
          <div className="preview-wrap">
            <img
              src={imageUrl}
              alt="Selected currency note"
              className="preview-img"
            />
            <div className="preview-badge">Ready for analysis</div>
          </div>
        )}

        {/* ── Validation error ─────────────────────────────────────────────── */}
        {fileError && (
          <p className="file-error" role="alert">{fileError}</p>
        )}

        {/* ── Hidden file input ─────────────────────────────────────────────── */}
        <input
          ref={fileInputRef}
          type="file"
          accept={ACCEPTED_TYPES.join(',')}
          className="hidden-input"
          onChange={onFileChange}
          aria-hidden="true"
        />
      </div>

      {/* ── Action buttons ─────────────────────────────────────────────────── */}
      <div className="uploader-card__footer">
        {!hasImage ? (
          <button className="btn btn-primary btn-full" onClick={openBrowser}>
            Choose Image
          </button>
        ) : (
          <div className="btn-row">
            <button
              className="btn btn-outline"
              onClick={onReset}
              disabled={loading}
            >
              Clear
            </button>
            <button
              className={`btn btn-primary ${loading ? 'btn-loading' : ''}`}
              onClick={onPredict}
              disabled={loading}
            >
              {loading ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  Analysing&hellip;
                </>
              ) : (
                'Identify Note'
              )}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

export default ImageUploader;
